/* SPDX-License-Identifier: LGPL-2.1-or-later */
#include <math.h>
#include <string.h>

#include "common/msg.h"
#include "demux/demux.h"
#include "demux/stheader.h"
#include "mpv_talloc.h"
#include "osdep/timer.h"
#include "stream/stream.h"
#include "video/out/display_rate.h"
#include "core.h"

// Called on the existing opener thread, before playback/prefetch starts.
// A separate, unbuffered demuxer leaves the real stream and resume position
// untouched. Only timestamps are retained; no frames are decoded.
void mp_probe_display_rates(struct demuxer *demux, int stream_flags)
{
    if (!demux->seekable || demux->partially_seekable || demux->is_streaming ||
        demux->is_network || demux->ts_resets_possible ||
        (strcmp(demux->desc->name, "mkv") && strcmp(demux->desc->name, "lavf")))
        return;
    char *path = mp_file_get_path(NULL, bstr0(demux->filename));
    bool local = path != NULL;
    talloc_free(path);
    if (!local)
        return;

    int num = demux_get_num_stream(demux), videos = 0;
    for (int n = 0; n < num; n++) {
        struct sh_stream *sh = demux_get_stream(demux, n);
        videos += sh->type == STREAM_VIDEO && !sh->image && !sh->still_image;
    }
    if (!videos)
        return;

    // Limit additional startup work. An incomplete scan is never called CFR:
    // playback will use a single higher refresh instead of guessing and then
    // changing it mid-episode. Cancellation uses the normal opener cancel tree.
    double deadline = mp_time_sec() + 5;
    struct demuxer_params params = {
        .force_format = (char *)demux->desc->name,
        .stream_flags = stream_flags,
        .disable_timeline = true,
        // is_top_level=false disables the packet cache, even with cache=yes.
    };
    struct demuxer *probe = demux_open_url(demux->filename, &params,
                                          demux->cancel, demux->global);
    if (!probe)
        return;

    struct samples {
        double *pts;
        int count;
        bool selected, invalid;
    } *samples = talloc_zero_array(NULL, struct samples, num);
    for (int n = 0; n < demux_get_num_stream(probe); n++) {
        struct sh_stream *sh = demux_get_stream(probe, n);
        struct sh_stream *orig = n < num ? demux_get_stream(demux, n) : NULL;
        bool selected = orig && sh->type == STREAM_VIDEO && !sh->image &&
                        !sh->still_image && !sh->missing_timestamps &&
                        orig->type == sh->type && orig->demuxer_id == sh->demuxer_id;
        demuxer_select_track(probe, sh, MP_NOPTS_VALUE, selected);
        if (selected)
            samples[n].selected = true;
    }

    bool complete = false;
    int total = 0;
    while (!demux_cancel_test(probe) && mp_time_sec() < deadline && total < 1000000) {
        struct demux_packet *pkt = demux_read_any_packet(probe);
        if (!pkt) {
            complete = !demux_read_interrupted(probe);
            break;
        }
        total++;
        if (pkt->stream >= 0 && pkt->stream < num && samples[pkt->stream].selected) {
            struct samples *s = &samples[pkt->stream];
            if (pkt->pts == MP_NOPTS_VALUE || !isfinite(pkt->pts) || pkt->segmented) {
                s->invalid = true;
            } else {
                MP_TARRAY_APPEND(samples, s->pts, s->count, pkt->pts);
            }
        }
        talloc_free(pkt);
    }
    if (complete && !demux_cancel_test(probe)) {
        for (int n = 0; n < num; n++) {
            struct samples *s = &samples[n];
            if (s->selected && !s->invalid) {
                struct sh_stream *sh = demux_get_stream(demux, n);
                struct mp_display_cadence *cadences =
                    talloc_array(sh, struct mp_display_cadence, s->count);
                int runs = mp_display_rate_analyze(s->pts, s->count, cadences);
                if (runs) {
                    sh->display_cadences = talloc_realloc(sh, cadences,
                                                        struct mp_display_cadence, runs);
                    sh->num_display_cadences = runs;
                    sh->whole_file_fps = runs == 1 ? 1 / sh->display_cadences[0].interval : 0;
                } else {
                    talloc_free(cadences);
                }
                MP_VERBOSE(demux, "Display refresh scan: track %d, %d timestamps, %s (%.3f fps, %d cadence runs).\n",
                           sh->demuxer_id, s->count,
                           sh->whole_file_fps > 0 ? "CFR" : runs ? "VFR" : "unknown",
                           sh->whole_file_fps, runs);
            }
        }
    } else {
        MP_VERBOSE(demux, "Display refresh scan incomplete; keeping a higher refresh for this file.\n");
    }
    talloc_free(samples);
    demux_cancel_and_free(probe);
}
