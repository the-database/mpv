/*
 * Subtitle render-ahead worker.
 *
 * Renders ASS subtitle bitmaps for upcoming frames on a background thread so the
 * VO display path serves a pre-rendered frame instead of running the ~33ms
 * libass render + atlas pack inline (which, stacked with the GPU upload, blows
 * the per-frame budget at 8K).
 *
 * Design (this is a rebuild; the first attempt was reverted for the two reasons
 * called out below):
 *
 * 1. The worker owns a SECOND, fully independent sd_ass instance (`worker_sd`)
 *    with its own ASS_Library/Renderer/Track. It renders that, so it NEVER
 *    takes the decoder lock (`sub->lock`) and never touches the decoder's track.
 *    -> fixes failure #1 (the old worker held sub->lock across the render, and
 *    sub_read_packets takes sub->lock every frame, so the decode path stalled).
 *    The own ASS_Library also means no font-cache race with the VO's fallback
 *    renderer (font_lock/fontselect are per-library).
 *
 * 2. The worker's libass thread count is capped separately
 *    (--sub-render-ahead-threads) so it does not starve the VO/upscaler.
 *    -> fixes failure #2 (drops scaled up with the old worker's thread count).
 *
 * 3. No `sub_pts <= last_pkt_pts` gate (it busy-spun on long-lived events); the
 *    worker waits on a cond when its window is full.
 *
 * Locking: `lock` protects the ring, the cursor/geometry, the gen, the timing
 * snapshot, the packet queue, and the pending video-params. It is never held
 * across a render. The decode path only ever takes `lock` briefly to enqueue a
 * packet copy. All mutation of `worker_sd` happens on the worker thread.
 */

#include <math.h>
#include <stdlib.h>

#include "common/msg.h"
#include "dec_sub.h"
#include "demux/packet.h"
#include "misc/mp_assert.h"
#include "osd.h"
#include "osdep/threads.h"
#include "sd.h"
#include "sub_ahead.h"
#include "video/mp_image.h"

struct sub_ahead_entry {
    bool valid;
    double video_pts;            // raw (pre-conversion) video pts -- the key
    struct mp_osd_res dim;
    int format;
    uint64_t gen;
    uint64_t content_id;         // for change_id unification
    struct sub_bitmaps *bmp;     // owned; NULL means "no subtitles at this pts"
};

struct sub_ahead {
    struct mp_log *log;
    struct sd *worker_sd;        // owned: the independent libass instance

    mp_thread thread;
    bool worker_running;
    mp_mutex lock;
    mp_cond wakeup;
    bool terminate;

    int depth;                   // lookahead frames
    double video_fps;
    double vo_pts;               // latest raw video pts the VO asked for
    struct mp_osd_res cur_dim;
    int cur_format;
    uint64_t gen;                // bumped on any invalidation

    struct sub_ahead_entry *ring;
    int ring_len;

    // pts mapping snapshot (mirrors dec_sub.c pts_to_subtitle), so the worker
    // converts target video pts -> subtitle pts without the decoder lock.
    double sub_speed;
    float delay;
    int play_dir;

    // packet queue: raw event packet copies for the worker to decode.
    struct demux_packet **queue;
    int num_queue;

    // pending controls to apply to worker_sd on the worker thread.
    bool params_pending;
    bool params_set;             // pending_params holds a real value to compare
    struct mp_image_params pending_params;
    bool reset_pending;

    // change_id unification across worker hits (osd.c accumulates change_id).
    uint64_t content_counter;
    uint64_t served_change_id;
    uint64_t last_served_content;
    bool have_last_served;

    int inline_active;           // VO is inline-rendering; worker must not race it

    // WP-A3 permanent render-ahead health counters (re-added; formerly TEMP).
    // Cheap integer increments, all touched under `lock`. served = pre-rendered
    // hit; empty = rendered-but-no-subs (cheap inline); miss = true miss ->
    // expensive inline render. Surfaced as MP_STATS ra-* value lines + a
    // periodic MSGL_V [render-ahead] line (format kept parse_stats.py-compatible).
    long dbg_served, dbg_empty, dbg_miss;
};

static double ahead_pts_to_subtitle(struct sub_ahead *a, double pts)
{
    if (pts == MP_NOPTS_VALUE)
        return pts;
    double speed = a->sub_speed != 0 ? a->sub_speed : 1.0;
    return (pts * a->play_dir - a->delay) / speed;
}

static void ahead_clear(struct sub_ahead *a)
{
    for (int n = 0; n < a->ring_len; n++) {
        if (a->ring[n].valid) {
            talloc_free(a->ring[n].bmp);
            a->ring[n] = (struct sub_ahead_entry){0};
        }
    }
}

static void queue_flush(struct sub_ahead *a)
{
    for (int n = 0; n < a->num_queue; n++)
        talloc_free(a->queue[n]);
    a->num_queue = 0;
}

// Index of a current-gen entry for raw video pts V at the given geometry, or -1.
// V is matched within half a frame interval.
static int ahead_find(struct sub_ahead *a, double V, struct mp_osd_res dim,
                      int format)
{
    double tol = (a->video_fps > 0 ? 1.0 / a->video_fps : 1.0 / 24.0) * 0.5;
    for (int n = 0; n < a->ring_len; n++) {
        struct sub_ahead_entry *e = &a->ring[n];
        if (e->valid && e->gen == a->gen && e->format == format &&
            osd_res_equals(e->dim, dim) && fabs(e->video_pts - V) < tol)
            return n;
    }
    return -1;
}

// Store a freshly rendered frame, taking ownership of bmp. Reuses a free slot,
// else evicts the oldest (smallest) video pts.
static void ahead_store(struct sub_ahead *a, double V, struct mp_osd_res dim,
                        int format, uint64_t gen, uint64_t content_id,
                        struct sub_bitmaps *bmp)
{
    int slot = -1;
    double oldest = INFINITY;
    for (int n = 0; n < a->ring_len; n++) {
        if (!a->ring[n].valid) {
            slot = n;
            break;
        }
        if (a->ring[n].video_pts < oldest) {
            oldest = a->ring[n].video_pts;
            slot = n;
        }
    }
    if (a->ring[slot].valid)
        talloc_free(a->ring[slot].bmp);
    a->ring[slot] = (struct sub_ahead_entry){
        .valid = true, .video_pts = V, .dim = dim, .format = format,
        .gen = gen, .content_id = content_id, .bmp = bmp,
    };
}

static MP_THREAD_VOID sub_ahead_thread(void *ptr)
{
    struct sub_ahead *a = ptr;
    mp_thread_set_name("sub/ahead");

    mp_mutex_lock(&a->lock);
    while (!a->terminate) {
        // Reset the worker's own track (seek) on this thread.
        if (a->reset_pending) {
            a->reset_pending = false;
            mp_mutex_unlock(&a->lock);
            if (a->worker_sd->driver->reset)
                a->worker_sd->driver->reset(a->worker_sd);
            mp_mutex_lock(&a->lock);
            continue;
        }

        // Apply any pending video params on this thread (worker_sd is only ever
        // touched here).
        if (a->params_pending) {
            struct mp_image_params p = a->pending_params;
            a->params_pending = false;
            mp_mutex_unlock(&a->lock);
            a->worker_sd->driver->control(a->worker_sd, SD_CTRL_SET_VIDEO_PARAMS, &p);
            mp_mutex_lock(&a->lock);
            continue;
        }

        // Drain queued packets into the worker's own track.
        if (a->num_queue) {
            struct demux_packet *pkt = a->queue[0];
            MP_TARRAY_REMOVE_AT(a->queue, a->num_queue, 0);
            mp_mutex_unlock(&a->lock);
            a->worker_sd->driver->decode(a->worker_sd, pkt);
            talloc_free(pkt);
            mp_mutex_lock(&a->lock);
            continue;
        }

        // Pause while the VO is inline-rendering a missed frame, so the two
        // heavy parallel renders don't oversaturate the cores (which turned one
        // miss into a ~250ms stall and a burst of drops).
        if (a->inline_active) {
            mp_cond_wait(&a->wakeup, &a->lock);
            continue;
        }

        double vo = a->vo_pts;
        int depth = a->depth;
        if (depth <= 0 || vo == MP_NOPTS_VALUE || a->cur_format == 0) {
            mp_cond_wait(&a->wakeup, &a->lock);
            continue;
        }
        struct mp_osd_res dim = a->cur_dim;
        int format = a->cur_format;
        uint64_t gen = a->gen;
        double interval = a->video_fps > 0 ? 1.0 / a->video_fps : 1.0 / 24.0;

        // Nearest upcoming frame not yet in the ring. Start at i=1 (the NEXT
        // frame), never the current one: if the current frame is a miss, the VO
        // renders it inline, and the worker must not render it too -- otherwise
        // both render the same heavy frame at once, the cores oversaturate, and
        // one miss balloons into a multi-frame stall. Skipping it keeps a miss a
        // single inline frame while the worker races ahead to refill.
        double target = MP_NOPTS_VALUE;
        for (int i = 1; i <= depth; i++) {
            double V = vo + i * interval;
            if (ahead_find(a, V, dim, format) < 0) {
                target = V;
                break;
            }
        }
        if (target == MP_NOPTS_VALUE) {
            // Window full; wait for the VO to advance (or a flush).
            mp_cond_wait(&a->wakeup, &a->lock);
            continue;
        }

        double sub_pts = ahead_pts_to_subtitle(a, target);
        mp_mutex_unlock(&a->lock);

        // Render on the worker's own sd -- no decoder lock involved. draw_flags=0
        // (no OSD_DRAW_SUB_ONLY) so --sub-render-res-limit applies to the worker's
        // renders exactly as it does on the inline VO path.
        struct sub_bitmaps *bmp =
            a->worker_sd->driver->get_bitmaps(a->worker_sd, dim, format, sub_pts, 0);
        uint64_t content_id = bmp ? bmp->change_id : 0;

        mp_mutex_lock(&a->lock);
        if (gen == a->gen && format == a->cur_format &&
            osd_res_equals(dim, a->cur_dim))
        {
            ahead_store(a, target, dim, format, gen, content_id, bmp);
        } else {
            talloc_free(bmp);    // stale (flush/resize during render)
        }
    }
    mp_mutex_unlock(&a->lock);
    MP_THREAD_RETURN();
}

struct sub_ahead *sub_ahead_create(struct dec_sub *sub, struct sd *worker_sd,
                                   int depth, int order)
{
    if (depth <= 0 || !worker_sd || !worker_sd->driver->get_bitmaps)
        return NULL;

    struct sub_ahead *a = talloc_zero(NULL, struct sub_ahead);
    a->log = worker_sd->log;
    a->worker_sd = worker_sd;
    a->depth = depth;
    a->vo_pts = MP_NOPTS_VALUE;
    a->cur_format = 0;
    a->sub_speed = 1.0;
    a->play_dir = 1;
    a->video_fps = 0;
    a->ring_len = depth + 2;
    a->ring = talloc_zero_array(a, struct sub_ahead_entry, a->ring_len);

    mp_mutex_init(&a->lock);
    mp_cond_init(&a->wakeup);

    if (mp_thread_create(&a->thread, sub_ahead_thread, a) != 0) {
        MP_WARN(a, "Failed to start render-ahead worker.\n");
        mp_cond_destroy(&a->wakeup);
        mp_mutex_destroy(&a->lock);
        talloc_free(a);
        return NULL;
    }
    a->worker_running = true;
    return a;
}

void sub_ahead_destroy(struct sub_ahead *a)
{
    if (!a)
        return;
    if (a->worker_running) {
        mp_mutex_lock(&a->lock);
        a->terminate = true;
        mp_cond_signal(&a->wakeup);
        mp_mutex_unlock(&a->lock);
        mp_thread_join(a->thread);
    }
    MP_VERBOSE(a, "[render-ahead] served=%ld empty=%ld miss=%ld\n",
               a->dbg_served, a->dbg_empty, a->dbg_miss);
    queue_flush(a);
    ahead_clear(a);
    if (a->worker_sd) {
        a->worker_sd->driver->uninit(a->worker_sd);
        talloc_free(a->worker_sd);
    }
    mp_cond_destroy(&a->wakeup);
    mp_mutex_destroy(&a->lock);
    talloc_free(a);
}

void sub_ahead_enqueue(struct sub_ahead *a, struct demux_packet *pkt)
{
    if (!a || !pkt)
        return;
    struct demux_packet *copy = demux_copy_packet(NULL, pkt);
    if (!copy)
        return;
    mp_mutex_lock(&a->lock);
    MP_TARRAY_APPEND(a, a->queue, a->num_queue, copy);
    mp_cond_signal(&a->wakeup);
    mp_mutex_unlock(&a->lock);
}

void sub_ahead_flush(struct sub_ahead *a)
{
    if (!a)
        return;
    mp_mutex_lock(&a->lock);
    queue_flush(a);
    ahead_clear(a);
    a->gen++;
    a->reset_pending = true;     // flush the worker's own track (on its thread)
    a->vo_pts = MP_NOPTS_VALUE;
    mp_cond_signal(&a->wakeup);
    mp_mutex_unlock(&a->lock);
}

void sub_ahead_set_timing(struct sub_ahead *a, double sub_speed, float delay,
                          int play_dir, double video_fps)
{
    if (!a)
        return;
    mp_mutex_lock(&a->lock);
    bool changed = a->sub_speed != sub_speed || a->delay != delay ||
                   a->play_dir != play_dir || a->video_fps != video_fps;
    a->sub_speed = sub_speed;
    a->delay = delay;
    a->play_dir = play_dir;
    a->video_fps = video_fps;
    if (changed) {
        ahead_clear(a);          // stale-delay/speed entries must not be served
        a->gen++;
    }
    mp_cond_signal(&a->wakeup);
    mp_mutex_unlock(&a->lock);
}

void sub_ahead_set_video_params(struct sub_ahead *a,
                                const struct mp_image_params *params)
{
    if (!a || !params)
        return;
    mp_mutex_lock(&a->lock);
    // This is sent every frame; only invalidate when the params actually change
    // (otherwise the ring is wiped every frame and the worker never gets ahead).
    if (!a->params_set || !mp_image_params_equal(&a->pending_params, params)) {
        a->pending_params = *params;
        a->params_pending = true;
        a->params_set = true;
        ahead_clear(a);          // colorspace may change -> drop stale entries
        a->gen++;
        mp_cond_signal(&a->wakeup);
    }
    mp_mutex_unlock(&a->lock);
}

void sub_ahead_inline_begin(struct sub_ahead *a)
{
    if (!a)
        return;
    mp_mutex_lock(&a->lock);
    a->inline_active++;
    mp_mutex_unlock(&a->lock);
}

void sub_ahead_inline_end(struct sub_ahead *a)
{
    if (!a)
        return;
    mp_mutex_lock(&a->lock);
    a->inline_active--;
    mp_cond_signal(&a->wakeup);   // let the worker resume
    mp_mutex_unlock(&a->lock);
}

struct sub_bitmaps *sub_ahead_get_bitmaps(struct sub_ahead *a,
                                          struct mp_osd_res dim, int format,
                                          double raw_video_pts)
{
    if (!a)
        return NULL;
    mp_mutex_lock(&a->lock);
    a->vo_pts = raw_video_pts;
    if (a->cur_format != format || !osd_res_equals(a->cur_dim, dim)) {
        a->cur_dim = dim;
        a->cur_format = format;
        a->gen++;
        ahead_clear(a);
    }
    int idx = ahead_find(a, raw_video_pts, dim, format);
    // WP-A3 render-ahead counters + telemetry (no-op unless stats/-v active).
    if (idx < 0)               a->dbg_miss++;
    else if (a->ring[idx].bmp) a->dbg_served++;
    else                       a->dbg_empty++;
    MP_STATS(a, "value %ld ra-served", a->dbg_served);
    MP_STATS(a, "value %ld ra-empty", a->dbg_empty);
    MP_STATS(a, "value %ld ra-miss", a->dbg_miss);
    if (a->dbg_served && a->dbg_served % 250 == 0)
        MP_VERBOSE(a, "[render-ahead] served=%ld empty=%ld miss=%ld\n",
                   a->dbg_served, a->dbg_empty, a->dbg_miss);
    struct sub_bitmaps *res = NULL;
    if (idx >= 0 && a->ring[idx].bmp) {
        res = sub_bitmaps_copy(NULL, a->ring[idx].bmp);
        // Unified monotonic change_id: report a fresh id only when the content
        // differs from what we last served, else 0 (osd.c treats 0 as "no
        // change, no re-upload").
        uint64_t cid = a->ring[idx].content_id;
        if (!a->have_last_served || cid != a->last_served_content) {
            a->last_served_content = cid;
            a->have_last_served = true;
            res->change_id = ++a->served_change_id;
        } else {
            res->change_id = 0;
        }
    }
    // idx>=0 with NULL bmp == "rendered, no subtitles here": fall back to inline
    // (cheap when there are no events). A true miss also returns NULL.
    mp_mutex_unlock(&a->lock);
    mp_cond_signal(&a->wakeup);  // nudge the worker to refill ahead of us
    return res;
}
