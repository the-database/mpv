/*
 * Native metadata probe for AJN's supervised media worker.
 * SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * Private ABI: callbacks read only the host-selected stream. No path, URL or
 * secondary protocol is opened by this helper. Stream discovery may decode a
 * bounded prefix; no playback, upscaling or encoding session is started.
 */
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/display.h>
#include <libavutil/pixdesc.h>
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/opt.h>

#include "include/mpv/client.h"
#include "common/ajn_probe.h"
#include "misc/json.h"
#include "misc/node.h"
#include "ta/ta_talloc.h"

static int deny_open(AVFormatContext *ctx, AVIOContext **pb, const char *url,
                     int flags, AVDictionary **options)
{
    return AVERROR(EACCES);
}

MPV_EXPORT int mpv_ajn_encoder_check_v1(const char *name, const char *options,
    int width, int height, int bit_depth, double fps)
{
    if (!name || !options || strlen(options) > 1024 || width < 2 || height < 2 ||
        width > 8192 || height > 8192 || !isfinite(fps) || fps <= 0 || fps > 480 ||
        (bit_depth != 8 && bit_depth != 10))
        return AVERROR(EINVAL);
    if (strcmp(name, "h264_nvenc") && strcmp(name, "hevc_nvenc") &&
        strcmp(name, "av1_nvenc") && strcmp(name, "h264_amf") && strcmp(name, "hevc_amf"))
        return AVERROR(EINVAL);
    const AVCodec *codec = avcodec_find_encoder_by_name(name);
    if (!codec)
        return AVERROR_ENCODER_NOT_FOUND;
    AVCodecContext *ctx = avcodec_alloc_context3(codec);
    if (!ctx)
        return AVERROR(ENOMEM);
    ctx->width = width;
    ctx->height = height;
    ctx->pix_fmt = bit_depth == 8 ? AV_PIX_FMT_NV12 : AV_PIX_FMT_P010LE;
    ctx->framerate = av_d2q(fps, 1000000);
    ctx->time_base = av_inv_q(ctx->framerate);
    AVDictionary *opts = NULL;
    int ret = av_dict_parse_string(&opts, options, "=", ",", 0);
    if (ret >= 0)
        ret = avcodec_open2(ctx, codec, &opts);
    // An unused option would make this readiness test differ from playback.
    if (ret >= 0 && av_dict_count(opts))
        ret = AVERROR_OPTION_NOT_FOUND;
    av_dict_free(&opts);
    avcodec_free_context(&ctx);
    return ret;
}

static void optional_string(mpv_node *node, const char *key, const char *value)
{
    if (value && strlen(value) <= 4096)
        node_map_add_string(node, key, value);
    else
        node_map_add(node, key, MPV_FORMAT_NONE);
}

static void metadata(mpv_node *node, const char *key, AVDictionary *tags,
                     const char *tag)
{
    AVDictionaryEntry *entry = av_dict_get(tags, tag, NULL, 0);
    optional_string(node, key, entry ? entry->value : NULL);
}

static void seconds(mpv_node *node, const char *key, int64_t ticks,
                    AVRational time_base)
{
    double value = ticks == AV_NOPTS_VALUE ? NAN : ticks * av_q2d(time_base);
    if (isfinite(value))
        node_map_add_double(node, key, value);
    else
        node_map_add(node, key, MPV_FORMAT_NONE);
}

static void rational(mpv_node *node, const char *key, AVRational value)
{
    if (value.den <= 0 || value.num <= 0) {
        node_map_add(node, key, MPV_FORMAT_NONE);
        return;
    }
    mpv_node *ratio = node_map_add(node, key, MPV_FORMAT_NODE_MAP);
    node_map_add_int64(ratio, "numerator", value.num);
    node_map_add_int64(ratio, "denominator", value.den);
}

static void positive_integer(mpv_node *node, const char *key, int value)
{
    if (value > 0) node_map_add_int64(node, key, value);
    else node_map_add(node, key, MPV_FORMAT_NONE);
}

MPV_EXPORT int mpv_ajn_probe_v1(void *opaque,
    int (*read_packet)(void *, uint8_t *, int),
    int64_t (*seek)(void *, int64_t, int), int (*cancel)(void *),
    int can_seek, char **json)
{
    if (!opaque || !read_packet || !cancel || !json)
        return AVERROR(EINVAL);
    *json = NULL;
    int result = AVERROR(ENOMEM);
    uint8_t *buffer = av_malloc(32768);
    AVIOContext *io = buffer ? avio_alloc_context(buffer, 32768, 0, opaque,
                                                 read_packet, NULL, seek) : NULL;
    AVFormatContext *format = NULL;
    AVDictionary *options = NULL;
    mpv_node root = {0};
    if (!io) { av_free(buffer); return result; }
    io->seekable = can_seek ? AVIO_SEEKABLE_NORMAL : 0;
    format = avformat_alloc_context();
    if (!format) goto done;
    format->pb = io;
    format->flags |= AVFMT_FLAG_CUSTOM_IO;
    format->io_open = deny_open;
    format->interrupt_callback = (AVIOInterruptCB){cancel, opaque};
    format->max_streams = 128;
    av_dict_set(&options, "format_whitelist", "matroska,webm,mov,avi,mpegts,srt,ass,webvtt", 0);
    av_dict_set(&options, "protocol_whitelist", "", 0);
    av_dict_set(&options, "probesize", "8388608", 0);
    av_dict_set(&options, "analyzeduration", "5000000", 0);
    result = avformat_open_input(&format, NULL, NULL, &options);
    if (result < 0) goto done;
    result = avformat_find_stream_info(format, NULL);
    if (result < 0) goto done;
    if (format->nb_streams > 128 || format->nb_chapters > 512) {
        result = AVERROR(EFBIG);
        goto done;
    }
    node_init(&root, MPV_FORMAT_NODE_MAP, NULL);
    node_map_add_int64(&root, "probeAbi", 1);
    optional_string(&root, "container", format->iformat->name);
    seconds(&root, "durationSeconds", format->duration, AV_TIME_BASE_Q);
    seconds(&root, "startSeconds", format->start_time, AV_TIME_BASE_Q);
    node_map_add_flag(&root, "seekable", can_seek != 0);
    node_map_add(&root, "tracks", MPV_FORMAT_NODE_ARRAY);
    node_map_add(&root, "attachments", MPV_FORMAT_NODE_ARRAY);
    // Adding a sibling can reallocate the root's value array. Acquire child
    // pointers only after both entries exist, before filling either array.
    mpv_node *tracks = node_map_get(&root, "tracks");
    mpv_node *attachments = node_map_get(&root, "attachments");
    int ordinals[AVMEDIA_TYPE_NB] = {0};
    for (unsigned i = 0; i < format->nb_streams; ++i) {
        AVStream *stream = format->streams[i];
        AVCodecParameters *p = stream->codecpar;
        if (p->codec_type == AVMEDIA_TYPE_ATTACHMENT) {
            mpv_node *item = node_array_add(attachments, MPV_FORMAT_NODE_MAP);
            node_map_add_int64(item, "streamIndex", i);
            metadata(item, "name", stream->metadata, "filename");
            metadata(item, "mimeType", stream->metadata, "mimetype");
            node_map_add_int64(item, "byteLength", p->extradata_size);
            node_map_add_flag(item, "font", p->codec_id == AV_CODEC_ID_TTF || p->codec_id == AV_CODEC_ID_OTF);
            continue;
        }
        if (p->codec_type != AVMEDIA_TYPE_VIDEO && p->codec_type != AVMEDIA_TYPE_AUDIO &&
            p->codec_type != AVMEDIA_TYPE_SUBTITLE)
            continue;
        mpv_node *track = node_array_add(tracks, MPV_FORMAT_NODE_MAP);
        node_map_add_int64(track, "streamIndex", i);
        node_map_add_int64(track, "typeOrdinal", ++ordinals[p->codec_type]);
        optional_string(track, "type", p->codec_type == AVMEDIA_TYPE_SUBTITLE ? "subtitle" : av_get_media_type_string(p->codec_type));
        optional_string(track, "codec", avcodec_get_name(p->codec_id));
        metadata(track, "language", stream->metadata, "language");
        metadata(track, "title", stream->metadata, "title");
        node_map_add_flag(track, "default", (stream->disposition & AV_DISPOSITION_DEFAULT) != 0);
        node_map_add_flag(track, "forced", (stream->disposition & AV_DISPOSITION_FORCED) != 0);
        seconds(track, "startSeconds", stream->start_time, stream->time_base);
        if (p->codec_type == AVMEDIA_TYPE_VIDEO) {
            optional_string(track, "pixelFormat", p->format >= 0 ? av_get_pix_fmt_name(p->format) : NULL);
            optional_string(track, "profile", avcodec_profile_name(p->codec_id, p->profile));
            const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(p->format);
            positive_integer(track, "bitDepth", desc && desc->nb_components ? desc->comp[0].depth : 0);
            positive_integer(track, "width", p->width);
            positive_integer(track, "height", p->height);
            AVRational sar = av_guess_sample_aspect_ratio(format, stream, NULL);
            rational(track, "pixelAspectRatio", sar);
            rational(track, "displayAspectRatio", p->width > 0 && p->height > 0 ?
                     av_mul_q(sar, (AVRational){p->width, p->height}) : (AVRational){0, 1});
            rational(track, "averageFrameRate", stream->avg_frame_rate);
            rational(track, "nominalFrameRate", stream->r_frame_rate);
            // Container averages alone cannot establish CFR versus VFR.
            node_map_add(track, "variableFrameRate", MPV_FORMAT_NONE);
            optional_string(track, "fieldOrder", p->field_order == AV_FIELD_PROGRESSIVE ? "progressive" : p->field_order == AV_FIELD_UNKNOWN ? NULL : "interlaced");
            optional_string(track, "colorPrimaries", p->color_primaries == AVCOL_PRI_UNSPECIFIED ? NULL : av_color_primaries_name(p->color_primaries));
            optional_string(track, "colorTransfer", p->color_trc == AVCOL_TRC_UNSPECIFIED ? NULL : av_color_transfer_name(p->color_trc));
            optional_string(track, "colorMatrix", p->color_space == AVCOL_SPC_UNSPECIFIED ? NULL : av_color_space_name(p->color_space));
            optional_string(track, "colorRange", p->color_range == AVCOL_RANGE_UNSPECIFIED ? NULL : av_color_range_name(p->color_range));
            const AVPacketSideData *rotation = av_packet_side_data_get(p->coded_side_data, p->nb_coded_side_data, AV_PKT_DATA_DISPLAYMATRIX);
            double angle = rotation && rotation->size >= 9 * sizeof(int32_t) ? -av_display_rotation_get((int32_t *)rotation->data) : 0;
            if (isfinite(angle)) node_map_add_double(track, "rotationDegrees", angle);
            else node_map_add(track, "rotationDegrees", MPV_FORMAT_NONE);
            node_map_add_flag(track, "hasMasteringDisplayMetadata", av_packet_side_data_get(p->coded_side_data, p->nb_coded_side_data, AV_PKT_DATA_MASTERING_DISPLAY_METADATA) != NULL);
            node_map_add_flag(track, "hasContentLightMetadata", av_packet_side_data_get(p->coded_side_data, p->nb_coded_side_data, AV_PKT_DATA_CONTENT_LIGHT_LEVEL) != NULL);
        } else if (p->codec_type == AVMEDIA_TYPE_AUDIO) {
            positive_integer(track, "channels", p->ch_layout.nb_channels);
            positive_integer(track, "sampleRate", p->sample_rate);
            char layout[128] = {0};
            av_channel_layout_describe(&p->ch_layout, layout, sizeof(layout));
            optional_string(track, "channelLayout", p->ch_layout.nb_channels > 0 ? layout : NULL);
        } else {
            const AVCodecDescriptor *desc = avcodec_descriptor_get(p->codec_id);
            optional_string(track, "subtitleKind", desc && (desc->props & AV_CODEC_PROP_TEXT_SUB) ? "text" : desc && (desc->props & AV_CODEC_PROP_BITMAP_SUB) ? "bitmap" : NULL);
        }
    }
    mpv_node *chapters = node_map_add(&root, "chapters", MPV_FORMAT_NODE_ARRAY);
    for (unsigned i = 0; i < format->nb_chapters; ++i) {
        AVChapter *chapter = format->chapters[i];
        mpv_node *item = node_array_add(chapters, MPV_FORMAT_NODE_MAP);
        seconds(item, "startSeconds", chapter->start, chapter->time_base);
        seconds(item, "endSeconds", chapter->end, chapter->time_base);
        metadata(item, "title", chapter->metadata, "title");
    }
    result = json_write(json, &root);
    if (result < 0 || !*json || strlen(*json) > 256 * 1024) {
        talloc_free(*json); *json = NULL; result = AVERROR(EFBIG);
    } else {
        result = 0;
    }
done:
    if (root.format != MPV_FORMAT_NONE) mpv_free_node_contents(&root);
    av_dict_free(&options);
    avformat_close_input(&format);
    av_freep(&io->buffer);
    avio_context_free(&io);
    return result;
}

MPV_EXPORT void mpv_ajn_probe_free_v1(char *json)
{
    talloc_free(json);
}
