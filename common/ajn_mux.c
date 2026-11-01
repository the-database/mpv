/* SPDX-License-Identifier: LGPL-2.1-or-later */
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <libavformat/avformat.h>
#include "common/ajn_mux.h"

struct output {
    void *opaque;
    int64_t (*open)(void *, const char *);
    int (*write)(void *, int64_t, const uint8_t *, int);
    int64_t (*seek)(void *, int64_t, int64_t, int);
    int (*close)(void *, int64_t);
};
struct output_file { struct output *owner; int64_t handle; };
static int write_packet(void *opaque, const uint8_t *data, int size)
{
    struct output_file *file = opaque;
    return file->owner->write(file->owner->opaque, file->handle, data, size);
}
static int64_t seek_packet(void *opaque, int64_t offset, int whence)
{
    struct output_file *file = opaque;
    return file->owner->seek(file->owner->opaque, file->handle, offset, whence);
}
static int open_output(AVFormatContext *format, AVIOContext **pb,
                       const char *url, int flags, AVDictionary **options)
{
    struct output *out = format->opaque;
    if (flags != AVIO_FLAG_WRITE) return AVERROR(EACCES);
    struct output_file *file = av_mallocz(sizeof(*file));
    uint8_t *buffer = av_malloc(32768);
    if (!file || !buffer) { av_free(file); av_free(buffer); return AVERROR(ENOMEM); }
    file->owner = out;
    file->handle = out->open(out->opaque, url);
    if (file->handle <= 0) { av_free(file); av_free(buffer); return AVERROR(EIO); }
    *pb = avio_alloc_context(buffer, 32768, 1, file, NULL, write_packet, seek_packet);
    if (!*pb) { out->close(out->opaque, file->handle); av_free(file); av_free(buffer); return AVERROR(ENOMEM); }
    return 0;
}
static int close_output(AVFormatContext *format, AVIOContext *pb)
{
    struct output_file *file = pb->opaque;
    avio_flush(pb);
    int error = pb->error;
    int closed = file->owner->close(file->owner->opaque, file->handle);
    av_free(file); av_freep(&pb->buffer); avio_context_free(&pb);
    return error < 0 ? error : closed;
}
static int deny_input(AVFormatContext *s, AVIOContext **pb, const char *url,
                      int flags, AVDictionary **options) { return AVERROR(EACCES); }

MPV_EXPORT int mpv_ajn_mux_v1(void *opaque,
    int (*read_packet)(void *, uint8_t *, int), int (*cancel)(void *),
    int (*wait_packet)(void *, double, double),
    int64_t (*open_file)(void *, const char *),
    int (*write_file)(void *, int64_t, const uint8_t *, int),
    int64_t (*seek_file)(void *, int64_t, int64_t, int),
    int (*close_file)(void *, int64_t), const char *container,
    int segmented, double segment_seconds)
{
    if (!opaque || !read_packet || !cancel || !wait_packet || !open_file || !write_file || !seek_file ||
        !close_file || !container || !isfinite(segment_seconds) ||
        segment_seconds < 0.5 || segment_seconds > 6 ||
        (strcmp(container, "matroska") && strcmp(container, "mpegts") && strcmp(container, "fragmentedMp4")))
        return AVERROR(EINVAL);
    int result = AVERROR(ENOMEM);
    AVFormatContext *input = NULL, *output = NULL;
    AVDictionary *options = NULL;
    AVPacket *packet = av_packet_alloc();
    uint8_t *buffer = av_malloc(32768);
    AVIOContext *io = buffer ? avio_alloc_context(buffer, 32768, 0, opaque, read_packet, NULL, NULL) : NULL;
    struct output callbacks = {opaque, open_file, write_file, seek_file, close_file};
    if (!io) av_free(buffer);
    if (!io || !packet) goto done;
    io->seekable = 0;
    input = avformat_alloc_context();
    if (!input) goto done;
    input->pb = io; input->flags |= AVFMT_FLAG_CUSTOM_IO;
    input->io_open = deny_input;
    input->interrupt_callback = (AVIOInterruptCB){cancel, opaque};
    input->max_streams = 4;
    // The upstream encoder is host-owned and always emits an intermediate MKV.
    av_dict_set(&options, "format_whitelist", "matroska,webm", 0);
    av_dict_set(&options, "probesize", "1048576", 0);
    av_dict_set(&options, "analyzeduration", "1000000", 0);
    result = avformat_open_input(&input, NULL, NULL, &options);
    av_dict_free(&options);
    if (result < 0) goto done;
    result = avformat_find_stream_info(input, NULL);
    if (result < 0) goto done;
    if (input->nb_streams < 1 || input->nb_streams > 4) { result = AVERROR(EINVAL); goto done; }
    bool mkv = !strcmp(container, "matroska"), mp4 = !strcmp(container, "fragmentedMp4");
    const char *muxer = segmented ? (mkv ? "segment" : "hls") : (mp4 ? "mp4" : container);
    const char *filename = segmented ? (mkv ? "part_%08d.mkv" : "index.m3u8") :
        (mkv ? "continuous.mkv" : mp4 ? "continuous.mp4" : "continuous.ts");
    result = avformat_alloc_output_context2(&output, NULL, muxer, filename);
    if (result < 0) goto done;
    output->opaque = &callbacks;
    output->io_open = open_output; output->io_close2 = close_output;
    output->interrupt_callback = (AVIOInterruptCB){cancel, opaque};
    output->flags |= AVFMT_FLAG_FLUSH_PACKETS;
    for (unsigned i = 0; i < input->nb_streams; i++) {
        AVStream *source = input->streams[i];
        if (source->codecpar->codec_type != AVMEDIA_TYPE_VIDEO && source->codecpar->codec_type != AVMEDIA_TYPE_AUDIO) {
            result = AVERROR(EINVAL); goto done;
        }
        AVStream *stream = avformat_new_stream(output, NULL);
        if (!stream) { result = AVERROR(ENOMEM); goto done; }
        result = avcodec_parameters_copy(stream->codecpar, source->codecpar);
        if (result < 0) goto done;
        stream->codecpar->codec_tag = 0;
        stream->time_base = source->time_base;
        stream->avg_frame_rate = source->avg_frame_rate;
    }
    char interval[32]; snprintf(interval, sizeof(interval), "%.6f", segment_seconds);
    if (segmented && mkv) {
        av_dict_set(&options, "segment_time", interval, 0);
        av_dict_set(&options, "segment_format", "matroska", 0);
        av_dict_set(&options, "segment_list", "index.csv", 0);
        av_dict_set(&options, "segment_list_type", "csv", 0);
        av_dict_set(&options, "segment_list_size", "128", 0);
        av_dict_set(&options, "reset_timestamps", "1", 0);
    } else if (segmented) {
        av_dict_set(&options, "hls_time", interval, 0);
        av_dict_set(&options, "hls_list_size", "128", 0);
        av_dict_set(&options, "hls_segment_type", mp4 ? "fmp4" : "mpegts", 0);
        av_dict_set(&options, "hls_fmp4_init_filename", "init.mp4", 0);
        av_dict_set(&options, "hls_segment_filename", mp4 ? "part_%08d.m4s" : "part_%08d.ts", 0);
        av_dict_set(&options, "hls_flags", "independent_segments+temp_file", 0);
    } else if (mp4) {
        av_dict_set(&options, "movflags", "frag_keyframe+empty_moov+default_base_moof", 0);
    } else if (mkv) {
        av_dict_set(&options, "live", "1", 0);
        av_dict_set(&options, "cluster_time_limit", "1000", 0);
    }
    if (!(output->oformat->flags & AVFMT_NOFILE)) {
        result = open_output(output, &output->pb, filename, AVIO_FLAG_WRITE, NULL);
        if (result < 0) goto done;
    }
    result = avformat_write_header(output, &options);
    av_dict_free(&options);
    if (result < 0) goto done;
    while ((result = av_read_frame(input, packet)) >= 0) {
        if (cancel(opaque)) { av_packet_unref(packet); result = AVERROR_EXIT; goto done; }
        AVStream *stream = input->streams[packet->stream_index];
        double start = packet->pts == AV_NOPTS_VALUE ? NAN : packet->pts * av_q2d(stream->time_base);
        double duration = packet->duration > 0 ? packet->duration * av_q2d(stream->time_base) :
            stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && stream->avg_frame_rate.num > 0 ? av_q2d(av_inv_q(stream->avg_frame_rate)) : 0;
        result = wait_packet(opaque, start, start + duration);
        if (result < 0) { av_packet_unref(packet); goto done; }
        av_packet_rescale_ts(packet, input->streams[packet->stream_index]->time_base,
                             output->streams[packet->stream_index]->time_base);
        packet->pos = -1;
        result = av_interleaved_write_frame(output, packet);
        av_packet_unref(packet);
        if (result < 0) goto done;
    }
    if (result == AVERROR_EOF) result = av_write_trailer(output);
done:
    av_dict_free(&options);
    if (output) {
        if (output->pb) {
            int close_error = close_output(output, output->pb); output->pb = NULL;
            if (result >= 0 && close_error < 0) result = close_error;
        }
        avformat_free_context(output);
    }
    avformat_close_input(&input);
    if (io) { av_freep(&io->buffer); avio_context_free(&io); }
    av_packet_free(&packet);
    return result;
}
