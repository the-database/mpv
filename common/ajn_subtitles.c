/* SPDX-License-Identifier: LGPL-2.1-or-later */
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include "common/ajn_subtitles.h"

static int deny_open(AVFormatContext *s, AVIOContext **pb, const char *url,
                     int flags, AVDictionary **options) { return AVERROR(EACCES); }

static bool supported(enum AVCodecID codec)
{
    return codec == AV_CODEC_ID_ASS || codec == AV_CODEC_ID_SSA ||
        codec == AV_CODEC_ID_SUBRIP || codec == AV_CODEC_ID_WEBVTT ||
        codec == AV_CODEC_ID_MOV_TEXT || codec == AV_CODEC_ID_TEXT;
}

MPV_EXPORT int mpv_ajn_subtitles_v1(void *opaque,
    int (*read_packet)(void *, uint8_t *, int),
    int64_t (*seek)(void *, int64_t, int), int (*cancel)(void *),
    int (*write)(void *, const uint8_t *, int), int can_seek,
    int stream_index, double start_seconds, double end_seconds)
{
    if (!opaque || !read_packet || !cancel || !write || stream_index < 0 ||
        !isfinite(start_seconds) || !isfinite(end_seconds) || start_seconds < 0 ||
        end_seconds <= start_seconds || end_seconds > 315576000)
        return AVERROR(EINVAL);
    int result = AVERROR(ENOMEM);
    AVFormatContext *input = NULL;
    AVCodecContext *decoder = NULL, *encoder = NULL;
    AVDictionary *options = NULL;
    AVPacket *packet = av_packet_alloc();
    uint8_t *buffer = av_malloc(32768), *cue = av_malloc(65536);
    AVIOContext *io = buffer ? avio_alloc_context(buffer, 32768, 0, opaque, read_packet, NULL, seek) : NULL;
    if (!io) av_free(buffer);
    if (!io || !packet || !cue) goto done;
    io->seekable = can_seek ? AVIO_SEEKABLE_NORMAL : 0;
    input = avformat_alloc_context(); if (!input) goto done;
    input->pb = io; input->flags |= AVFMT_FLAG_CUSTOM_IO;
    input->io_open = deny_open; input->interrupt_callback = (AVIOInterruptCB){cancel, opaque};
    input->max_streams = 128;
    av_dict_set(&options, "format_whitelist", "matroska,webm,mov,avi,mpegts,srt,ass,webvtt", 0);
    av_dict_set(&options, "protocol_whitelist", "", 0);
    av_dict_set(&options, "probesize", "4194304", 0);
    av_dict_set(&options, "analyzeduration", "5000000", 0);
    result = avformat_open_input(&input, NULL, NULL, &options); av_dict_free(&options);
    if (result < 0) goto done;
    result = avformat_find_stream_info(input, NULL); if (result < 0) goto done;
    if ((unsigned)stream_index >= input->nb_streams) { result = AVERROR(EINVAL); goto done; }
    AVStream *stream = input->streams[stream_index];
    if (stream->codecpar->codec_type != AVMEDIA_TYPE_SUBTITLE || !supported(stream->codecpar->codec_id)) {
        result = AVERROR(ENOSYS); goto done;
    }
    const AVCodec *decode_codec = avcodec_find_decoder(stream->codecpar->codec_id);
    const AVCodec *encode_codec = avcodec_find_encoder(AV_CODEC_ID_WEBVTT);
    if (!decode_codec || !encode_codec) { result = AVERROR(ENOSYS); goto done; }
    decoder = avcodec_alloc_context3(decode_codec); encoder = avcodec_alloc_context3(encode_codec);
    if (!decoder || !encoder) { result = AVERROR(ENOMEM); goto done; }
    result = avcodec_parameters_to_context(decoder, stream->codecpar); if (result < 0) goto done;
    decoder->pkt_timebase = stream->time_base;
    result = avcodec_open2(decoder, decode_codec, NULL); if (result < 0) goto done;
    if (decoder->subtitle_header_size <= 0 || decoder->subtitle_header_size > 1024 * 1024) { result = AVERROR_INVALIDDATA; goto done; }
    encoder->subtitle_header = av_mallocz(decoder->subtitle_header_size + AV_INPUT_BUFFER_PADDING_SIZE);
    if (!encoder->subtitle_header) { result = AVERROR(ENOMEM); goto done; }
    memcpy(encoder->subtitle_header, decoder->subtitle_header, decoder->subtitle_header_size);
    encoder->subtitle_header_size = decoder->subtitle_header_size;
    // WebVTT requires an encoder time base even though this helper formats
    // source-timed cue boundaries itself at millisecond precision.
    encoder->time_base = (AVRational){1, 1000};
    result = avcodec_open2(encoder, encode_codec, NULL); if (result < 0) goto done;
    result = write(opaque, (const uint8_t *)"WEBVTT\n\n", 8); if (result < 0) goto done;
    // Scan selected packets so cues crossing the requested start remain present.
    // The supervisor bounds wall time, read bytes, output bytes and process memory.
    while ((result = av_read_frame(input, packet)) >= 0) {
        if (cancel(opaque)) { result = AVERROR_EXIT; goto done; }
        if (packet->stream_index != stream_index) { av_packet_unref(packet); continue; }
        AVSubtitle subtitle = {0}; int got = 0;
        result = avcodec_decode_subtitle2(decoder, &subtitle, &got, packet);
        if (result < 0) { avsubtitle_free(&subtitle); goto done; }
        if (got && subtitle.num_rects) {
            double pts = subtitle.pts != AV_NOPTS_VALUE ? subtitle.pts / (double)AV_TIME_BASE :
                packet->pts != AV_NOPTS_VALUE ? packet->pts * av_q2d(stream->time_base) : NAN;
            double start = pts + subtitle.start_display_time / 1000.0;
            double end = pts + subtitle.end_display_time / 1000.0;
            if (subtitle.end_display_time == UINT32_MAX || end <= start)
                end = pts + packet->duration * av_q2d(stream->time_base);
            if (!isfinite(start) || !isfinite(end) || end <= start) {
                avsubtitle_free(&subtitle); result = AVERROR_INVALIDDATA; goto done;
            }
            if (end > start_seconds && start < end_seconds) {
                int count = avcodec_encode_subtitle(encoder, cue, 65536, &subtitle);
                if (count < 0) { avsubtitle_free(&subtitle); result = count; goto done; }
                if (count > 0) {
                    int64_t a = llround(fmax(start, start_seconds) * 1000);
                    int64_t b = llround(fmin(end, end_seconds) * 1000);
                    char times[128];
                    int size = snprintf(times, sizeof(times),
                        "%02"PRId64":%02"PRId64":%02"PRId64".%03"PRId64" --> %02"PRId64":%02"PRId64":%02"PRId64".%03"PRId64"\n",
                        a / 3600000, a / 60000 % 60, a / 1000 % 60, a % 1000,
                        b / 3600000, b / 60000 % 60, b / 1000 % 60, b % 1000);
                    result = write(opaque, (const uint8_t *)times, size);
                    if (result >= 0) result = write(opaque, cue, count);
                    if (result >= 0) result = write(opaque, (const uint8_t *)"\n\n", 2);
                    if (result < 0) { avsubtitle_free(&subtitle); goto done; }
                }
            }
        }
        avsubtitle_free(&subtitle); av_packet_unref(packet);
    }
    if (result == AVERROR_EOF) result = 0;
done:
    av_dict_free(&options); av_packet_free(&packet); av_free(cue);
    avcodec_free_context(&decoder); avcodec_free_context(&encoder);
    avformat_close_input(&input);
    if (io) { av_freep(&io->buffer); avio_context_free(&io); }
    return result;
}
