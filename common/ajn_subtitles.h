/* SPDX-License-Identifier: LGPL-2.1-or-later */
#ifndef MP_AJN_SUBTITLES_H
#define MP_AJN_SUBTITLES_H
#include <stdint.h>
#include "include/mpv/client.h"
#define AJN_SUBTITLES_VERSION 1
MPV_EXPORT int mpv_ajn_subtitles_v1(void *opaque,
    int (*read_packet)(void *, uint8_t *, int),
    int64_t (*seek)(void *, int64_t, int), int (*cancel)(void *),
    int (*write)(void *, const uint8_t *, int), int can_seek,
    int stream_index, double start_seconds, double end_seconds);
#endif
