/* SPDX-License-Identifier: LGPL-2.1-or-later */
#ifndef MP_AJN_MUX_H
#define MP_AJN_MUX_H
#include <stdint.h>
#include "include/mpv/client.h"
#define AJN_MUX_VERSION 1
// Private supervised-worker ABI. The host owns all output handles and quotas.
MPV_EXPORT int mpv_ajn_mux_v1(void *opaque,
    int (*read_packet)(void *, uint8_t *, int), int (*cancel)(void *),
    int (*wait_packet)(void *, double, double),
    int64_t (*open_file)(void *, const char *),
    int (*write_file)(void *, int64_t, const uint8_t *, int),
    int64_t (*seek_file)(void *, int64_t, int64_t, int),
    int (*close_file)(void *, int64_t), const char *container,
    int segmented, double segment_seconds);
#endif
