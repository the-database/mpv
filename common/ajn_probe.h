// Private ABI for the trusted AJN media worker, not an addon ABI.
// SPDX-License-Identifier: LGPL-2.1-or-later
#ifndef MP_AJN_PROBE_H
#define MP_AJN_PROBE_H
#include <stdint.h>
#include "include/mpv/client.h"
#define AJN_PROBE_VERSION 1
MPV_EXPORT int mpv_ajn_probe_v1(void *opaque,
    int (*read_packet)(void *, uint8_t *, int),
    int64_t (*seek)(void *, int64_t, int), int (*cancel)(void *),
    int can_seek, char **json);
MPV_EXPORT void mpv_ajn_probe_free_v1(char *json);
// Private trusted-host readiness probe. Opens and closes the actual selected
// hardware encoder using a bounded, validated output combination.
MPV_EXPORT int mpv_ajn_encoder_check_v1(const char *name, const char *options,
    int width, int height, int bit_depth, double fps);
#endif
