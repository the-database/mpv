/* Private AJN media-worker ABI. This is not an addon-facing native API.
 * SPDX-License-Identifier: LGPL-2.1-or-later */
#pragma once
#include <stdint.h>
#include <stddef.h>

#define AJN_SAMPLE_MAGIC 0x31464a41u
#define AJN_SAMPLE_VERSION 1
#define AJN_SAMPLE_WIDTH 320
#define AJN_SAMPLE_HEIGHT 180
#define AJN_SAMPLE_BYTES (AJN_SAMPLE_WIDTH * AJN_SAMPLE_HEIGHT * 4)

// The supervisor owns config. The filter owns epoch/status and the publication
// sequence. Pixels and metadata form one seqlock publication; readers copy then
// recheck sequence/config/epoch. No writer waits for an IPC consumer.
struct ajn_sample_shared {
    uint32_t magic, version, capacity, header_bytes;
    int64_t config;              // generation:32, fps:8, height:12, width:12
    int64_t epoch;
    int64_t sequence;            // odd while writing, even after publication
    uint64_t frame_id;
    int64_t sample_config, sample_epoch, pts_us, produced_ns;
    uint32_t width, height, stride, payload_bytes;
    uint32_t source_width, source_height;
    int32_t status, dropped;     // independent atomic status/counter
    uint32_t primaries, transfer, rotation, par_num, par_den;
    int32_t error;
    uint32_t crop_x, crop_y, crop_width, crop_height, vflip;
    // Optional private player-observation extension. Old worker mappings keep
    // these bytes zero. Writers serialize publication across filter rebuilds;
    // a retired producer cannot replace a newer producer's status.
    int32_t writer;
    int64_t producer;
    int64_t lease_until_ms;      // GetTickCount64; named player mappings only
    int64_t player_status;       // producer:32, status:32; stale status is ignored
    uint32_t analysis, source_stage; // optional v2 paired masks; v1 remains BGRA
    int64_t gpu_us, age_us;
    uint8_t reserved[48];
    uint8_t pixels[AJN_SAMPLE_BYTES];
};

enum ajn_sample_status {
    AJN_SAMPLE_IDLE = 1, AJN_SAMPLE_PENDING, AJN_SAMPLE_READY,
    AJN_SAMPLE_UNSUPPORTED, AJN_SAMPLE_FAILED, AJN_SAMPLE_CLOSED,
    AJN_SAMPLE_LOADING, AJN_SAMPLE_MODEL_MISSING, AJN_SAMPLE_BUILDING,
};
enum ajn_sample_primaries { AJN_PRIM_709 = 1, AJN_PRIM_601_525, AJN_PRIM_601_625 };
enum ajn_sample_transfer { AJN_TRC_1886 = 1, AJN_TRC_SRGB, AJN_TRC_G22, AJN_TRC_G24 };

_Static_assert(offsetof(struct ajn_sample_shared, config) == 16, "config ABI");
_Static_assert(offsetof(struct ajn_sample_shared, sequence) == 32, "sequence ABI");
_Static_assert(offsetof(struct ajn_sample_shared, status) == 104, "status ABI");
_Static_assert(offsetof(struct ajn_sample_shared, producer) == 160, "producer ABI");
_Static_assert(offsetof(struct ajn_sample_shared, lease_until_ms) == 168, "lease ABI");
_Static_assert(offsetof(struct ajn_sample_shared, pixels) == 256, "pixel ABI");
_Static_assert(sizeof(struct ajn_sample_shared) == 256 + AJN_SAMPLE_BYTES, "map ABI");

static inline int ajn_sample_config(int64_t config, int *w, int *h, int *fps)
{
    *w = config & 0xfff;
    *h = (config >> 12) & 0xfff;
    *fps = (config >> 24) & 0xff;
    return config > 0 && (config >> 32) > 0 && *w >= 1 && *w <= AJN_SAMPLE_WIDTH &&
           *h >= 1 && *h <= AJN_SAMPLE_HEIGHT && *fps >= 1 && *fps <= 60;
}
