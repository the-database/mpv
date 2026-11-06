/* Private trusted-player/host transport; never mapped into an addon.
 * SPDX-License-Identifier: LGPL-2.1-or-later */
#ifndef MPV_AJN_SCENE_SHARED_H
#define MPV_AJN_SCENE_SHARED_H
#include <stddef.h>
#include <stdint.h>

#define AJN_SCENE_MAGIC 0x31434a41
#define AJN_SCENE_VERSION 1
#define AJN_SCENE_MVTOOLS_VERSION 1
#define AJN_SCENE_ENHANCED_VERSION 1
#define AJN_SCENE_WIDTH 320
#define AJN_SCENE_HEIGHT 180
#define AJN_SCENE_PLANE (AJN_SCENE_WIDTH * AJN_SCENE_HEIGHT)
#define AJN_SCENE_BYTES (2 * AJN_SCENE_PLANE)
struct ajn_scene_shared {
    int32_t magic, version, capacity, header_bytes;
    int32_t lock, enabled;
    int64_t generation, request_id, next_id, response_id;
    int64_t lease_until_ms, deadline_ms;
    double previous_pts, current_pts;
    int32_t source_width, source_height, width, height;
    int32_t decision, status, timeouts, accepted, budget_ms, consecutive_timeouts;
    int64_t epoch;
    // Optional extension: only used with privateMvtoolsSceneAbi=1 metadata.
    // Zero preserves the original sandboxed detector protocol and offsets.
    // algorithm: 0 = custom, 1 = legacy MVTools, 2 = enhanced (separate capability).
    int32_t algorithm, thscd1, thscd2, changed_blocks, total_blocks, cut_pairs;
    int64_t analysis_us, block_sad_threshold;
    int32_t source_bits;
    uint8_t reserved[76];
    uint8_t pixels[AJN_SCENE_BYTES];
};
_Static_assert(offsetof(struct ajn_scene_shared, pixels) == 256, "scene ABI layout");
#endif
