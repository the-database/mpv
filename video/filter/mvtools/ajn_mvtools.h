/* Optional trusted native detector. No VapourSynth runtime or guest DLLs.
 * SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef AJN_MVTOOLS_H
#define AJN_MVTOOLS_H
#include <stddef.h>
#include <stdint.h>
#define AJN_MVTOOLS_ABI 1
#if defined(_WIN32) && defined(AJN_MVTOOLS_BUILD)
#define AJN_MVTOOLS_EXPORT __declspec(dllexport)
#elif defined(AJN_MVTOOLS_BUILD)
#define AJN_MVTOOLS_EXPORT __attribute__((visibility("default")))
#else
#define AJN_MVTOOLS_EXPORT
#endif
#ifdef __cplusplus
extern "C" {
#endif
struct ajn_mvtools_frame {
    int32_t width, height, bits, sub_w, sub_h;
    const uint8_t *planes[3];
    ptrdiff_t strides[3];
    double pts;
};
struct ajn_mvtools_result {
    int32_t cut, changed_blocks, total_blocks;
    int32_t block_count_threshold;
    int64_t block_sad_threshold;
};
AJN_MVTOOLS_EXPORT int ajn_mvtools_version(void);
AJN_MVTOOLS_EXPORT void *ajn_mvtools_create(void);
AJN_MVTOOLS_EXPORT void ajn_mvtools_destroy(void *context);
AJN_MVTOOLS_EXPORT void ajn_mvtools_reset(void *context);
/* Returns 0 on success, -1 on unsupported/invalid input, -2 on an error.
 * Frames are full-resolution planar integer YUV or gray, with positive strides.
 * thscd1/thscd2 use MVTools units (legacy defaults 240/130).
 * One context belongs to one serial playback pipeline. */
AJN_MVTOOLS_EXPORT int ajn_mvtools_analyze(void *context,
    const struct ajn_mvtools_frame *previous, const struct ajn_mvtools_frame *current,
    int thscd1, int thscd2, struct ajn_mvtools_result *result);
/* Enhanced mode independently classifies motion residuals using appearance
 * within changed blocks, rejecting illumination/effects and detecting local
 * picture cuts without a quiet-history requirement. No lookahead.
 * Same ABI/result layout; this optional export is advertised separately. */
AJN_MVTOOLS_EXPORT int ajn_mvtools_analyze_enhanced(void *context,
    const struct ajn_mvtools_frame *previous, const struct ajn_mvtools_frame *current,
    int thscd1, int thscd2, struct ajn_mvtools_result *result);
#ifdef __cplusplus
}
#endif
#endif
