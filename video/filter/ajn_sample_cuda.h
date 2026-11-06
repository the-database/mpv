#pragma once
#include <windows.h>
#include <libavutil/buffer.h>
#include "aji_analysis.h"
struct mp_image;
struct ajn_sample_cuda;
struct ajn_sample_cuda *ajn_sample_cuda_create(AVBufferRef *frames, int width, int height, bool mask,
    aji_observer_cancel_fn cancel, aji_observer_state_fn state, void *opaque, char *error, size_t error_size);
bool ajn_sample_cuda_matches(struct ajn_sample_cuda *p, struct mp_image *image, int width, int height);
HRESULT ajn_sample_cuda_read(struct ajn_sample_cuda *p, struct mp_image *image, const RECT *crop,
    unsigned char *color, uint16_t *mask, double *gpu_ms);
void ajn_sample_cuda_destroy(struct ajn_sample_cuda *p);
