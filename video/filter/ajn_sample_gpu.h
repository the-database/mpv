/* SPDX-License-Identifier: LGPL-2.1-or-later */
#pragma once
#include <windows.h>
#include <d3d11.h>

struct ajn_sample_gpu;

// Only scaling, matrix/range conversion and a small staging copy. No temporal
// processing, enhancement, transfer conversion, gamut conversion or tone map.
struct ajn_sample_gpu *ajn_sample_gpu_create(ID3D11Device *device,
    UINT coded_w, UINT coded_h, DXGI_FORMAT format, UINT width, UINT height,
    HRESULT *error);
HRESULT ajn_sample_gpu_submit(struct ajn_sample_gpu *gpu, ID3D11Texture2D *source,
    UINT array_slice, const RECT *crop, BOOL matrix_709, BOOL full);
// S_FALSE means GPU busy. S_OK means width*height tightly packed BGRA8 bytes.
// Never waits for the GPU. The caller must retain the source until success or
// device removal; a failed sample must not return a still-used texture to a pool.
HRESULT ajn_sample_gpu_read(struct ajn_sample_gpu *gpu, void *pixels);
void ajn_sample_gpu_destroy(struct ajn_sample_gpu *gpu);
