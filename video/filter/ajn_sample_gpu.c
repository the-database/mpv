/* SPDX-License-Identifier: LGPL-2.1-or-later */
#ifndef COBJMACROS
#define COBJMACROS
#endif
#include <stdlib.h>
#include <string.h>
#include <d3d10.h>
#include "ajn_sample_gpu.h"

struct ajn_sample_gpu {
    ID3D11DeviceContext *context;
    ID3D11VideoDevice *video;
    ID3D11VideoContext *video_context;
    ID3D11VideoProcessorEnumerator *enumerator;
    ID3D11VideoProcessor *processor;
    ID3D11Texture2D *target, *staging;
    ID3D11VideoProcessorOutputView *output;
    ID3D11Query *completed;
    UINT width, height, number;
};

#define RELEASE(x) do { if (x) IUnknown_Release((IUnknown *)(x)); } while (0)
#define CHECK(call) do { hr = (call); if (FAILED(hr)) goto fail; } while (0)

void ajn_sample_gpu_destroy(struct ajn_sample_gpu *p)
{
    if (!p) return;
    RELEASE(p->completed); RELEASE(p->output); RELEASE(p->staging);
    RELEASE(p->target); RELEASE(p->processor); RELEASE(p->enumerator);
    RELEASE(p->video_context); RELEASE(p->video); RELEASE(p->context);
    free(p);
}

struct ajn_sample_gpu *ajn_sample_gpu_create(ID3D11Device *device,
    UINT coded_w, UINT coded_h, DXGI_FORMAT format, UINT width, UINT height,
    HRESULT *error)
{
    HRESULT hr = E_INVALIDARG;
    struct ajn_sample_gpu *p = NULL;
    if (!device || !width || width > 320 || !height || height > 180 ||
        !coded_w || !coded_h) goto fail;
    // Event queries account for VideoProcessorBlt on feature level 11+.
    if (ID3D11Device_GetFeatureLevel(device) < D3D_FEATURE_LEVEL_11_0) goto fail;
    p = calloc(1, sizeof(*p));
    if (!p) { hr = E_OUTOFMEMORY; goto fail; }
    p->width = width; p->height = height;
    ID3D10Multithread *multithread = NULL;
    CHECK(ID3D11Device_QueryInterface(device, &IID_ID3D10Multithread, (void **)&multithread));
    ID3D10Multithread_SetMultithreadProtected(multithread, TRUE);
    ID3D10Multithread_Release(multithread);
    ID3D11Device_GetImmediateContext(device, &p->context);
    CHECK(ID3D11Device_QueryInterface(device, &IID_ID3D11VideoDevice, (void **)&p->video));
    CHECK(ID3D11DeviceContext_QueryInterface(p->context, &IID_ID3D11VideoContext,
                                           (void **)&p->video_context));
    D3D11_VIDEO_PROCESSOR_CONTENT_DESC desc = {
        .InputFrameFormat = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE,
        .InputWidth = coded_w, .InputHeight = coded_h,
        .InputFrameRate = {60, 1}, .OutputFrameRate = {60, 1},
        .OutputWidth = width, .OutputHeight = height,
        .Usage = D3D11_VIDEO_USAGE_OPTIMAL_QUALITY,
    };
    CHECK(ID3D11VideoDevice_CreateVideoProcessorEnumerator(p->video, &desc, &p->enumerator));
    UINT support = 0;
    CHECK(ID3D11VideoProcessorEnumerator_CheckVideoProcessorFormat(p->enumerator, format, &support));
    if (!(support & D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_INPUT)) { hr = E_NOTIMPL; goto fail; }
    CHECK(ID3D11VideoProcessorEnumerator_CheckVideoProcessorFormat(p->enumerator, DXGI_FORMAT_B8G8R8A8_UNORM, &support));
    if (!(support & D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_OUTPUT)) { hr = E_NOTIMPL; goto fail; }
    CHECK(ID3D11VideoDevice_CreateVideoProcessor(p->video, p->enumerator, 0, &p->processor));
    D3D11_TEXTURE2D_DESC texture = {
        .Width = width, .Height = height, .MipLevels = 1, .ArraySize = 1,
        .Format = DXGI_FORMAT_B8G8R8A8_UNORM, .SampleDesc = {1, 0},
        .Usage = D3D11_USAGE_DEFAULT, .BindFlags = D3D11_BIND_RENDER_TARGET,
    };
    CHECK(ID3D11Device_CreateTexture2D(device, &texture, NULL, &p->target));
    texture.Usage = D3D11_USAGE_STAGING;
    texture.BindFlags = 0; texture.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    CHECK(ID3D11Device_CreateTexture2D(device, &texture, NULL, &p->staging));
    D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC output = { .ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D };
    CHECK(ID3D11VideoDevice_CreateVideoProcessorOutputView(p->video, (ID3D11Resource *)p->target,
        p->enumerator, &output, &p->output));
    D3D11_QUERY_DESC query = { .Query = D3D11_QUERY_EVENT };
    CHECK(ID3D11Device_CreateQuery(device, &query, &p->completed));
    RECT destination = {0, 0, width, height};
    ID3D11VideoContext_VideoProcessorSetStreamFrameFormat(p->video_context, p->processor, 0,
        D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE);
    ID3D11VideoContext_VideoProcessorSetStreamAutoProcessingMode(p->video_context, p->processor, 0, FALSE);
    ID3D11VideoContext_VideoProcessorSetStreamDestRect(p->video_context, p->processor, 0, TRUE, &destination);
    ID3D11VideoContext_VideoProcessorSetOutputTargetRect(p->video_context, p->processor, TRUE, &destination);
    ID3D11VideoContext_VideoProcessorSetOutputAlphaFillMode(p->video_context, p->processor,
        D3D11_VIDEO_PROCESSOR_ALPHA_FILL_MODE_OPAQUE, 0);
    *error = S_OK;
    return p;
fail:
    ajn_sample_gpu_destroy(p);
    *error = hr;
    return NULL;
}

HRESULT ajn_sample_gpu_submit(struct ajn_sample_gpu *p, ID3D11Texture2D *source,
    UINT array_slice, const RECT *crop, BOOL matrix_709, BOOL full)
{
    HRESULT hr;
    D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC input = {
        .ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D,
        .Texture2D.ArraySlice = array_slice,
    };
    ID3D11VideoProcessorInputView *view = NULL;
    CHECK(ID3D11VideoDevice_CreateVideoProcessorInputView(p->video, (ID3D11Resource *)source,
        p->enumerator, &input, &view));
    ID3D11VideoContext_VideoProcessorSetStreamSourceRect(p->video_context, p->processor, 0, TRUE, crop);
    // Use matrix and range, independently of primaries. E.g. SD input can use a
    // BT.601 matrix with BT.709 primaries; a DXGI P601/P709 shortcut conflates them.
    D3D11_VIDEO_PROCESSOR_COLOR_SPACE color = {
        .Usage = 1, .RGB_Range = !full, .YCbCr_Matrix = matrix_709,
        .Nominal_Range = full ? D3D11_VIDEO_PROCESSOR_NOMINAL_RANGE_0_255 : D3D11_VIDEO_PROCESSOR_NOMINAL_RANGE_16_235,
    };
    ID3D11VideoContext_VideoProcessorSetStreamColorSpace(p->video_context, p->processor, 0, &color);
    color.RGB_Range = 0; color.Nominal_Range = D3D11_VIDEO_PROCESSOR_NOMINAL_RANGE_0_255;
    ID3D11VideoContext_VideoProcessorSetOutputColorSpace(p->video_context, p->processor, &color);
    D3D11_VIDEO_PROCESSOR_STREAM stream = { .Enable = TRUE, .pInputSurface = view };
    CHECK(ID3D11VideoContext_VideoProcessorBlt(p->video_context, p->processor, p->output, p->number++, 1, &stream));
    ID3D11DeviceContext_CopyResource(p->context, (ID3D11Resource *)p->staging, (ID3D11Resource *)p->target);
    ID3D11DeviceContext_End(p->context, (ID3D11Asynchronous *)p->completed);
    ID3D11DeviceContext_Flush(p->context); // submit; does not wait for completion
fail:
    RELEASE(view);
    return hr;
}

HRESULT ajn_sample_gpu_read(struct ajn_sample_gpu *p, void *pixels)
{
    BOOL complete = FALSE;
    HRESULT hr = ID3D11DeviceContext_GetData(p->context, (ID3D11Asynchronous *)p->completed,
        &complete, sizeof(complete), D3D11_ASYNC_GETDATA_DONOTFLUSH);
    if (hr != S_OK || !complete) return FAILED(hr) ? hr : S_FALSE;
    D3D11_MAPPED_SUBRESOURCE mapped;
    hr = ID3D11DeviceContext_Map(p->context, (ID3D11Resource *)p->staging, 0,
        D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped);
    if (hr == DXGI_ERROR_WAS_STILL_DRAWING) return S_FALSE;
    if (FAILED(hr)) return hr;
    for (UINT y = 0; y < p->height; y++)
        memcpy((char *)pixels + y * p->width * 4, (char *)mapped.pData + y * mapped.RowPitch, p->width * 4);
    ID3D11DeviceContext_Unmap(p->context, (ID3D11Resource *)p->staging, 0);
    return S_OK;
}
