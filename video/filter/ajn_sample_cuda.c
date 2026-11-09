/* Optional CUDA observation adapter. Never loaded without a sample filter.
 * SPDX-License-Identifier: LGPL-2.1-or-later */
#include "config.h"
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <ffnvcodec/dynlink_loader.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixfmt.h>
#include "video/mp_image.h"
#include "ajn_sample_cuda.h"

struct ajn_sample_cuda {
    HMODULE module;
    AVBufferRef *frames;
    aji_observer *observer;
    int width, height;
    aji_observer *(*create)(void *, int, int, const char *, aji_observer_cancel_fn, aji_observer_state_fn, void *, char *, size_t);
    int (*run)(aji_observer *, const aji_frame *, const int *, void *, unsigned char *, uint16_t *, double *);
    void (*destroy)(aji_observer *);
};
static AVCUDADeviceContext *device(AVBufferRef *frames)
{
    if (!frames) return NULL;
    AVHWFramesContext *f = (void *)frames->data;
    return f->device_ctx->type == AV_HWDEVICE_TYPE_CUDA ? f->device_ctx->hwctx : NULL;
}
struct ajn_sample_cuda *ajn_sample_cuda_create(AVBufferRef *frames, int width, int height, bool mask,
    aji_observer_cancel_fn cancel, aji_observer_state_fn state, void *opaque, char *error, size_t error_size)
{
    struct ajn_sample_cuda *p = calloc(1,sizeof(*p));
    if (!p) { av_buffer_unref(&frames); return NULL; }
    p->frames = frames; p->width = width; p->height = height;
    AVCUDADeviceContext *ctx = device(frames);
    wchar_t root[32768], path[32768]; HMODULE self;
    if (!ctx || !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCWSTR)(uintptr_t)ajn_sample_cuda_create, &self)) goto fail;
    DWORD length = GetModuleFileNameW(self,root,32768);
    if (!length || length >= 32768) goto fail;
    wchar_t *slash = wcsrchr(root,L'\\'); if (!slash) goto fail; *slash = 0;
    if (swprintf(path,32768,L"%ls\\animejanai\\inference\\aji_analysis.dll",root)<0) goto fail;
    p->module = LoadLibraryExW(path,NULL,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!p->module) { snprintf(error,error_size,"Optional analysis runtime is unavailable (Windows error %lu). Install the matching TensorRT component.",GetLastError()); goto fail; }
    *(void **)&p->create = (void *)GetProcAddress(p->module,"aji_observer_create");
    *(void **)&p->run = (void *)GetProcAddress(p->module,"aji_observer_run");
    *(void **)&p->destroy = (void *)GetProcAddress(p->module,"aji_observer_destroy");
    if (!p->create || !p->run || !p->destroy) goto fail;
    char *model = NULL;
    if (mask) {
        if (swprintf(path,32768,L"%ls\\animejanai\\analysis\\isnetis.f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99.onnx",root)<0) goto fail;
        int bytes = WideCharToMultiByte(CP_UTF8,0,path,-1,NULL,0,NULL,NULL);
        if (!bytes || !(model = malloc(bytes))) goto fail;
        WideCharToMultiByte(CP_UTF8,0,path,-1,model,bytes,NULL,NULL);
    }
    p->observer = p->create(ctx->cuda_ctx,width,height,model,cancel,state,opaque,error,error_size);
    free(model);
    if (!p->observer) goto fail;
    return p;
fail:
    ajn_sample_cuda_destroy(p); return NULL;
}
bool ajn_sample_cuda_matches(struct ajn_sample_cuda *p, struct mp_image *image, int width, int height)
{
    AVCUDADeviceContext *ctx = device(image->hwctx);
    return p && ctx && ctx->cuda_ctx == device(p->frames)->cuda_ctx && width == p->width && height == p->height;
}
HRESULT ajn_sample_cuda_read(struct ajn_sample_cuda *p, struct mp_image *image, const RECT *crop,
    unsigned char *color, uint16_t *mask, double *gpu_ms)
{
    AVHWFramesContext *frames = (void *)image->hwctx->data;
    enum AVPixelFormat sw = frames->sw_format;
    int format = sw == AV_PIX_FMT_NV12 ? AJI_FMT_NV12 : sw == AV_PIX_FMT_P010 ? AJI_FMT_P010 :
        sw == AV_PIX_FMT_YUV444P16 ? AJI_FMT_YUV444P16 :
#ifdef AV_PIX_FMT_YUV444P10MSB
        sw == AV_PIX_FMT_YUV444P10MSB ? AJI_FMT_YUV444P10MSB :
        sw == AV_PIX_FMT_YUV444P12MSB ? AJI_FMT_YUV444P12MSB :
#endif
        0;
    if (!format) return E_NOTIMPL;
    aji_frame input = { .width=image->w, .height=image->h, .format=format,
        .matrix=image->params.repr.sys==PL_COLOR_SYSTEM_BT_709?AJI_MATRIX_BT709:AJI_MATRIX_BT601,
        .range=image->params.repr.levels==PL_COLOR_LEVELS_FULL?AJI_RANGE_FULL:AJI_RANGE_LIMITED,
        .siting=image->params.chroma_location==PL_CHROMA_CENTER?AJI_SITING_CENTER:
            image->params.chroma_location==PL_CHROMA_TOP_LEFT?AJI_SITING_TOPLEFT:AJI_SITING_LEFT,
        .plane={image->planes[0],image->planes[1],image->planes[2]}, .stride={image->stride[0],image->stride[1],image->stride[2]} };
    int rect[4]={crop->left,crop->top,crop->right-crop->left,crop->bottom-crop->top};
    return p->run(p->observer,&input,rect,device(image->hwctx)->stream,color,mask,gpu_ms)==AJI_OK?S_OK:E_FAIL;
}
void ajn_sample_cuda_destroy(struct ajn_sample_cuda *p)
{
    if(!p)return;
    if(p->observer)p->destroy(p->observer);
    av_buffer_unref(&p->frames);
    if(p->module)FreeLibrary(p->module);
    free(p);
}
