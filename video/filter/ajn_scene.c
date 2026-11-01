/* Bounded scene-decision bridge for sandboxed addon algorithms.
 * SPDX-License-Identifier: LGPL-2.1-or-later */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>
#include <windows.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
#include "common/common.h"
#include "video/mp_image.h"
#include "video/mp_image_pool.h"
#include "video/sws_utils.h"
#include "video/fmt-conversion.h"
#include "ajn_scene.h"
#include "ajn_scene_shared.h"
#include "mvtools/ajn_mvtools.h"

struct ajn_scene {
    struct ajn_scene_shared *shared;
    HANDLE ready, reply;
    char name[96];
    struct mp_sws_context *sws;
    struct mp_image_pool *pool;
    uint8_t pixels[AJN_SCENE_BYTES], cached[AJN_SCENE_PLANE];
    double cached_pts;
    int cached_w, cached_h, source_w, source_h;
    HMODULE mv_module;
    void *mv_context;
    bool mv_load_failed;
    int64_t mv_generation;
    struct mp_image *mv_cached;
    void *(*mv_create)(void);
    void (*mv_destroy)(void *);
    void (*mv_reset)(void *);
    int (*mv_analyze)(void *, const struct ajn_mvtools_frame *,
                      const struct ajn_mvtools_frame *, int, int,
                      struct ajn_mvtools_result *);
    int (*mv_enhanced)(void *, const struct ajn_mvtools_frame *,
                       const struct ajn_mvtools_frame *, int, int,
                       struct ajn_mvtools_result *);
};
static void mv_close(struct ajn_scene *s)
{
    if (s->mv_context) s->mv_destroy(s->mv_context);
    s->mv_context = NULL;
    if (s->mv_module) FreeLibrary(s->mv_module);
    s->mv_module = NULL;
    TA_FREEP(&s->mv_cached);
    s->mv_load_failed = false;
}
static bool mv_open(struct ajn_scene *s)
{
    if (s->mv_context) return true;
    if (s->mv_load_failed) return false;
    // Fixed trusted installation path. Never load a DLL supplied by an addon,
    // nor search PATH/current directory. Called only for an enabled MVTools mode.
    wchar_t path[32768];
    DWORD n = GetModuleFileNameW(NULL, path, MP_ARRAY_SIZE(path));
    wchar_t *slash = n && n < MP_ARRAY_SIZE(path) ? wcsrchr(path, L'\\') : NULL;
    if (!slash || slash - path + 20 >= MP_ARRAY_SIZE(path)) goto fail;
    wcscpy(slash + 1, L"ajn-mvtools.dll");
    s->mv_module = LoadLibraryExW(path, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!s->mv_module) goto fail;
    int (*version)(void) = (void *)GetProcAddress(s->mv_module, "ajn_mvtools_version");
    s->mv_create = (void *)GetProcAddress(s->mv_module, "ajn_mvtools_create");
    s->mv_destroy = (void *)GetProcAddress(s->mv_module, "ajn_mvtools_destroy");
    s->mv_reset = (void *)GetProcAddress(s->mv_module, "ajn_mvtools_reset");
    s->mv_analyze = (void *)GetProcAddress(s->mv_module, "ajn_mvtools_analyze");
    s->mv_enhanced = (void *)GetProcAddress(s->mv_module, "ajn_mvtools_analyze_enhanced");
    if (!version || version() != AJN_MVTOOLS_ABI || !s->mv_create ||
        !s->mv_destroy || !s->mv_reset || !s->mv_analyze) goto fail;
    s->mv_context = s->mv_create();
    if (s->mv_context) return true;
fail:
    mv_close(s); s->mv_load_failed = true; return false;
}
static bool enter(struct ajn_scene_shared *s)
{
    return !InterlockedCompareExchange((volatile LONG *)&s->lock, 1, 0);
}
static void leave(struct ajn_scene_shared *s)
{
    InterlockedExchange((volatile LONG *)&s->lock, 0);
}
bool ajn_scene_active(struct ajn_scene *s)
{
    if (!s || !s->shared || !enter(s->shared)) return false;
    struct ajn_scene_shared *h = s->shared;
    bool active = h->enabled && h->lease_until_ms > (int64_t)GetTickCount64() &&
                  h->consecutive_timeouts < 3;
    leave(h);
    return active;
}
struct ajn_scene *ajn_scene_create(struct mp_log *log)
{
    struct ajn_scene *s = talloc_zero(NULL, struct ajn_scene);
    s->sws = mp_sws_alloc(s);
    s->sws->log = log;
    s->sws->flags = SWS_BILINEAR;
    s->pool = mp_image_pool_new(s);
    s->cached_pts = MP_NOPTS_VALUE;
    return s;
}
void ajn_scene_reset(struct ajn_scene *s)
{
    if (!s) return;
    s->cached_pts = MP_NOPTS_VALUE;
    if (s->mv_context) s->mv_reset(s->mv_context);
    TA_FREEP(&s->mv_cached);
    if (s->shared) {
        // A seek must invalidate pending replies even while the host copies
        // the sample under the header lock. The producer itself is serial.
        InterlockedIncrement64((volatile LONG64 *)&s->shared->epoch);
        InterlockedExchange64((volatile LONG64 *)&s->shared->request_id, 0);
        InterlockedExchange64((volatile LONG64 *)&s->shared->response_id, 0);
    }
}
bool ajn_scene_connect(struct ajn_scene *s, const char *name)
{
    if (!s || !name) return false;
    if (!strcmp(s->name, name)) return true;
    const char *prefix = "Local\\AJN.Scene.";
    size_t prefix_len = strlen(prefix);
    if (*name && (strlen(name) != prefix_len + 32 ||
        strncmp(name, prefix, prefix_len) ||
        strspn(name + prefix_len, "0123456789abcdef") != 32)) return false;
    ajn_scene_reset(s);
    mv_close(s);
    if (s->shared) UnmapViewOfFile(s->shared);
    if (s->ready) CloseHandle(s->ready);
    if (s->reply) CloseHandle(s->reply);
    s->shared = NULL; s->ready = s->reply = NULL; s->name[0] = 0;
    if (!*name) return true;
    wchar_t wide[128];
    MultiByteToWideChar(CP_UTF8, 0, name, -1, wide, MP_ARRAY_SIZE(wide));
    HANDLE mapping = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, wide);
    if (!mapping) return false;
    s->shared = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(*s->shared));
    CloseHandle(mapping);
    if (!s->shared) return false;
    struct ajn_scene_shared *h = s->shared;
    if (h->magic != AJN_SCENE_MAGIC || h->version != AJN_SCENE_VERSION ||
        h->capacity != AJN_SCENE_BYTES || h->header_bytes != 256) goto fail;
    wcscat(wide, L".Ready");
    s->ready = OpenEventW(EVENT_MODIFY_STATE, FALSE, wide);
    wide[wcslen(wide) - 6] = 0; wcscat(wide, L".Reply");
    s->reply = OpenEventW(SYNCHRONIZE, FALSE, wide);
    if (!s->ready || !s->reply) goto fail;
    snprintf(s->name, sizeof(s->name), "%s", name);
    ajn_scene_reset(s);
    return true;
fail:
    UnmapViewOfFile(s->shared); s->shared = NULL;
    if (s->ready) CloseHandle(s->ready);
    if (s->reply) CloseHandle(s->reply);
    s->ready = s->reply = NULL;
    return false;
}
void ajn_scene_destroy(struct ajn_scene *s)
{
    if (!s) return;
    ajn_scene_connect(s, "");
    talloc_free(s);
}
static bool sample(struct ajn_scene *s, struct mp_image *image,
                   int width, int height, uint8_t *pixels)
{
    if (image->fields & MP_IMGFIELD_INTERLACED || image->params.stereo3d ||
        image->params.color.transfer == PL_COLOR_TRC_PQ ||
        image->params.color.transfer == PL_COLOR_TRC_HLG) return false;
    struct mp_image *cpu = image->hwctx ? mp_image_hw_download(image, s->pool) :
                                        mp_image_new_ref(image);
    struct mp_image *small = mp_image_alloc(IMGFMT_Y8, width, height);
    if (!cpu || !small) { talloc_free(cpu); talloc_free(small); return false; }
    small->params.color = cpu->params.color;
    small->params.repr = cpu->params.repr;
    small->params.repr.levels = PL_COLOR_LEVELS_FULL;
    bool ok = mp_sws_scale(s->sws, small, cpu) >= 0;
    if (ok) {
        for (int y = 0; y < height; y++)
            memcpy(pixels + y * width, small->planes[0] + y * small->stride[0], width);
    }
    talloc_free(cpu); talloc_free(small);
    return ok;
}
static struct mp_image *mv_frame(struct ajn_scene *s, struct mp_image *image)
{
    if (image->fields & MP_IMGFIELD_INTERLACED || image->params.stereo3d ||
        image->params.color.transfer == PL_COLOR_TRC_PQ ||
        image->params.color.transfer == PL_COLOR_TRC_HLG) return NULL;
    struct mp_image *cpu = image->hwctx ? mp_image_hw_download(image, s->pool) :
                                        mp_image_new_ref(image);
    if (!cpu) return NULL;
    const AVPixFmtDescriptor *fmt = av_pix_fmt_desc_get(imgfmt2pixfmt(cpu->imgfmt));
    if (!fmt || (fmt->flags & (AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_FLOAT)) ||
        (fmt->nb_components != 1 && fmt->nb_components != 3) ||
        fmt->log2_chroma_w > 1 || fmt->log2_chroma_h > 1 ||
        fmt->comp[0].depth < 8 || fmt->comp[0].depth > 16) {
        talloc_free(cpu); return NULL;
    }
    int bits = fmt->comp[0].depth;
    char format[32];
    if (fmt->nb_components == 1) {
        if (bits == 8) snprintf(format, sizeof(format), "gray");
        else snprintf(format, sizeof(format), "gray%dle", bits);
    }
    else {
        const char *sub = fmt->log2_chroma_w ? (fmt->log2_chroma_h ? "420" : "422") :
                                             (fmt->log2_chroma_h ? "440" : "444");
        if (bits == 8) snprintf(format, sizeof(format), "yuv%sp", sub);
        else snprintf(format, sizeof(format), "yuv%sp%dle", sub, bits);
    }
    int target = pixfmt2imgfmt(av_get_pix_fmt(format));
    if (cpu->imgfmt == target) { cpu->pts = image->pts; return cpu; }
    struct mp_image *planar = target ? mp_image_alloc(target, cpu->w, cpu->h) : NULL;
    if (!planar) { talloc_free(cpu); return NULL; }
    // Unpack NV12/P010 without resizing, changing range or dropping chroma.
    planar->params.color = cpu->params.color;
    planar->params.repr = cpu->params.repr;
    planar->params.chroma_location = cpu->params.chroma_location;
    bool ok = mp_sws_scale(s->sws, planar, cpu) >= 0;
    talloc_free(cpu);
    if (!ok) { talloc_free(planar); return NULL; }
    planar->pts = image->pts;
    return planar;
}
static struct ajn_mvtools_frame mv_description(struct mp_image *image)
{
    const AVPixFmtDescriptor *fmt = av_pix_fmt_desc_get(imgfmt2pixfmt(image->imgfmt));
    struct ajn_mvtools_frame f = { .width = image->w, .height = image->h,
        .bits = fmt->comp[0].depth, .sub_w = fmt->log2_chroma_w,
        .sub_h = fmt->log2_chroma_h, .pts = image->pts };
    for (int p = 0; p < fmt->nb_components; p++) {
        f.planes[p] = image->planes[p]; f.strides[p] = image->stride[p];
    }
    return f;
}
static int mv_decide(struct ajn_scene *s, struct mp_image *a, struct mp_image *b,
                     int64_t generation, int th1, int th2, bool enhanced)
{
    LARGE_INTEGER start, end, frequency;
    QueryPerformanceCounter(&start); QueryPerformanceFrequency(&frequency);
    int state = 7, decision = -1;
    struct ajn_mvtools_result result = {0};
    struct mp_image *previous = NULL, *current = NULL;
    int bits = 0;
    if (!mv_open(s)) goto done;
    if (enhanced && !s->mv_enhanced) goto done;
    if (s->mv_generation != generation) {
        s->mv_reset(s->mv_context); TA_FREEP(&s->mv_cached);
        s->mv_generation = generation;
    }
    if (s->mv_cached && s->mv_cached->pts == a->pts &&
        s->mv_cached->w == a->w && s->mv_cached->h == a->h)
        previous = mp_image_new_ref(s->mv_cached);
    else previous = mv_frame(s, a);
    current = mv_frame(s, b);
    state = 4;
    if (!previous || !current) goto done;
    struct ajn_mvtools_frame fa = mv_description(previous), fb = mv_description(current);
    bits = fb.bits;
    int error = enhanced ? s->mv_enhanced(s->mv_context, &fa, &fb, th1, th2, &result) :
                           s->mv_analyze(s->mv_context, &fa, &fb, th1, th2, &result);
    state = error == -1 ? 4 : error ? 8 : 2;
    if (!error) decision = result.cut;
    TA_FREEP(&s->mv_cached);
    s->mv_cached = current; current = NULL;
done:
    talloc_free(previous); talloc_free(current);
    QueryPerformanceCounter(&end);
    struct ajn_scene_shared *h = s->shared;
    if (!enter(h)) return -1;
    // Disable, seek or reconfiguration must not consume an in-flight decision.
    if (!h->enabled || h->generation != generation ||
        h->lease_until_ms <= (int64_t)GetTickCount64()) { leave(h); return -1; }
    h->status = state; h->source_width = b->w; h->source_height = b->h;
    h->previous_pts = a->pts; h->current_pts = b->pts;
    h->changed_blocks = result.changed_blocks; h->total_blocks = result.total_blocks;
    h->block_sad_threshold = result.block_sad_threshold; h->source_bits = bits;
    h->analysis_us = (end.QuadPart - start.QuadPart) * 1000000 / frequency.QuadPart;
    if (decision >= 0 && h->accepted < INT32_MAX) h->accepted++;
    if (decision == 1 && h->cut_pairs < INT32_MAX) h->cut_pairs++;
    leave(h);
    return decision;
}
int ajn_scene_decide(struct ajn_scene *s, struct mp_image *a, struct mp_image *b,
                     bool supported)
{
    if (!s || !s->shared) return -1;
    struct ajn_scene_shared *h = s->shared;
    if (!enter(h)) return -1;
    int64_t now = GetTickCount64(), generation = h->generation;
    int width = h->width, height = h->height, budget = h->budget_ms;
    int algorithm = h->algorithm, th1 = h->thscd1, th2 = h->thscd2;
    bool active = h->enabled && h->lease_until_ms > now && h->consecutive_timeouts < 3;
    if (active && !supported) h->status = 5;
    leave(h);
    if (!active || !supported || width < 1 || width > AJN_SCENE_WIDTH ||
        height < 1 || height > AJN_SCENE_HEIGHT || budget < 5 || budget > 100 ||
        !isfinite(a->pts) || !isfinite(b->pts) || a->pts == MP_NOPTS_VALUE ||
        b->pts <= a->pts || a->w != b->w || a->h != b->h) return -1;
    if (algorithm == 1 || algorithm == 2)
        return mv_decide(s, a, b, generation, th1, th2, algorithm == 2);
    if (algorithm != 0) return -1;
    int bytes = width * height;
    bool cached = s->cached_pts == a->pts && s->cached_w == width &&
        s->cached_h == height && s->source_w == a->w && s->source_h == a->h;
    if (cached) memcpy(s->pixels, s->cached, bytes);
    if ((!cached && !sample(s, a, width, height, s->pixels)) ||
        !sample(s, b, width, height, s->pixels + bytes)) {
        if (enter(h)) { h->status = 4; leave(h); }
        s->cached_pts = MP_NOPTS_VALUE;
        return -1;
    }
    memcpy(s->cached, s->pixels + bytes, bytes);
    s->cached_pts = b->pts; s->cached_w = width; s->cached_h = height;
    s->source_w = b->w; s->source_h = b->h;
    if (!enter(h)) return -1;
    now = GetTickCount64();
    if (!h->enabled || h->generation != generation || h->lease_until_ms <= now ||
        h->next_id == INT64_MAX) { leave(h); return -1; }
    int64_t request = ++h->next_id, deadline = now + budget;
    h->request_id = request; h->response_id = 0;
    h->deadline_ms = deadline; h->decision = -1; h->status = 1;
    h->previous_pts = a->pts; h->current_pts = b->pts;
    h->source_width = b->w; h->source_height = b->h;
    memcpy(h->pixels, s->pixels, 2 * bytes);
    leave(h);
    SetEvent(s->ready);
    while ((now = GetTickCount64()) < deadline) {
        WaitForSingleObject(s->reply, (DWORD)(deadline - now));
        if (!enter(h)) continue;
        if (!h->enabled || h->generation != generation || h->request_id != request) {
            leave(h); return -1;
        }
        if (h->response_id == request && h->deadline_ms >= (int64_t)GetTickCount64()) {
            int decision = h->decision;
            h->status = 2;
            if (h->accepted < INT32_MAX) h->accepted++;
            h->consecutive_timeouts = 0;
            leave(h);
            return decision >= -1 && decision <= 1 ? decision : -1;
        }
        leave(h);
    }
    if (enter(h)) {
        if (h->request_id == request) {
            if (h->timeouts < INT32_MAX) h->timeouts++;
            h->consecutive_timeouts++;
            h->status = h->consecutive_timeouts >= 3 ? 6 : 3;
            h->deadline_ms = 0;
        }
        leave(h);
    }
    return -1;
}
