/* Private GPU sample branch for supervised AJN media workers.
 * SPDX-License-Identifier: LGPL-2.1-or-later */
#include <limits.h>
#include <math.h>
#include <windows.h>
#include "config.h"
#include "common/common.h"
#include "filters/filter.h"
#include "filters/filter_internal.h"
#include "filters/user_filters.h"
#include "options/m_option.h"
#include "osdep/threads.h"
#include "osdep/timer.h"
#include "video/mp_image.h"
#include "ajn_sample_shared.h"
#include "ajn_sample_gpu.h"
#include "ajn_sample_lifetime.h"
#if HAVE_CUDA_HWACCEL
#include "ajn_sample_cuda.h"
#endif

struct opts { int64_t mapping; char *name; };
static LONG player_buffers;
static LONG player_generation;
struct priv {
    struct opts *opts;
    struct ajn_sample_shared *shared;
    mp_mutex lock;
    mp_thread thread;
    HANDLE wake;
    bool started, stopping, failed, detached, reserved;
    struct ajn_sample_lifetime lifetime;
    int64_t producer;
    struct ajn_sample_gpu *gpu;
#if HAVE_CUDA_HWACCEL
    struct ajn_sample_cuda *cuda;
#endif
    ID3D11Device *device;
    DXGI_FORMAT format;
    UINT coded_w, coded_h;
    int width, height;
    struct mp_image *pending;
    int64_t pending_config, pending_epoch, last_submit;
    uint32_t primaries, transfer;
    RECT crop;
    uint8_t scratch[AJN_SAMPLE_BYTES];
    uint16_t *mask;
    int64_t gpu_us, submitted_ns;
    char cuda_error[384];
    bool reported_error;
};

static void cleanup(struct priv *p);
static void release(struct priv *p)
{
    if (ajn_sample_lifetime_release(&p->lifetime)) {
        cleanup(p);
        talloc_free(p);
    }
}

static int64_t read64(int64_t *p)
{
    return InterlockedCompareExchange64((volatile LONG64 *)p, 0, 0);
}

static void state(struct priv *p, enum ajn_sample_status status, HRESULT error)
{
    if (p->detached) {
        if (read64(&p->shared->producer) != p->producer) return;
        InterlockedExchange64((volatile LONG64 *)&p->shared->player_status,
            (p->producer << 32) | status);
    }
    InterlockedExchange((volatile LONG *)&p->shared->error, error);
    InterlockedExchange((volatile LONG *)&p->shared->status, status);
}

static void drop(struct priv *p)
{
    InterlockedIncrement((volatile LONG *)&p->shared->dropped);
}

static bool color_supported(struct priv *p, const struct mp_image *image)
{
    const struct mp_image_params *q = &image->params;
    // Preserve encoded SDR primaries/transfer. Only the matrix/range changes to
    // full-range RGB. HDR and unusual matrices need a separately negotiated tap.
    if (q->repr.sys != PL_COLOR_SYSTEM_BT_601 && q->repr.sys != PL_COLOR_SYSTEM_BT_709 &&
        q->repr.sys != PL_COLOR_SYSTEM_RGB) return false;
    if (q->repr.levels != PL_COLOR_LEVELS_FULL && q->repr.levels != PL_COLOR_LEVELS_LIMITED) return false;
    switch (q->color.primaries) {
    case PL_COLOR_PRIM_BT_709: p->primaries = AJN_PRIM_709; break;
    case PL_COLOR_PRIM_BT_601_525: p->primaries = AJN_PRIM_601_525; break;
    case PL_COLOR_PRIM_BT_601_625: p->primaries = AJN_PRIM_601_625; break;
    default: return false;
    }
    switch (q->color.transfer) {
    case PL_COLOR_TRC_BT_1886: p->transfer = AJN_TRC_1886; break;
    case PL_COLOR_TRC_SRGB: p->transfer = AJN_TRC_SRGB; break;
    case PL_COLOR_TRC_GAMMA22: p->transfer = AJN_TRC_G22; break;
    case PL_COLOR_TRC_GAMMA24: p->transfer = AJN_TRC_G24; break;
    default: return false;
    }
    return !(image->fields & MP_IMGFIELD_INTERLACED) && q->stereo3d == 0;
}

static void publish(struct priv *p)
{
    struct ajn_sample_shared *s = p->shared;
    const struct mp_image *image = p->pending;
    if (InterlockedCompareExchange((volatile LONG *)&s->writer, 1, 0)) {
        drop(p);
        return;
    }
    if (p->pending_config != read64(&s->config) ||
        p->pending_epoch != read64(&s->epoch) ||
        (p->detached && (ajn_sample_lifetime_stopping(&p->lifetime) ||
            read64(&s->producer) != p->producer ||
            read64(&s->lease_until_ms) <= (int64_t)GetTickCount64()))) {
        InterlockedExchange((volatile LONG *)&s->writer, 0);
        drop(p);
        return;
    }
    InterlockedIncrement64((volatile LONG64 *)&s->sequence);
    s->frame_id++;
    s->sample_config = p->pending_config;
    s->sample_epoch = p->pending_epoch;
    double pts = image->pts * 1000000.0;
    s->pts_us = isfinite(pts) && pts > (double)INT64_MIN && pts < (double)INT64_MAX
        ? (int64_t)pts : INT64_MIN;
    s->produced_ns = mp_time_ns();
    s->width = p->width; s->height = p->height; s->stride = p->width * 4;
    s->payload_bytes = s->stride * s->height;
    s->source_width = image->w; s->source_height = image->h;
    s->primaries = p->primaries; s->transfer = p->transfer;
    s->rotation = image->params.rotate;
    s->par_num = image->params.p_w; s->par_den = image->params.p_h;
    s->crop_x = p->crop.left; s->crop_y = p->crop.top;
    s->crop_width = p->crop.right - p->crop.left;
    s->crop_height = p->crop.bottom - p->crop.top;
    s->vflip = image->params.vflip;
    memcpy(s->pixels, p->scratch, s->payload_bytes);
    if (p->mask) {
        memcpy(s->pixels + AJN_SAMPLE_BYTES, p->mask, p->width * p->height * 2);
        s->gpu_us = p->gpu_us;
        s->age_us = (mp_time_ns() - p->submitted_ns) / 1000;
    }
    InterlockedIncrement64((volatile LONG64 *)&s->sequence);
    InterlockedExchange((volatile LONG *)&s->writer, 0);
    state(p, AJN_SAMPLE_READY, S_OK);
}

static int cancelled(void *context)
{
    struct priv *p = context;
    return p->detached ? ajn_sample_lifetime_stopping(&p->lifetime) ||
        read64(&p->shared->lease_until_ms) <= (int64_t)GetTickCount64() ||
        read64(&p->shared->producer) != p->producer || !(read64(&p->shared->config) & 0xffffff) : p->stopping;
}

static void analysis_state(void *context, int status)
{
    state(context, status, S_OK);
}

static MP_THREAD_VOID readback(void *context)
{
    struct priv *p = context;
    mp_thread_set_name("ajn-sample");
    ULONGLONG retired_at = 0;
    for (;;) {
        mp_mutex_lock(&p->lock);
        bool stopping = p->detached ? ajn_sample_lifetime_stopping(&p->lifetime) : p->stopping;
        // A lost host lease must retire analysis even if the Lua bridge stops
        // running. Only the optional worker observes this lease.
        if (p->detached && cancelled(p)) stopping = true;
        if (stopping && !retired_at) retired_at = GetTickCount64();
        if (p->pending && (!p->failed || p->detached)) {
            HRESULT hr;
#if HAVE_CUDA_HWACCEL
            bool cuda_frame = p->pending->imgfmt == IMGFMT_CUDA;
            if (cuda_frame && !ajn_sample_cuda_matches(p->cuda, p->pending, p->width, p->height)) {
                // Retain only the device context during setup, not a decoder
                // surface. All model loading/building is off the playback thread.
                AVBufferRef *frames = av_buffer_ref(p->pending->hwctx);
                TA_FREEP(&p->pending);
                state(p, AJN_SAMPLE_LOADING, S_OK);
                ajn_sample_cuda_destroy(p->cuda); p->cuda = NULL;
                if (frames && !cancelled(p)) p->cuda = ajn_sample_cuda_create(frames, p->width, p->height,
                    p->mask != NULL, cancelled, analysis_state, p, p->cuda_error, sizeof(p->cuda_error));
                else av_buffer_unref(&frames);
                if (!p->cuda) { p->failed = true; state(p, AJN_SAMPLE_FAILED, E_FAIL); }
                else state(p, AJN_SAMPLE_IDLE, S_OK);
                mp_mutex_unlock(&p->lock);
                continue;
            }
            if (cuda_frame) {
                double gpu_ms = 0;
                hr = ajn_sample_cuda_read(p->cuda, p->pending, &p->crop, p->scratch, p->mask, &gpu_ms);
                p->gpu_us = gpu_ms * 1000;
            } else
#endif
                hr = ajn_sample_gpu_read(p->gpu, p->scratch);
            if (FAILED(hr)) {
                // Preserve the source until completion or confirmed device
                // removal. Player retirement cannot free a decoder-pool image
                // while the GPU may still be reading it.
                p->failed = hr != E_NOTIMPL;
                state(p, hr == E_NOTIMPL ? AJN_SAMPLE_UNSUPPORTED : AJN_SAMPLE_FAILED, hr);
#if HAVE_CUDA_HWACCEL
                if (cuda_frame) TA_FREEP(&p->pending); // private stream was drained
                else
#endif
                if (p->detached && FAILED(ID3D11Device_GetDeviceRemovedReason(p->device)))
                    TA_FREEP(&p->pending);
            } else if (hr == S_OK) {
                if (p->pending_config == read64(&p->shared->config) &&
                    p->pending_epoch == read64(&p->shared->epoch)) publish(p);
                else drop(p);
                TA_FREEP(&p->pending);
            }
        }
        if (stopping && (!p->pending || (p->failed && !p->detached))) {
            TA_FREEP(&p->pending);
            mp_mutex_unlock(&p->lock);
            break;
        }
        bool poll = p->pending && (!p->failed || p->detached);
        mp_mutex_unlock(&p->lock);
        WaitForSingleObject(p->wake, poll ? (retired_at && GetTickCount64() - retired_at > 250 ? 100 : 1) : (p->detached ? 500 : INFINITE));
    }
    if (p->detached) release(p);
    MP_THREAD_RETURN();
}

static void sample(struct mp_filter *f, struct mp_image *image)
{
    struct priv *p = f->priv;
    if (!p->shared || !p->started) return;
    int width, height, fps;
    int64_t config = read64(&p->shared->config);
    if (!ajn_sample_config(config, &width, &height, &fps) ||
        (p->detached && read64(&p->shared->lease_until_ms) <= (int64_t)GetTickCount64())) {
        state(p, AJN_SAMPLE_IDLE, S_OK);
        return;
    }
    int64_t now = mp_time_ns();
    if (now - p->last_submit < 1000000000 / fps) return;
    if (mp_mutex_trylock(&p->lock)) { drop(p); return; }
    if (p->pending || p->failed) {
        if (p->failed && !p->reported_error && p->cuda_error[0]) { MP_WARN(f,"Optional observation failed: %s\n",p->cuda_error); p->reported_error = true; }
        drop(p); goto done;
    }
    bool cuda_frame = false;
#if HAVE_CUDA_HWACCEL
    cuda_frame = image->imgfmt == IMGFMT_CUDA;
#endif
    if ((!cuda_frame && (image->imgfmt != IMGFMT_D3D11 || p->mask)) || !color_supported(p, image)) {
        state(p, AJN_SAMPLE_UNSUPPORTED, E_NOTIMPL);
        goto done;
    }
    if (cuda_frame) {
        if (!image->hwctx || image->params.repr.sys == PL_COLOR_SYSTEM_RGB) {
            state(p, AJN_SAMPLE_UNSUPPORTED, E_NOTIMPL); goto done;
        }
        p->width = width; p->height = height;
        p->pending = mp_image_new_ref(image);
        if (!p->pending) { state(p, AJN_SAMPLE_FAILED, E_OUTOFMEMORY); goto done; }
        p->pending_config = config; p->pending_epoch = read64(&p->shared->epoch);
        struct mp_rect crop = mp_image_crop_valid(&image->params) ? image->params.crop : (struct mp_rect){0,0,image->w,image->h};
        p->crop = (RECT){crop.x0,crop.y0,crop.x1,crop.y1};
        p->submitted_ns = p->last_submit = now;
        state(p, AJN_SAMPLE_PENDING, S_OK); SetEvent(p->wake); goto done;
    }
    ID3D11Texture2D *texture = (void *)image->planes[0];
    D3D11_TEXTURE2D_DESC desc;
    ID3D11Texture2D_GetDesc(texture, &desc);
    if (image->w <= 0 || image->h <= 0 || image->w > desc.Width || image->h > desc.Height ||
        (uintptr_t)image->planes[1] >= desc.ArraySize) {
        state(p, AJN_SAMPLE_UNSUPPORTED, E_INVALIDARG);
        goto done;
    }
    ID3D11Device *device;
    ID3D11Texture2D_GetDevice(texture, &device);
    HRESULT hr = S_OK;
    if (!p->gpu || p->device != device || p->coded_w != desc.Width || p->coded_h != desc.Height ||
        p->format != desc.Format || p->width != width || p->height != height) {
        ajn_sample_gpu_destroy(p->gpu); p->gpu = NULL;
        if (p->device) ID3D11Device_Release(p->device);
        p->device = device; ID3D11Device_AddRef(device);
        p->coded_w = desc.Width; p->coded_h = desc.Height; p->format = desc.Format;
        p->width = width; p->height = height;
        p->gpu = ajn_sample_gpu_create(device, desc.Width, desc.Height, desc.Format, width, height, &hr);
    }
    ID3D11Device_Release(device);
    if (!p->gpu) {
        p->failed = true; state(p, AJN_SAMPLE_FAILED, hr);
        MP_WARN(f, "Small GPU sample unavailable (0x%08lx). Video continues.\n", (unsigned long)hr);
        goto done;
    }
    // The inference output pool must not reuse the texture while this branch
    // reads it. Hold an mp_image reference through GPU completion, even on seek.
    p->pending = mp_image_new_ref(image);
    if (!p->pending) { state(p, AJN_SAMPLE_FAILED, E_OUTOFMEMORY); goto done; }
    p->pending_config = config;
    p->pending_epoch = read64(&p->shared->epoch);
    struct mp_rect crop = mp_image_crop_valid(&image->params) ? image->params.crop : (struct mp_rect){0, 0, image->w, image->h};
    p->crop = (RECT){crop.x0, crop.y0, crop.x1, crop.y1};
    hr = ajn_sample_gpu_submit(p->gpu, texture, (uintptr_t)image->planes[1], &p->crop,
        image->params.repr.sys == PL_COLOR_SYSTEM_BT_709, image->params.repr.levels == PL_COLOR_LEVELS_FULL);
    if (FAILED(hr)) {
        TA_FREEP(&p->pending); p->failed = true; state(p, AJN_SAMPLE_FAILED, hr);
    } else {
        p->last_submit = now;
        state(p, AJN_SAMPLE_PENDING, S_OK);
        SetEvent(p->wake);
    }
done:
    mp_mutex_unlock(&p->lock);
}

static void process(struct mp_filter *f)
{
    if (!mp_pin_can_transfer_data(f->ppins[1], f->ppins[0])) return;
    struct mp_frame frame = mp_pin_out_read(f->ppins[0]);
    if (frame.type == MP_FRAME_VIDEO) sample(f, frame.data);
    // Preserve full-resolution pixels, attributes, timing and EOF unchanged.
    mp_pin_in_write(f->ppins[1], frame);
}

static void reset(struct mp_filter *f)
{
    struct priv *p = f->priv;
    if (p->shared) InterlockedIncrement64((volatile LONG64 *)&p->shared->epoch);
    p->last_submit = 0;
}

static bool command(struct mp_filter *f, struct mp_filter_command *cmd)
{
    if (cmd->type != MP_FILTER_COMMAND_TEXT || !cmd->cmd || strcmp(cmd->cmd, "invalidate")) return false;
    reset(f);
    return true;
}

static void cleanup(struct priv *p)
{
#if HAVE_CUDA_HWACCEL
    ajn_sample_cuda_destroy(p->cuda);
#endif
    free(p->mask);
    ajn_sample_gpu_destroy(p->gpu);
    if (p->device) ID3D11Device_Release(p->device);
    if (p->wake) CloseHandle(p->wake);
    if (p->shared) { state(p, AJN_SAMPLE_CLOSED, S_OK); UnmapViewOfFile(p->shared); }
    mp_mutex_destroy(&p->lock);
    if (p->reserved) InterlockedDecrement(&player_buffers);
}

static void destroy(struct mp_filter *f)
{
    struct priv *p = f->priv;
    if (p->detached) {
        // The owner reference keeps p/wake valid until SetEvent completes even
        // if the thread finishes concurrently. No mutex or driver wait here.
        ajn_sample_lifetime_stop(&p->lifetime);
        if (p->started) { SetEvent(p->wake); mp_thread_detach(p->thread); }
        release(p);
        return;
    }
    if (p->started) {
        mp_mutex_lock(&p->lock); p->stopping = true; mp_mutex_unlock(&p->lock);
        SetEvent(p->wake);
        // Ordinary shutdown drains the sole GPU sample. The supervising native
        // process has a bounded shutdown deadline if a GPU driver never replies.
        mp_thread_join(p->thread);
    }
    cleanup(p);
}

static const struct mp_filter_info filter = {
    .name = "ajn-sample", .priv_size = sizeof(struct priv),
    .process = process, .reset = reset, .destroy = destroy, .command = command,
};

static struct mp_filter *create(struct mp_filter *parent, void *options)
{
    struct mp_filter *f = mp_filter_create(parent, &filter);
    if (!f) { talloc_free(options); return NULL; }
    struct priv *p = f->priv;
    mp_mutex_init(&p->lock);
    p->opts = talloc_steal(p, options);
    p->detached = p->opts->name && p->opts->name[0];
    HANDLE mapping = (HANDLE)(uintptr_t)p->opts->mapping;
    if (p->detached) {
        talloc_steal(NULL, p);
        ajn_sample_lifetime_init(&p->lifetime);
        // Eight live/retiring branches bound retained driver work across rapid
        // reconfiguration. Failure leaves this observation filter a pass-through.
        if (InterlockedIncrement(&player_buffers) > 8) {
            InterlockedDecrement(&player_buffers);
            goto fail;
        }
        p->reserved = true;
        p->producer = InterlockedIncrement(&player_generation);
        if (p->producer <= 0) goto fail;
        const char *prefix = "Local\\AJN.PlayerFrames.";
        size_t prefix_len = strlen(prefix);
        if (p->opts->mapping || strlen(p->opts->name) != prefix_len + 32 ||
            strncmp(p->opts->name, prefix, prefix_len)) goto fail;
        for (size_t i = prefix_len; i < prefix_len + 32; i++)
            if (!strchr("0123456789abcdef", p->opts->name[i])) goto fail;
        wchar_t name[80];
        if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, p->opts->name, -1, name, 80)) goto fail;
        mapping = OpenFileMappingW(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, name);
        if (!mapping) goto fail;
    } else if (p->opts->mapping <= 0) goto fail;
    p->shared = MapViewOfFile(mapping, FILE_MAP_READ | FILE_MAP_WRITE,
        0, 0, sizeof(*p->shared));
    if (p->shared && p->shared->version == 2 && p->shared->analysis == 1 && p->shared->source_stage == 1) {
        UnmapViewOfFile(p->shared);
        p->shared = MapViewOfFile(mapping, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, sizeof(*p->shared) + AJN_SAMPLE_BYTES / 2);
        if (p->shared) p->mask = calloc(AJN_SAMPLE_WIDTH * AJN_SAMPLE_HEIGHT, sizeof(uint16_t));
        if (!p->mask && p->shared) { UnmapViewOfFile(p->shared); p->shared = NULL; }
    }
    if (p->detached) CloseHandle(mapping);
    if (!p->shared) goto fail;
    if (p->shared->magic != AJN_SAMPLE_MAGIC || p->shared->version != (p->mask ? 2 : AJN_SAMPLE_VERSION) ||
        p->shared->capacity != AJN_SAMPLE_BYTES + (p->mask ? AJN_SAMPLE_BYTES / 2 : 0) || p->shared->header_bytes != 256) {
        UnmapViewOfFile(p->shared);
        p->shared = NULL;
        goto fail;
    }
    if (p->detached) {
        // Readback may outlive the mpv context. Pin this DLL and its imported
        // dependencies until process exit so asynchronous cleanup has valid code.
        HMODULE module;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
            (LPCWSTR)(uintptr_t)readback, &module)) goto fail;
        InterlockedExchange64((volatile LONG64 *)&p->shared->producer, p->producer);
    }
    p->wake = CreateEventW(NULL, FALSE, FALSE, NULL);
    if (!p->wake) goto fail;
    if (p->detached) ajn_sample_lifetime_acquire(&p->lifetime);
    if (mp_thread_create(&p->thread, readback, p)) {
        if (p->detached) release(p);
        goto fail;
    }
    p->started = true;
    reset(f);
    state(p, AJN_SAMPLE_IDLE, S_OK);
    mp_filter_add_pin(f, MP_PIN_IN, "in");
    mp_filter_add_pin(f, MP_PIN_OUT, "out");
    return f;
fail:
    MP_ERR(f, "Invalid private AJN sample mapping.\n");
    if (p->detached) {
        // An unavailable observer must never make the user's video fail.
        mp_filter_add_pin(f, MP_PIN_IN, "in");
        mp_filter_add_pin(f, MP_PIN_OUT, "out");
        return f;
    }
    talloc_free(f);
    return NULL;
}

#define OPT_BASE_STRUCT struct opts
const struct mp_user_filter_entry vf_ajn_sample = {
    .desc = {
        .description = "private bounded AJN GPU sample branch",
        .name = "ajn-sample", .priv_size = sizeof(struct opts),
        .options = (const m_option_t[]) {{"mapping", OPT_INT64(mapping)}, {"name", OPT_STRING(name)}, {0}},
    },
    .create = create,
};
