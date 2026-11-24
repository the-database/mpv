/*
 * vf_animejanai: GPU-resident AI upscaling filter (CUDA / D3D11)
 *
 * The filter keeps frames in GPU memory end to end: IMGFMT_CUDA,
 * IMGFMT_D3D11 or IMGFMT_VULKAN in, own AVHWFramesContext output pool, same
 * format out. Vulkan frames (hwdec=vulkan) go to libaji's aji_create_vk on
 * the decoder's own device: plain upscale chains only for now (no RIFE,
 * temporal models or held-frame reuse on that path yet).
 * Inference runs in the libaji shim, loaded at runtime across a strict
 * C ABI (see aji.h) so no inference toolchain links into mpv; the shim
 * dispatches on the conf's backend key (TensorRT on CUDA frames,
 * DirectML on D3D11 frames).
 *
 * On the D3D11 path, decoder textures are neither shareable nor
 * implicitly synchronized (see vf_amf.c), so each input frame is staged
 * with a GPU copy into a filter-owned SHARED|NTHANDLE texture; the
 * output pool's textures get the same flags so the shim's D3D12 side
 * can open everything it touches.
 *
 * Modes:
 *  - conf=animejanai.conf (+ model-dir/trtexec/slot): full chain selection
 *    per slot and video properties, engines built on first play.
 *  - engine=file.engine: single fixed engine (spike/debug).
 *  - neither: synchronized GPU-side plane copy (passthrough).
 *
 * Upscale chains run pipelined (queue-depth frames in flight, default 3):
 * aji_infer only submits, completion is gated per frame through the
 * shim's ticket API (aji_flush/aji_wait), and the refqueue's future-ref
 * window provides the input read-ahead. RIFE chains stay synchronous
 * (their scene-change decision is a CPU readback per frame pair).
 *
 * Held-frame reuse (plain upscale chains): a source frame that is the same
 * picture as the last upscaled one (the anchor) skips inference, and the
 * anchor's output goes out again under the frame's pts. libaji's held-frame
 * engine (aji_held_*) decides, with the two frames after it as look-ahead;
 * see held_decide().
 *
 * Runtime: `vf-command animejanai slot <N>` switches the active slot.
 *
 * This file is part of mpv.
 *
 * mpv is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * mpv is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with mpv.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>

#include "config.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <ffnvcodec/dynlink_loader.h>

#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixfmt.h>

#if HAVE_D3D11
#include <libavutil/hwcontext_d3d11va.h>
#endif

#if HAVE_VULKAN
#include <libavutil/hwcontext_vulkan.h>
#endif

#include "common/common.h"
#include "filters/filter.h"
#include "filters/filter_internal.h"
#include "filters/user_filters.h"
#include "options/m_option.h"
#include "options/path.h"
#include "osdep/io.h"
#include "osdep/timer.h"
#include "refqueue.h"
#include "video/fmt-conversion.h"
#include "video/mp_image.h"
#include "video/mp_image_pool.h"

#include "aji.h"
#include "ajn_scene.h"

#ifdef _WIN32
#define AJI_DEFAULT_LIB "aji.dll"
#else
#define AJI_DEFAULT_LIB "libaji.so"
#endif

static void *aji_lib_open(const char *path)
{
#ifdef _WIN32
    // With an explicit path, resolve the shim's own dependencies (the
    // TensorRT runtime) from its directory, so the whole inference runtime
    // can live in one self-contained dir instead of next to mpv.exe or on
    // PATH. The search flags need a fully qualified, normalized path
    // (config expansion produces absolute-but-unnormalized ones).
    // The path is UTF-8, like every mpv string: load it wide. The ANSI calls
    // read it in the process code page, which breaks a non-ASCII install
    // path in a host without a UTF-8 manifest (mpv.net).
    wchar_t *wpath = mp_from_utf8(NULL, path);
    HMODULE lib;
    if (strpbrk(path, "/\\")) {
        DWORD need = GetFullPathNameW(wpath, 0, NULL, NULL);
        wchar_t *full = need ? talloc_array(wpath, wchar_t, need) : NULL;
        DWORD n = full ? GetFullPathNameW(wpath, need, full, NULL) : 0;
        lib = LoadLibraryExW(n > 0 && n < need ? full : wpath, NULL,
                             LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                             LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    } else {
        lib = LoadLibraryW(wpath);
    }
    DWORD err = GetLastError();  // for aji_lib_error()
    talloc_free(wpath);
    SetLastError(err);
    return (void *)lib;
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

static void *aji_lib_sym(void *handle, const char *name)
{
#ifdef _WIN32
    return (void *)GetProcAddress((HMODULE)handle, name);
#else
    return dlsym(handle, name);
#endif
}

static void aji_lib_close(void *handle)
{
#ifdef _WIN32
    FreeLibrary((HMODULE)handle);
#else
    dlclose(handle);
#endif
}

static const char *aji_lib_error(void)
{
#ifdef _WIN32
    static __thread char buf[32];
    snprintf(buf, sizeof(buf), "error %lu", (unsigned long)GetLastError());
    return buf;
#else
    return dlerror();
#endif
}

struct opts {
    bool passthrough;
    char *engine;
    char *conf;
    char *model_dir;
    char *trtexec;
    char *trtexec_libdir;
    char *stats;
    char *lib;
    char *rife_model_dir;
    int slot;
    bool skip_seek_pre_target;
    bool output_444;
    int queue_depth;
    int stream_mode; // 0 ordinary, 1 cached-only stream, 2 prepare, 3 check
    int reuse_held;  // 0 auto (animejanai.conf decides), 1 no
};

#define MAX_DEPTH 4

struct aji_api {
    void *handle;
    aji_ctx *(*create)(const aji_create_params *params);
    // optional: Vulkan frames on the caller's device (NULL on older shims)
    aji_ctx *(*create_vk)(const aji_create_params *params, const aji_vk_device *device);
    int (*set_slot)(aji_ctx *c, int slot);
    int (*configure)(aji_ctx *c, int w, int h, double fps, int *out_w,
                     int *out_h);
    int (*infer)(aji_ctx *c, const aji_frame *in, const aji_frame *out,
                 void *cu_stream);
    uint64_t (*flush)(aji_ctx *c, void *cu_stream);
    int (*done)(aji_ctx *c, uint64_t ticket);
    int (*wait)(aji_ctx *c, uint64_t ticket);
    int (*rife_factor)(aji_ctx *c, int *num, int *den);
    int (*rife_before_upscale)(aji_ctx *c);  // optional (may be NULL)
    int (*pre_resize)(aji_ctx *c, int *work_w, int *work_h);  // optional
    int (*resize)(aji_ctx *c, const aji_frame *in, const aji_frame *out,
                  void *cu_stream);          // optional; pre-RIFE downscale
    // optional: Vulkan frames' exact copy (held-frame anchors, RIFE alone)
    int (*copy_frame)(aji_ctx *c, const aji_frame *in, const aji_frame *out);
    int (*infer_rife)(aji_ctx *c, const aji_frame *a, const aji_frame *b,
                      double t, const aji_frame *out, void *cu_stream);
    int (*rife_scene_supported)(aji_ctx *c);
    int (*rife_enhanced_scene)(aji_ctx *c); // optional configure-time query
    int (*infer_rife_with_scene)(aji_ctx *c, const aji_frame *a, const aji_frame *b,
                      double t, const aji_frame *out, void *cu_stream, int scene);
    // optional: the frames' ids, so libaji keeps a frame's work for the next pair
    int (*infer_rife_ids)(aji_ctx *c, const aji_frame *a, int64_t a_id, const aji_frame *b,
                          int64_t b_id, double t, const aji_frame *out, void *cu_stream, int scene);
    int (*poll)(aji_ctx *c);
    int (*temporal_radius)(aji_ctx *c);
    int (*ingest)(aji_ctx *c, const aji_frame *in, uint64_t seq,
                  void *cu_stream);
    int (*infer_seq)(aji_ctx *c, uint64_t seq, const aji_frame *out,
                     void *cu_stream);
    void (*temporal_reset)(aji_ctx *c);
    const char *(*current_log)(aji_ctx *c);
    const char *(*last_error)(aji_ctx *c);
    void (*destroy)(aji_ctx **c);
    // Held-frame reuse, all optional: an older shim has none of them, and
    // then no frame is reused.
    int (*reuse_for_config)(const char *conf_path);
    aji_held *(*held_create)(aji_ctx *c, const aji_held_params *params);
    int (*held_set_trace)(aji_held *h, const char *path);
    int (*held_decide)(aji_held *h, const aji_held_frame *p,
                       const aji_held_frame *n, const aji_held_frame *n2,
                       void *cu_stream, aji_held_decision *out);
    int (*held_set_anchor)(aji_held *h, const aji_held_frame *copy);
    void (*held_reset)(aji_held *h);
    void (*held_get_stats)(aji_held *h, aji_held_stats *out);
    void (*held_destroy)(aji_held **h);
};

// D3D11 + held-frame reuse: staged copies of refqueue frames kept, at most
// the current frame plus the decision look-ahead (depth + 2).
#define MAX_STAGED (MAX_DEPTH + 3)

struct priv {
    struct opts *opts;

    // Expanded copies of the path options (~~/ etc.). Kept separate from
    // opts: the option machinery owns and frees those strings itself.
    struct {
        char *conf, *model_dir, *trtexec, *trtexec_libdir, *stats, *engine,
             *lib, *rife_model_dir;
    } path;

    CudaFunctions *cu;
    AVBufferRef *av_device_ref;
    CUcontext cuda_ctx;
    CUstream stream;
    CUstream own_stream;
    CUstream decode_stream;          // stream NVDEC decode/copy runs on (may be NULL/default)
    CUevent  decode_evt[MAX_DEPTH];  // per-frame decode->inference ordering events
    int      decode_evt_next;
    bool     decode_evt_ok;

    bool is_d3d11;
    bool is_vk;                      // Vulkan frames on the decoder's device (aji_create_vk)
#if HAVE_VULKAN
    aji_vk_device vkdev;             // its handles; av_device_ref keeps them alive
#endif
#if HAVE_D3D11
    ID3D11Device *d3d_dev;          // borrowed; av_device_ref keeps it alive
    ID3D11DeviceContext *d3d_ctx;
    // Shareable input staging textures, one per pipelined frame: the
    // D3D11 copy into a stage and the shim's D3D12 reads of it are not
    // ordered across APIs, so a stage must not be rewritten while a
    // queued frame still reads it. Slot reuse is safe because a slot
    // comes around again only after its previous frame's ticket was
    // waited at emission.
    ID3D11Texture2D *d3d_stage[MAX_DEPTH];
    int d3d_stage_count, d3d_stage_next;
    int d3d_stage_w, d3d_stage_h;
    DXGI_FORMAT d3d_stage_fmt;
#endif

    AVBufferRef *hw_pool;

    struct aji_api api;
    aji_ctx *aji;
    int aji_fmt;
    int out_fmt;         // output frames: aji_fmt or AJI_FMT_YUV444P16
    bool aji_active;     // a chain/engine is configured; else passthrough copy
    bool configured;     // saw at least one reinit
    int cur_slot, pending_slot;

    struct mp_refqueue *queue;
    struct mp_image_params params, out_params;
    double fps;
    // Software layout of params.hw_subfmt, for plane geometry only.
    struct mp_image layout;

    // Pipelined inference: frames submitted to the GPU but not yet
    // emitted, oldest first. ring[i].src identifies the refqueue frame at
    // relative position i (borrowed pointer, identity only - the
    // refqueue's future-ref window keeps it alive and thus the decoder
    // surface pinned until emission waits the ticket). depth is the
    // active queue depth: 1 (synchronous) for RIFE chains and bypass,
    // else the queue-depth option.
    struct {
        struct mp_image *src;
        struct mp_image *out;
        struct mp_image *stage;  // D3D11 + reuse: the staged input, kept
                                 // until the ticket was waited (or NULL)
        uint64_t ticket;
    } ring[MAX_DEPTH];
    int ring_n;
    int depth;

    // Held-frame reuse (see held_decide). Frames are identified by their
    // refqueue sequence number (seq0 + rel, as for temporal chains).
    aji_held *held;
    bool held_on;                  // reusing frames in the current chain
    const char *held_why;          // with a chain active and reuse off: why
    struct mp_image *anchor_out;   // the anchor's upscaled output
    aji_vk_image anchor_vk;        // Vulkan: the anchor's image, as the engine holds it
    struct mp_image *anchor_src;   // the anchor's source copy: CUDA a frame
                                   // of src_pool, D3D11 its staged copy
    uint64_t anchor_ticket;        // the anchor's inference ticket
    double held_last_pts;
    aji_held_stats held_base;      // counters at the last configure
    double stats_time;             // last write_stats, mp_time_sec()
    CUstream diff_stream;          // CUDA: the comparisons and anchor copies
    CUevent held_evt;              // CUDA: decode progress for diff_stream
    bool stats_warned;
#if HAVE_D3D11
    struct {
        uint64_t seq;
        struct mp_image *img;
    } staged[MAX_STAGED];
    int staged_n;
#endif

    // RIFE: outputs live on a uniform grid of num/den times the input
    // rate (vsmlrt video_player semantics: output j sits at input
    // position j*den/num; integer positions pass the source frame
    // through, fractional ones interpolate at t = frac). Per input pair
    // the due grid points are emitted: the first goes out through the
    // refqueue (consuming the input), the rest wait in outq and are
    // written on subsequent process() calls without consuming input.
    // With fractional factors some source frames are never emitted and
    // only serve as interpolation endpoints.
    bool rife_on;
    struct ajn_scene *scene;
    bool scene_builtin;
    int scene_decision;
    int rife_num, rife_den;     // reduced output multiplier num/den
    int rife_acc;               // next grid point's offset into the
                                // current pair, in 1/num units (0, num]
    struct mp_image *rife_prev; // last upscaled frame (left endpoint)
    // aji_infer_rife_ids: each new RIFE endpoint takes the next number and keeps it (rife_prev's in
    // rife_prev_id), so libaji can tell a pair's a is the last pair's b; numbers are never reused
    int64_t rife_next_id, rife_prev_id;
    struct mp_image *outq[8];
    int outq_n, outq_pos;
    // Temporal (multi-frame) chains: the engine keeps the frame history and
    // upscales frame N from N-r..N+r. Frames are numbered by refqueue
    // position: the frame at relative position k is seq0 + k, and seq0
    // advances with every frame the refqueue emits. seq_next is the next
    // number not yet handed to the engine (frames are ingested in order).
    int temporal_r;
    uint64_t seq0, seq_next;
    // RIFE-first: interpolate the source pair, then upscale every emitted
    // frame. rife_prev then holds the previous work-res source frame; src_pool
    // supplies source-resolution shareable copies of the decoder frames (the
    // D3D11 rife endpoints / pre-resize input).
    bool rife_first;
    AVBufferRef *src_pool;
    // Pre-RIFE downscale (rife-first): when the engine reports one via
    // aji_pre_resize, source frames are downscaled to (work_w, work_h) with
    // api.resize before RIFE + upscaling, so RIFE runs on the smaller frame.
    // The interp temps and upscale inputs are then at the work resolution
    // (work_pool). When there is no pre-resize, work_w/h == params.w/h.
    bool has_pre_resize;
    int work_w, work_h;
    AVBufferRef *work_pool;          // work-res frames (interp temps + the
                                     // pre-RIFE downscale output); only used
                                     // when has_pre_resize
};

static int check_cu(struct mp_filter *vf, CUresult err, const char *func)
{
    struct priv *p = vf->priv;
    if (err == CUDA_SUCCESS)
        return 0;
    const char *err_name = NULL;
    const char *err_string = NULL;
    p->cu->cuGetErrorName(err, &err_name);
    p->cu->cuGetErrorString(err, &err_string);
    MP_ERR(vf, "%s failed: %s (%s)\n", func, err_name ? err_name : "unknown",
           err_string ? err_string : "unknown");
    return -1;
}

#define CHECK_CU(x) check_cu(vf, (x), #x)

// Inference reads the decoder surface on p->stream; decode/copy ran on
// p->decode_stream. When they differ (we made a private non-blocking
// stream), nothing orders them, so a HAGS-scheduled read can race ahead
// of decode and pick up a recycled pool slot's previous frame. Insert an
// explicit dependency: record decode's progress, make inference wait.
static bool order_after_decode(struct mp_filter *vf)
{
    struct priv *p = vf->priv;
    if (p->is_d3d11 || p->is_vk || p->stream == p->decode_stream || !p->decode_evt_ok)
        return true;                       // already ordered / nothing to do
    CUcontext dummy;
    if (CHECK_CU(p->cu->cuCtxPushCurrent(p->cuda_ctx)) < 0)
        return false;
    CUevent ev = p->decode_evt[p->decode_evt_next];
    p->decode_evt_next = (p->decode_evt_next + 1) % MAX_DEPTH;
    bool ok = CHECK_CU(p->cu->cuEventRecord(ev, p->decode_stream)) >= 0 &&
              CHECK_CU(p->cu->cuStreamWaitEvent(p->stream, ev, 0)) >= 0;
    p->cu->cuCtxPopCurrent(&dummy);
    return ok;
}

static void aji_log_bridge(void *opaque, int level, const char *msg)
{
    struct mp_filter *vf = opaque;
    int mp_level = level <= 1 ? MSGL_ERR : level == 2 ? MSGL_WARN : MSGL_V;
    mp_msg(vf->log, mp_level, "[aji] %s\n", msg);
}

static int map_matrix(struct mp_image_params *params)
{
    switch (params->repr.sys) {
    case PL_COLOR_SYSTEM_BT_601:     return AJI_MATRIX_BT601;
    case PL_COLOR_SYSTEM_BT_2020_NC: return AJI_MATRIX_BT2020;
    case PL_COLOR_SYSTEM_BT_709:     return AJI_MATRIX_BT709;
    default:
        // untagged: same SD heuristic as the legacy pipeline
        return params->h < 720 ? AJI_MATRIX_BT601 : AJI_MATRIX_BT709;
    }
}

static int map_range(struct mp_image_params *params)
{
    return params->repr.levels == PL_COLOR_LEVELS_FULL ? AJI_RANGE_FULL
                                                       : AJI_RANGE_LIMITED;
}

static int map_siting(struct mp_image_params *params)
{
    switch (params->chroma_location) {
    case PL_CHROMA_CENTER:   return AJI_SITING_CENTER;
    case PL_CHROMA_TOP_LEFT: return AJI_SITING_TOPLEFT;
    case PL_CHROMA_LEFT:
    default:                 return AJI_SITING_LEFT;
    }
}

static struct mp_image *alloc_src(struct mp_filter *vf);
static void held_destroy(struct mp_filter *vf);
static void held_destroy_cuda(struct mp_filter *vf);
#if HAVE_D3D11
static struct mp_image *stage_src_d3d11(struct mp_filter *vf,
                                        struct mp_image *src);
#endif

/*
 * Held-frame reuse, plain upscale chains only (RIFE and temporal chains keep
 * it off, like aji_encode's defaults). Every refqueue frame is decided once,
 * in order, as submit_frame takes it: libaji's engine compares it with the
 * anchor and with the two frames after it (rel + 1 and rel + 2, so the
 * refqueue buffers depth + 1 future frames; at the end of the stream they
 * are missing and the frame is upscaled). REUSE puts a new ref of the
 * anchor's output into the ring under the frame's pts, with the anchor's
 * ticket: tickets complete in order and nothing is upscaled between the
 * anchor and a frame that reuses it (every upscale drops the anchor), so the
 * ring's wait and drain rules stay as they are. Frame count and pts are
 * those without reuse; damaged frames are never reused nor anchors.
 *
 * CUDA: the comparisons run on diff_stream, which waits on an event recorded
 * on the decode stream when the frame is decided; the frames after it were
 * decoded before. The anchor's source is a copy in src_pool, so a long still
 * doesn't pin a decoder surface.
 *
 * D3D11: decoder surfaces aren't shareable, so each frame gets a staged
 * copy in src_pool (pooled, refcounted, written once) the first time it is
 * compared; the same copy is the inference input and, if the frame is
 * upscaled, the anchor's source. The d3d_stage ring is not used then.
 * Marks are 0: a comparison waits for the work queued on the immediate
 * context, never for inference.
 */

static aji_held_stats held_stats(struct priv *p)
{
    aji_held_stats s = {.struct_size = sizeof(s)};
    if (p->held)
        p->api.held_get_stats(p->held, &s);
    return s;
}

// Forget the anchor, the look-ahead and the staged copies: a seek, a
// discontinuity, a reconfigure or a pipeline resync. Counters stay.
static void held_reset(struct mp_filter *vf)
{
    struct priv *p = vf->priv;
    if (p->held)
        p->api.held_reset(p->held);
    mp_image_unrefp(&p->anchor_out);
    mp_image_unrefp(&p->anchor_src);
    p->anchor_ticket = 0;
    p->held_last_pts = MP_NOPTS_VALUE;
#if HAVE_D3D11
    for (int i = 0; i < p->staged_n; i++)
        mp_image_unrefp(&p->staged[i].img);
    p->staged_n = 0;
#endif
}

// Reuse stops for the rest of this chain (until the next configure).
static void held_fail(struct mp_filter *vf, const char *what)
{
    struct priv *p = vf->priv;
    MP_WARN(vf, "Held-frame reuse failed (%s); reusing no more frames\n", what);
    held_reset(vf);
    p->held_on = false;
    p->held_why = "failed during playback";
}

#if HAVE_VULKAN
// An AVVkFrame handed to libaji as an aji_vk_image (one multi-planar image),
// locked from vk_take to vk_give; vk_give writes back the semaphore value,
// layout, access and queue family libaji leaves (AVVkFrame's rule) when ok.
struct vk_held {
    AVVkFrame *f;
    AVHWFramesContext *fc;
    aji_vk_image d;
};

static bool vk_take(struct mp_filter *vf, struct mp_image *img, struct vk_held *h)
{
    h->f = (AVVkFrame *)img->planes[0];
    h->fc = img->hwctx ? (void *)img->hwctx->data : NULL;
    if (!h->f || !h->fc) {
        MP_ERR(vf, "Vulkan frame without its frames context\n");
        return false;
    }
    if (h->f->img[1]) {
        MP_ERR(vf, "Vulkan frames with one image per plane are not supported\n");
        return false;
    }
    AVVulkanFramesContext *hc = h->fc->hwctx;
    if (hc->lock_frame)
        hc->lock_frame(h->fc, h->f);
    h->d = (aji_vk_image){
        .image = h->f->img[0], .layout = h->f->layout[0], .queue_family = h->f->queue_family[0],
        .access = h->f->access[0], .semaphore = h->f->sem[0], .value = h->f->sem_value[0],
    };
    return true;
}

static void vk_give(struct vk_held *h, bool ok)
{
    if (ok) {
        h->f->layout[0] = h->d.layout;
        h->f->access[0] = h->d.access;
        h->f->queue_family[0] = h->d.queue_family;
        h->f->sem_value[0] = h->d.value;
    }
    AVVulkanFramesContext *hc = h->fc->hwctx;
    if (hc->unlock_frame)
        hc->unlock_frame(h->fc, h->f);
}

#endif

static void held_desc(struct priv *p, aji_held_frame *d, struct mp_image *img,
                      uint64_t id, bool damaged)
{
    *d = (aji_held_frame){
        .frame = {
            .width = p->params.w, .height = p->params.h,
            .format = p->aji_fmt,
            .matrix = map_matrix(&p->params),
            .range = map_range(&p->params),
            .siting = map_siting(&p->params),
            .plane = {img->planes[0], img->planes[1]},
        },
        .id = id,
        .flags = damaged ? AJI_HELD_FRAME_DAMAGED : 0,
    };
    if (!p->is_d3d11) {
        d->frame.plane[2] = img->planes[2];
        for (int i = 0; i < 3; i++)
            d->frame.stride[i] = img->stride[i];
    }
}

#if HAVE_D3D11
// The staged copy of the refqueue frame at rel, made on first use and kept
// while the frame can still be compared (its sequence number >= seq0).
static struct mp_image *staged_get(struct mp_filter *vf, int rel)
{
    struct priv *p = vf->priv;
    const uint64_t seq = p->seq0 + rel;
    struct mp_image *found = NULL;
    int n = 0;
    for (int i = 0; i < p->staged_n; i++) {
        if (p->staged[i].seq < p->seq0) {
            mp_image_unrefp(&p->staged[i].img);
            continue;
        }
        if (p->staged[i].seq == seq)
            found = p->staged[i].img;
        p->staged[n++] = p->staged[i];
    }
    p->staged_n = n;
    if (found)
        return found;
    struct mp_image *src = mp_refqueue_get(p->queue, rel);
    if (!src || p->staged_n == MAX_STAGED)
        return NULL;
    struct mp_image *img = stage_src_d3d11(vf, src);
    if (!img)
        return NULL;
    p->staged[p->staged_n].seq = seq;
    p->staged[p->staged_n].img = img;
    p->staged_n++;
    return img;
}
#endif

// Decide the refqueue frame at rel (in). true: reuse the anchor's output.
// *stage (D3D11) gets a ref of in's staged copy for the inference input; it
// stays NULL on CUDA, or if staging failed (then in is staged as without
// reuse). A failure turns reuse off for the chain and decides UPSCALE.
static bool held_decide(struct mp_filter *vf, struct mp_image *in, int rel,
                        struct mp_image **stage)
{
    struct priv *p = vf->priv;
    *stage = NULL;

    // pts going back (or not moving) without a reset: a discontinuity
    if (in->pts != MP_NOPTS_VALUE && p->held_last_pts != MP_NOPTS_VALUE &&
        in->pts <= p->held_last_pts)
        held_reset(vf);
    p->held_last_pts = in->pts;

    struct mp_image *f[3] = {
        in,
        mp_refqueue_get(p->queue, rel + 1),
        mp_refqueue_get(p->queue, rel + 2),
    };
    const int nf = f[1] && f[2] ? 3 : 1;  // no look-ahead: forced upscale
    aji_held_frame d[3];
    void *stream = NULL;
    aji_held_decision dec = {0};
    int dret = AJI_ERR;
    bool decided = false;
#if HAVE_VULKAN
    if (p->is_vk) {
        // the frames (and the anchor's copy, which the engine compares them with) locked
        // for the call, their image state as libaji leaves it written back after
        struct vk_held h[3], ha;
        int n = 0;
        while (n < nf && vk_take(vf, f[n], &h[n]))
            n++;
        const bool have_anchor = n == nf && p->anchor_src && vk_take(vf, p->anchor_src, &ha);
        if (n == nf && (have_anchor || !p->anchor_src)) {
            if (have_anchor)
                p->anchor_vk = ha.d;
            for (int i = 0; i < nf; i++) {
                held_desc(p, &d[i], f[i], p->seq0 + rel + i, f[i]->corrupt);
                d[i].frame.plane[0] = &h[i].d;
                d[i].frame.plane[1] = d[i].frame.plane[2] = NULL;
                for (int k = 0; k < 3; k++)
                    d[i].frame.stride[k] = 0;
            }
            dret = p->api.held_decide(p->held, &d[0], nf == 3 ? &d[1] : NULL,
                                      nf == 3 ? &d[2] : NULL, NULL, &dec);
            decided = true;
        }
        if (have_anchor) {
            ha.d = p->anchor_vk;
            vk_give(&ha, true);
        }
        while (n-- > 0)
            vk_give(&h[n], true);
        if (!decided) {
            held_fail(vf, "locking a frame");
            return false;
        }
    } else
#endif
#if HAVE_D3D11
    if (p->is_d3d11) {
        struct mp_image *s[3] = {0};
        for (int i = 0; i < nf; i++) {
            if (!(s[i] = staged_get(vf, rel + i))) {
                held_fail(vf, "staging a frame");
                return false;
            }
        }
        for (int i = 0; i < nf; i++)
            held_desc(p, &d[i], s[i], p->seq0 + rel + i, f[i]->corrupt);
        *stage = mp_image_new_ref(s[0]);
        MP_HANDLE_OOM(*stage);
    } else
#endif
    {
        CUcontext dummy;
        if (CHECK_CU(p->cu->cuCtxPushCurrent(p->cuda_ctx)) < 0) {
            held_fail(vf, "CUDA context");
            return false;
        }
        bool ok = CHECK_CU(p->cu->cuEventRecord(p->held_evt,
                                                p->decode_stream)) >= 0 &&
                  CHECK_CU(p->cu->cuStreamWaitEvent(p->diff_stream,
                                                    p->held_evt, 0)) >= 0;
        p->cu->cuCtxPopCurrent(&dummy);
        if (!ok) {
            held_fail(vf, "CUDA event");
            return false;
        }
        for (int i = 0; i < nf; i++)
            held_desc(p, &d[i], f[i], p->seq0 + rel + i, f[i]->corrupt);
        stream = p->diff_stream;
    }

    if (!decided)
        dret = p->api.held_decide(p->held, &d[0], nf == 3 ? &d[1] : NULL,
                                  nf == 3 ? &d[2] : NULL, stream, &dec);
    if (dret != AJI_OK || dec.error)
    {
        MP_WARN(vf, "Held-frame comparison failed: %s\n",
                p->api.last_error(p->aji));
        held_fail(vf, "comparison");
        return false;
    }
    if (dec.action != AJI_HELD_REUSE)
        return false;
    if (!p->anchor_out) {  // can't happen: the engine reuses only an anchor
        held_reset(vf);
        return false;
    }
    MP_TRACE(vf, "held frame pts=%f reuses frame %" PRIu64 " (run %d)\n",
             in->pts, dec.anchor_id, dec.run);
    return true;
}

// in (refqueue position rel) was just upscaled into out: it becomes the
// anchor. stage is its D3D11 staged copy (NULL on CUDA).
static void held_set_anchor(struct mp_filter *vf, struct mp_image *in,
                            int rel, struct mp_image *stage,
                            struct mp_image *out, uint64_t ticket)
{
    struct priv *p = vf->priv;
    // the engine dropped the last anchor when it decided to upscale
    mp_image_unrefp(&p->anchor_out);
    if (in->corrupt)
        return;  // never an anchor
    if (p->is_vk) {
#if HAVE_VULKAN
        // an exact copy in src_pool (the decoder's frame goes back to its pool)
        if (!p->anchor_src && !(p->anchor_src = alloc_src(vf))) {
            held_fail(vf, "anchor copy");
            return;
        }
        struct vk_held hi, ha;
        bool ok = false;
        if (vk_take(vf, in, &hi)) {
            if (vk_take(vf, p->anchor_src, &ha)) {
                const aji_frame fi = {
                    .width = p->params.w, .height = p->params.h, .format = p->aji_fmt,
                    .plane = {&hi.d},
                };
                aji_frame fa = fi;
                fa.plane[0] = &ha.d;
                ok = p->api.copy_frame(p->aji, &fi, &fa) == AJI_OK;
                p->anchor_vk = ha.d;
                vk_give(&ha, true);
            }
            vk_give(&hi, true);
        }
        if (!ok) {
            held_fail(vf, "anchor copy");
            return;
        }
#endif
    } else if (p->is_d3d11) {
        mp_image_unrefp(&p->anchor_src);
        p->anchor_src = mp_image_new_ref(stage);
        MP_HANDLE_OOM(p->anchor_src);
    } else {
        // The engine dropped the previous anchor when it decided to upscale,
        // and its comparisons are done, so the copy can be overwritten.
        if (!p->anchor_src && !(p->anchor_src = alloc_src(vf))) {
            held_fail(vf, "anchor copy");
            return;
        }
        CUcontext dummy;
        if (CHECK_CU(p->cu->cuCtxPushCurrent(p->cuda_ctx)) < 0) {
            held_fail(vf, "anchor copy");
            return;
        }
        // diff_stream already waits for in's decode (held_decide)
        bool ok = true;
        for (int n = 0; ok && n < p->layout.fmt.num_planes; n++) {
            CUDA_MEMCPY2D cpy = {
                .srcMemoryType = CU_MEMORYTYPE_DEVICE,
                .srcDevice     = (CUdeviceptr)(uintptr_t)in->planes[n],
                .srcPitch      = in->stride[n],
                .dstMemoryType = CU_MEMORYTYPE_DEVICE,
                .dstDevice     = (CUdeviceptr)(uintptr_t)p->anchor_src->planes[n],
                .dstPitch      = p->anchor_src->stride[n],
                .WidthInBytes  = mp_image_plane_w(&p->layout, n) *
                                 p->layout.fmt.bpp[n] / 8,
                .Height        = mp_image_plane_h(&p->layout, n),
            };
            ok = CHECK_CU(p->cu->cuMemcpy2DAsync(&cpy, p->diff_stream)) >= 0;
        }
        ok = ok && CHECK_CU(p->cu->cuStreamSynchronize(p->diff_stream)) >= 0;
        p->cu->cuCtxPopCurrent(&dummy);
        if (!ok) {
            held_fail(vf, "anchor copy");
            return;
        }
    }
    aji_held_frame a;
    held_desc(p, &a, p->anchor_src, p->seq0 + rel, false);
    if (p->is_vk) {
        // the engine keeps this frame: plane[0] is the struct held_decide refreshes
        a.frame.plane[0] = &p->anchor_vk;
        a.frame.plane[1] = a.frame.plane[2] = NULL;
        for (int k = 0; k < 3; k++)
            a.frame.stride[k] = 0;
    }
    if (p->api.held_set_anchor(p->held, &a) != AJI_OK) {
        held_fail(vf, "setting the anchor");
        return;
    }
    p->anchor_out = mp_image_new_ref(out);
    MP_HANDLE_OOM(p->anchor_out);
    p->anchor_ticket = ticket;
}

// Wait until every submitted frame's GPU work completed. The ring entries
// stay queued for emission; tickets complete in order, so waiting the
// newest covers all.
static bool drain_ring(struct mp_filter *vf)
{
    struct priv *p = vf->priv;
    if (!p->ring_n)
        return true;
    if (p->api.wait(p->aji, p->ring[p->ring_n - 1].ticket) != AJI_OK) {
        MP_ERR(vf, "pipeline drain failed: %s\n", p->api.last_error(p->aji));
        return false;
    }
    return true;
}

static void clear_ring(struct mp_filter *vf)
{
    struct priv *p = vf->priv;
    drain_ring(vf);
    for (int i = 0; i < p->ring_n; i++) {
        mp_image_unrefp(&p->ring[i].out);
        mp_image_unrefp(&p->ring[i].stage);
    }
    p->ring_n = 0;
    // the dropped frames come back undecided (lost sync) or not at all
    held_reset(vf);
}

static void drain_backend(struct mp_filter *vf)
{
    struct priv *p = vf->priv;
    if (!p->aji)
        return;

    uint64_t ticket = p->api.flush(p->aji, p->stream);
    if (ticket && p->api.wait(p->aji, ticket) != AJI_OK) {
        MP_WARN(vf, "backend drain failed during reset: %s\n",
                p->api.last_error(p->aji));
    }
}

static void flush_frames(struct mp_filter *vf)
{
    struct priv *p = vf->priv;
    ajn_scene_reset(p->scene);
    p->scene_decision = -1;
    // GPU reads of the queued input frames must finish before the
    // refqueue drops them
    clear_ring(vf);
    // RIFE-first also uses immediate resize/interp work that is not tracked
    // in the upscaler ring. Drain the backend before releasing frame pools on
    // seek/reset so old work cannot contend with the post-seek chain.
    drain_backend(vf);
    mp_refqueue_flush(p->queue);
    if (p->aji && p->temporal_r)
        p->api.temporal_reset(p->aji);
    p->seq0 = p->seq_next = 0;
    mp_image_unrefp(&p->rife_prev);
    for (int i = 0; i < p->outq_n; i++)
        mp_image_unrefp(&p->outq[p->outq_pos + i]);
    p->outq_n = p->outq_pos = 0;
    // the output grid restarts at the next frame (= a source frame)
    p->rife_acc = p->rife_den;

    if (p->rife_first) {
        av_buffer_unref(&p->src_pool);
        av_buffer_unref(&p->work_pool);
    }
#if HAVE_D3D11
    p->d3d_stage_next = 0;
#endif
    p->decode_evt_next = 0;
}

// Tear down everything whose lifetime is tied to the currently adopted hardware
// device/frames context, leaving the filter in a clean state that the reinit
// path will rebuild against the new device. A context switch (e.g. hwupload_cuda
// combined with a seek, or a decode-mode fallback) invalidates the old CUcontext,
// stream, inference context and pools; reusing them faults the engine with
// CUDA_ERROR_ILLEGAL_ADDRESS. Library/process-level state (the aji handle, the
// refqueue, the CUDA function table) is deliberately kept -- only uninit() frees
// those.
static void release_device_state(struct mp_filter *vf)
{
    struct priv *p = vf->priv;

    // Finish submitted inference before releasing referenced surfaces or objects
    // created in the old CUDA/D3D device context. clear_ring() waits on the
    // upscaler ring; drain_backend() also flushes RIFE-first immediate work that
    // runs on the decode stream and is not tracked in the ring.
    clear_ring(vf);
    drain_backend(vf);

    mp_image_unrefp(&p->rife_prev);
    for (int i = 0; i < p->outq_n; i++)
        mp_image_unrefp(&p->outq[p->outq_pos + i]);
    p->outq_n = p->outq_pos = 0;
    p->rife_num = p->rife_den = 1;
    p->rife_acc = 1;

    held_destroy(vf);
    if (p->aji)
        p->api.destroy(&p->aji);

#if HAVE_D3D11
    for (int i = 0; i < p->d3d_stage_count; i++) {
        if (p->d3d_stage[i])
            ID3D11Texture2D_Release(p->d3d_stage[i]);
        p->d3d_stage[i] = NULL;
    }
    p->d3d_stage_count = 0;
    p->d3d_stage_next = 0;
    p->d3d_stage_w = p->d3d_stage_h = 0;
    p->d3d_stage_fmt = DXGI_FORMAT_UNKNOWN;
    p->d3d_dev = NULL;
    p->d3d_ctx = NULL;
#endif

    if (p->cu && p->cuda_ctx) {
        CUcontext dummy;
        CUresult err = p->cu->cuCtxPushCurrent(p->cuda_ctx);
        if (err == CUDA_SUCCESS) {
            if (p->own_stream)
                p->cu->cuStreamSynchronize(p->own_stream);
            for (int i = 0; i < MAX_DEPTH; i++) {
                if (p->decode_evt[i])
                    p->cu->cuEventDestroy(p->decode_evt[i]);
            }
            if (p->own_stream)
                p->cu->cuStreamDestroy(p->own_stream);
            held_destroy_cuda(vf);
            p->cu->cuCtxPopCurrent(&dummy);
        } else if (p->own_stream || p->decode_evt_ok || p->diff_stream) {
            MP_WARN(vf, "Could not make the old CUDA context current while "
                        "releasing AnimeJaNai device state\n");
        }
    }

    for (int i = 0; i < MAX_DEPTH; i++)
        p->decode_evt[i] = NULL;
    p->diff_stream = NULL;
    p->held_evt = NULL;

    av_buffer_unref(&p->hw_pool);
    av_buffer_unref(&p->src_pool);
    av_buffer_unref(&p->work_pool);
    av_buffer_unref(&p->av_device_ref);

    p->cuda_ctx = NULL;
    p->stream = NULL;
    p->own_stream = NULL;
    p->decode_stream = NULL;
    p->decode_evt_next = 0;
    p->decode_evt_ok = false;
    p->is_d3d11 = false;
    p->is_vk = false;
    p->aji_active = false;
    p->configured = false;
    p->rife_on = false;
    p->rife_first = false;
    p->temporal_r = 0;
    p->seq0 = p->seq_next = 0;
}

static const char held_why_no_engine[] = "the inference library is too old";

static bool write_text(const char *path, const char *text)
{
    FILE *f = fopen(path, "w");
    if (!f)
        return false;
    bool ok = fputs(text, f) >= 0;
    return fclose(f) == 0 && ok;
}

// The chain's log, plus a held-frame line while an upscale chain is active
// (the stats overlay parses it; same wording as aji_encode):
// "held frames: reused 123 of 1000 (12.3%)", counted since the chain was
// configured, or "held frames: not reused (<why>)". No line when the
// inference library has no held-frame engine. Rewritten about once a second
// while frames are reused, through a temporary file, so a reader never sees
// it half written.
static void write_stats(struct mp_filter *vf)
{
    struct priv *p = vf->priv;
    if (!p->path.stats || !p->aji)
        return;
    p->stats_time = mp_time_sec();
    const char *log = p->api.current_log(p->aji);
    char *text = talloc_strdup(NULL, log ? log : "");
    if (p->held_on ||
        (p->aji_active && p->held_why && p->held_why != held_why_no_engine)) {
        size_t len = strlen(text);
        if (len && text[len - 1] != '\n')
            text = talloc_strdup_append(text, "\n");
        if (p->held_on) {
            aji_held_stats s = held_stats(p);
            long long seen = s.seen - p->held_base.seen;
            long long reused = s.reused - p->held_base.reused;
            text = talloc_asprintf_append(text,
                "held frames: reused %lld of %lld (%.1f%%)\n", reused, seen,
                seen > 0 ? 100.0 * reused / seen : 0.0);
        } else {
            text = talloc_asprintf_append(text,
                "held frames: not reused (%s)\n", p->held_why);
        }
    }
    char *tmp = talloc_asprintf(text, "%s.tmp", p->path.stats);
    bool ok = write_text(tmp, text) && rename(tmp, p->path.stats) == 0;
    if (!ok) {
        // e.g. the overlay has the file open right now: write it in place
        remove(tmp);
        ok = write_text(p->path.stats, text);
    }
    if (!ok && !p->stats_warned) {
        MP_WARN(vf, "Cannot write stats file %s\n", p->path.stats);
        p->stats_warned = true;
    }
    talloc_free(text);
}

static void held_log_stats(struct mp_filter *vf)
{
    struct priv *p = vf->priv;
    aji_held_stats s = held_stats(p);
    if (!s.seen)
        return;
    MP_VERBOSE(vf, "Held frames: %lld of %lld reused (%.1f%%), %lld "
               "comparisons, %.3f ms each\n", (long long)s.reused,
               (long long)s.seen, 100.0 * s.reused / s.seen,
               (long long)s.comparisons,
               s.comparisons ? s.compare_ns / 1e6 / s.comparisons : 0.0);
}

// Destroy the engine (before aji_destroy). The CUDA stream and event go in
// held_destroy_cuda, with the context current.
static void held_destroy(struct mp_filter *vf)
{
    struct priv *p = vf->priv;
    if (p->held_on)
        write_stats(vf);  // the final count
    held_reset(vf);
    if (p->held) {
        held_log_stats(vf);
        p->api.held_destroy(&p->held);
    }
    p->held = NULL;
    p->held_on = false;
    p->held_why = NULL;
}

static void held_destroy_cuda(struct mp_filter *vf)
{
    struct priv *p = vf->priv;
    if (p->diff_stream) {
        p->cu->cuStreamSynchronize(p->diff_stream);
        p->cu->cuStreamDestroy(p->diff_stream);
    }
    if (p->held_evt)
        p->cu->cuEventDestroy(p->held_evt);
    p->diff_stream = NULL;
    p->held_evt = NULL;
}

// Whether this chain reuses held frames; one line says so, or why not.
// Called by configure_aji with an upscale chain active.
static void held_setup(struct mp_filter *vf)
{
    struct priv *p = vf->priv;
    const char *why = NULL;
    if (p->opts->reuse_held == 1)
        why = "reuse-held=no";
    else if (!p->api.held_create || !p->api.reuse_for_config)
        why = held_why_no_engine;
    else if (!p->api.reuse_for_config(p->path.conf))
        why = "reuse_held_frames is off in animejanai.conf";
    else if (p->rife_on)
        why = "RIFE chain";
    else if (p->temporal_r)
        why = "temporal chain";
    else if (p->is_vk && !p->api.copy_frame)
        why = "Vulkan frames need a libaji with aji_copy_frame";
    if (!why && !p->held) {
        const aji_held_params hp = {
            .struct_size = sizeof(hp),
            .threshold = AJI_HELD_THRESHOLD_DEFAULT,
            .fade_threshold = AJI_HELD_FADE_DEFAULT,
            .max_run = AJI_HELD_MAX_RUN_DEFAULT,
        };
        p->held = p->api.held_create(p->aji, &hp);
        if (!p->held) {
            why = "the backend can't compare frames";
        } else {
            const char *trace = getenv("AJI_REUSE_TRACE");
            if (trace && *trace && p->api.held_set_trace &&
                p->api.held_set_trace(p->held, trace) != AJI_OK)
                MP_WARN(vf, "Cannot write held-frame trace %s\n", trace);
        }
    }
    if (!why && !p->is_d3d11 && !p->is_vk && !p->diff_stream) {
        CUcontext dummy;
        bool ok = CHECK_CU(p->cu->cuCtxPushCurrent(p->cuda_ctx)) >= 0;
        if (ok) {
            ok = CHECK_CU(p->cu->cuStreamCreate(&p->diff_stream,
                                                CU_STREAM_NON_BLOCKING)) >= 0 &&
                 CHECK_CU(p->cu->cuEventCreate(&p->held_evt,
                                               CU_EVENT_DISABLE_TIMING)) >= 0;
            if (!ok)
                held_destroy_cuda(vf);
            p->cu->cuCtxPopCurrent(&dummy);
        }
        if (!ok)
            why = "CUDA stream setup failed";
    }
    p->held_why = why;
    if (why) {
        MP_INFO(vf, "Held-frame reuse: off (%s)\n", why);
        return;
    }
    p->held_on = true;
    p->held_base = held_stats(p);
    MP_INFO(vf, "Held-frame reuse: on (unchanged frames reuse the last "
            "upscale)\n");
}

// Pipeline depth for the current configuration. RIFE chains stay
// synchronous (their scene-change decision is a CPU readback per pair),
// as do bypass and passthrough; active upscale chains run queue-depth
// frames deep, with the refqueue's future-ref window supplying the
// read-ahead (depth - 1 buffered future frames).
//
// Temporal chains also run synchronously: the refqueue already holds r future
// frames for the model's window, and read-ahead on top of that would pin more
// decoder surfaces than the hwdec pool's spare frames.
static void update_depth(struct mp_filter *vf)
{
    struct priv *p = vf->priv;
    int depth = MPCLAMP(p->opts->queue_depth, 1, MAX_DEPTH);
    p->depth = (p->aji_active && !p->rife_on && !p->temporal_r) ? depth : 1;
    // Held-frame reuse decides the newest submitted frame (rel depth - 1)
    // with the two after it.
    int future = p->temporal_r ? p->temporal_r : p->depth - 1;
    if (p->held_on)
        future = p->depth + 1;
    mp_refqueue_set_refs(p->queue, 0,
                         MPMAX(future, p->scene && p->rife_on ? 2 : 0));
}

// Versioned, path-free signal for the trusted libmpv host. Never forward raw
// inference logs to addons. Preparation/check modes produce no video frames.
#define AJN_STREAM_VERSION 1
static void stream_status(struct mp_filter *vf, const char *state)
{
    struct priv *p = vf->priv;
    if (!p->opts->stream_mode)
        return;
    MP_INFO(vf, "AJN_STREAM_V1 %s %s %d %d %d %d %d %d %.6f\n", state,
            p->is_vk ? "Vulkan" : p->is_d3d11 ? "DirectML" : "TensorRT", p->cur_slot,
            p->params.w, p->params.h, p->out_params.w, p->out_params.h,
            p->rife_on ? 1 : 0, p->fps * (p->rife_on ? (double)p->rife_num / p->rife_den : 1));
}

static void configure_scene(struct mp_filter *vf, bool rife_active)
{
    struct priv *p = vf->priv;
    // External scene-map users retain ownership while attached. With the
    // experimental setting off, scene stays NULL and the existing frame path
    // (including queue depth) is unchanged. Query the setting only at setup.
    if (p->scene && !p->scene_builtin) return;
    bool enabled = p->aji && rife_active && p->api.rife_enhanced_scene &&
                   p->api.rife_enhanced_scene(p->aji);
    if (enabled && p->is_vk) {
        // its detector reads CUDA / D3D11 frames; libaji's own detection runs
        MP_WARN(vf, "Experimental RIFE scene detection is not on Vulkan frames; using the built-in one\n");
        enabled = false;
    }
    if (!enabled) {
        if (p->scene_builtin) {
            ajn_scene_destroy(p->scene); p->scene = NULL;
            p->scene_builtin = false;
        }
        return;
    }
    // A reconfigure invalidates pending pairs, but an enabled chain can reuse
    // its worker and native context instead of retiring and reloading them.
    if (p->scene) return;
    if (!p->api.infer_rife_with_scene || !p->api.rife_scene_supported ||
        !p->api.rife_scene_supported(p->aji)) {
        MP_WARN(vf, "Experimental RIFE scene detection requires scene override support\n");
        return;
    }
    p->scene = ajn_scene_create(vf->log);
    if (!ajn_scene_start_enhanced(p->scene)) {
        ajn_scene_destroy(p->scene); p->scene = NULL;
        MP_WARN(vf, "Experimental RIFE scene detection could not start\n");
        return;
    }
    p->scene_builtin = true;
    MP_INFO(vf, "Experimental RIFE scene detection enabled\n");
}

// (Re)configure the shim for the current stream params and slot. Updates
// out_params (and thus the output pool geometry). May block on first-play
// engine builds, like the legacy pipeline.
static bool configure_aji(struct mp_filter *vf)
{
    struct priv *p = vf->priv;

    if (p->scene_builtin) ajn_scene_reset(p->scene);
    p->scene_decision = -1;
    p->out_params = p->params;
    p->aji_active = false;
    // A new chain (or none): forget the anchor; held_setup decides anew,
    // and re-reads the conf, once the chain is known.
    held_reset(vf);
    p->held_on = false;
    p->held_why = NULL;
    if (!p->aji) {
        configure_scene(vf, false);
        update_depth(vf);
        return true;
    }

    p->api.set_slot(p->aji, p->pending_slot);
    p->cur_slot = p->pending_slot;

    int ow = 0, oh = 0;
    int ret = p->api.configure(p->aji, p->params.w, p->params.h, p->fps,
                               &ow, &oh);
    if (ret < 0) {
        configure_scene(vf, false);
        const char *error = p->api.last_error(p->aji);
        stream_status(vf, strstr(error, "AJN_ENGINE_MISSING") ? "engineMissing" :
                          strstr(error, "AJN_ENGINE_INCOMPATIBLE") ? "engineIncompatible" : "failed");
        MP_ERR(vf, "configure failed: %s\n", p->api.last_error(p->aji));
        return false;
    }
    write_stats(vf);

    // Every configure resets the engine's history, including asynchronous
    // preparation. Clear the caller's sequence state before any early return.
    p->temporal_r = ret > 0 ? p->api.temporal_radius(p->aji) : 0;
    p->seq0 = p->seq_next = 0;
    if (p->temporal_r)
        MP_VERBOSE(vf, "Temporal model: %d-frame window\n",
                   2 * p->temporal_r + 1);

    const char *details = p->api.current_log(p->aji);
    if (p->opts->stream_mode && strstr(details, "Engine build FAILED")) {
        configure_scene(vf, false);
        stream_status(vf, "preparationFailed");
        return false;
    }
    if (p->opts->stream_mode && strstr(details, "Building TensorRT engine")) {
        configure_scene(vf, false);
        stream_status(vf, "building");
        return true; // Only preparation can reach here on the matching backend.
    }

    int rn = 0, rd = 0;
    p->rife_on = false;
    p->rife_first = false;
    p->has_pre_resize = false;
    p->work_w = p->params.w;
    p->work_h = p->params.h;
    p->rife_num = p->rife_den = 1;
    mp_image_unrefp(&p->rife_prev);  // never interpolate across a reconfigure
    if (p->api.rife_factor(p->aji, &rn, &rd) && rd > 0 && rn > rd) {
        int a = rn, b = rd;
        while (b) {  // reduce the fraction
            int t = a % b;
            a = b;
            b = t;
        }
        rn /= a;
        rd /= a;
        if (rn <= 8 * rd) {  // at most 8 outputs per input pair (outq size)
            p->rife_num = rn;
            p->rife_den = rd;
            p->rife_on = true;
            MP_VERBOSE(vf, "RIFE active: %d/%d interpolation\n", rn, rd);
        } else {
            MP_WARN(vf, "RIFE factor %d/%d exceeds the supported 8x output "
                        "multiplication; interpolation disabled\n", rn, rd);
        }
    }
    p->rife_acc = p->rife_den;
    if (p->rife_on && p->is_vk && ret == 0 && !p->api.copy_frame) {
        // RIFE alone interpolates copies of the decoder frames: aji_copy_frame on Vulkan frames
        MP_WARN(vf, "RIFE without an upscale model needs a libaji with aji_copy_frame on Vulkan frames; interpolation disabled\n");
        p->rife_on = false;
    }

    configure_scene(vf, p->rife_on);

    if (ret == 0) {
        MP_VERBOSE(vf, "No chain active for %dx%d@%.3f slot %d (passthrough)\n",
                   p->params.w, p->params.h, p->fps, p->cur_slot);
        // RIFE-only chains interpolate the passthrough copies, which stay
        // in the input format; never inherit a stale out_fmt from a
        // previously configured upscale chain (out_params was just reset
        // to the input geometry above).
        p->out_fmt = p->aji_fmt;
        update_depth(vf);
        stream_status(vf, p->rife_on ? "active" : "bypass");
        return true;  // no scaling chain; rife (if any) still applies
    }

    p->aji_active = true;
    p->out_fmt = p->aji_fmt;
    if ((p->opts->output_444 || aji_format_is_444(p->aji_fmt)) &&
        !p->is_d3d11 && !p->is_vk) {
        // Preserve 4:4:4 sources in full 16-bit output, including MSB input.
        // The option forces full-resolution 4:4:4 even from a 4:2:0 source
        // (exceeds the reference pipeline, which always subsampled back to
        // 4:2:0). D3D11/DirectML stays 4:2:0 - DXGI has no planar 16-bit
        // 4:4:4 video format for the pool.
        p->out_fmt = AJI_FMT_YUV444P16;
    }
    if (p->out_fmt == AJI_FMT_YUV444P16) {
        // Either the option forced it, or a 4:4:4 source mirrors straight
        // through to 4:4:4 output. The model emits full-16-bit 4:4:4 planar;
        // tag the pool to match (444 input is CUDA-only, gated at reinit).
        p->out_params.hw_subfmt = pixfmt2imgfmt(AV_PIX_FMT_YUV444P16);
        // out_params was copied from the input; its bit encoding must follow
        // the format change to 16-bit, or the CUDA->Vulkan interop
        // (gpu-api=vulkan) renders the planes with a ~0.5 LSB normalization
        // shift (the d3d11/sw path re-derives this and is unaffected).
        p->out_params.repr.bits = (struct pl_bit_encoding){
            .sample_depth = 16, .color_depth = 16, .bit_shift = 0,
        };
    }
    double sx = (double)ow / p->params.w, sy = (double)oh / p->params.h;
    p->out_params.w = ow;
    p->out_params.h = oh;
    p->out_params.crop.x0 = lrint(p->params.crop.x0 * sx);
    p->out_params.crop.y0 = lrint(p->params.crop.y0 * sy);
    p->out_params.crop.x1 = lrint(p->params.crop.x1 * sx);
    p->out_params.crop.y1 = lrint(p->params.crop.y1 * sy);
    // The shim's RGB->YUV output is always left-sited, matching the
    // reference pipeline (VS/zimg doesn't propagate chroma location from
    // unsubsampled sources); tag the output accordingly.
    p->out_params.chroma_location = PL_CHROMA_LEFT;
    // RIFE-first: interpolate at the source resolution, then upscale every
    // frame. The shim configured RIFE for the matching resolution, so the
    // order here must agree with what aji_rife_before_upscale() reports
    // (TensorRT and DirectML both support it; older shims report 0).
    p->rife_first = p->rife_on && p->aji_active &&
                    p->api.rife_before_upscale &&
                    p->api.rife_before_upscale(p->aji);
    if (p->rife_first)
        MP_VERBOSE(vf, "RIFE-first: interpolating before upscaling\n");
    // Pre-RIFE downscale: when rife-first, ask the engine whether the first
    // model's "resize before upscale" was hoisted ahead of RIFE. If so, source
    // frames are downscaled to (work_w, work_h) via api.resize before being
    // interpolated and upscaled, so the RIFE + upscale path runs at work res.
    if (p->rife_first && p->api.pre_resize && p->api.resize) {
        int ww = 0, wh = 0;
        if (p->api.pre_resize(p->aji, &ww, &wh) && ww > 0 && wh > 0) {
            p->has_pre_resize = true;
            p->work_w = ww;
            p->work_h = wh;
            MP_VERBOSE(vf, "Pre-RIFE downscale: %dx%d -> %dx%d\n",
                       p->params.w, p->params.h, ww, wh);
        }
    }
    held_setup(vf);
    update_depth(vf);
    MP_VERBOSE(vf, "Configured slot %d: %dx%d -> %dx%d (depth %d)\n",
                p->cur_slot, p->params.w, p->params.h, ow, oh, p->depth);
    write_stats(vf);  // again, with the held-frame line
    stream_status(vf, "active");
    return true;
}


#if HAVE_D3D11
// Like mp_update_av_hw_frames_pool for IMGFMT_D3D11, but the textures
// also get the SHARED|NTHANDLE misc flags so the inference shim's D3D12
// device can open them.
static bool update_d3d11_pool(struct mp_filter *vf)
{
    struct priv *p = vf->priv;

    if (p->hw_pool) {
        AVHWFramesContext *hw_frames = (void *)p->hw_pool->data;
        if (hw_frames->width != p->out_params.w ||
            hw_frames->height != p->out_params.h ||
            hw_frames->sw_format != imgfmt2pixfmt(p->out_params.hw_subfmt))
            av_buffer_unref(&p->hw_pool);
    }
    if (p->hw_pool)
        return true;

    p->hw_pool = av_hwframe_ctx_alloc(p->av_device_ref);
    if (!p->hw_pool)
        return false;
    AVHWFramesContext *hw_frames = (void *)p->hw_pool->data;
    hw_frames->format = AV_PIX_FMT_D3D11;
    hw_frames->sw_format = imgfmt2pixfmt(p->out_params.hw_subfmt);
    hw_frames->width = p->out_params.w;
    hw_frames->height = p->out_params.h;
    AVD3D11VAFramesContext *d3d_frames = hw_frames->hwctx;
    d3d_frames->BindFlags = D3D11_BIND_RENDER_TARGET |
                            D3D11_BIND_SHADER_RESOURCE;
    d3d_frames->MiscFlags = D3D11_RESOURCE_MISC_SHARED |
                            D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
    if (av_hwframe_ctx_init(p->hw_pool) < 0) {
        av_buffer_unref(&p->hw_pool);
        return false;
    }
    return true;
}

// RIFE-first (D3D11): a source-resolution shareable pool for interpolation
// temps (and shareable staged copies of the source frames). Mirrors
// update_d3d11_pool but at the input geometry.
static bool update_d3d11_src_pool(struct mp_filter *vf)
{
    struct priv *p = vf->priv;

    if (p->src_pool) {
        AVHWFramesContext *hw_frames = (void *)p->src_pool->data;
        if (hw_frames->width != p->params.w ||
            hw_frames->height != p->params.h ||
            hw_frames->sw_format != imgfmt2pixfmt(p->params.hw_subfmt))
            av_buffer_unref(&p->src_pool);
    }
    if (p->src_pool)
        return true;

    p->src_pool = av_hwframe_ctx_alloc(p->av_device_ref);
    if (!p->src_pool)
        return false;
    AVHWFramesContext *hw_frames = (void *)p->src_pool->data;
    hw_frames->format = AV_PIX_FMT_D3D11;
    hw_frames->sw_format = imgfmt2pixfmt(p->params.hw_subfmt);
    hw_frames->width = p->params.w;
    hw_frames->height = p->params.h;
    AVD3D11VAFramesContext *d3d_frames = hw_frames->hwctx;
    d3d_frames->BindFlags = D3D11_BIND_RENDER_TARGET |
                            D3D11_BIND_SHADER_RESOURCE;
    d3d_frames->MiscFlags = D3D11_RESOURCE_MISC_SHARED |
                            D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
    if (av_hwframe_ctx_init(p->src_pool) < 0) {
        av_buffer_unref(&p->src_pool);
        return false;
    }
    return true;
}

// RIFE-first pre-resize (D3D11): a work-resolution shareable pool for the
// downscaled source frames + interpolation temps. Mirrors update_d3d11_src_pool
// but at the work geometry.
static bool update_d3d11_work_pool(struct mp_filter *vf)
{
    struct priv *p = vf->priv;

    if (p->work_pool) {
        AVHWFramesContext *hw_frames = (void *)p->work_pool->data;
        if (hw_frames->width != p->work_w ||
            hw_frames->height != p->work_h ||
            hw_frames->sw_format != imgfmt2pixfmt(p->params.hw_subfmt))
            av_buffer_unref(&p->work_pool);
    }
    if (p->work_pool)
        return true;

    p->work_pool = av_hwframe_ctx_alloc(p->av_device_ref);
    if (!p->work_pool)
        return false;
    AVHWFramesContext *hw_frames = (void *)p->work_pool->data;
    hw_frames->format = AV_PIX_FMT_D3D11;
    hw_frames->sw_format = imgfmt2pixfmt(p->params.hw_subfmt);
    hw_frames->width = p->work_w;
    hw_frames->height = p->work_h;
    AVD3D11VAFramesContext *d3d_frames = hw_frames->hwctx;
    d3d_frames->BindFlags = D3D11_BIND_RENDER_TARGET |
                            D3D11_BIND_SHADER_RESOURCE;
    d3d_frames->MiscFlags = D3D11_RESOURCE_MISC_SHARED |
                            D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
    if (av_hwframe_ctx_init(p->work_pool) < 0) {
        av_buffer_unref(&p->work_pool);
        return false;
    }
    return true;
}
#endif

static struct mp_image *alloc_out(struct mp_filter *vf)
{
    struct priv *p = vf->priv;

    bool pool_ok;
    if (p->is_d3d11) {
#if HAVE_D3D11
        pool_ok = update_d3d11_pool(vf);
#else
        pool_ok = false;
#endif
    } else if (p->is_vk) {
        pool_ok = mp_update_av_hw_frames_pool(&p->hw_pool, p->av_device_ref,
                                              IMGFMT_VULKAN,
                                              p->out_params.hw_subfmt,
                                              p->out_params.w,
                                              p->out_params.h, false);
    } else {
        pool_ok = mp_update_av_hw_frames_pool(&p->hw_pool, p->av_device_ref,
                                              IMGFMT_CUDA,
                                              p->out_params.hw_subfmt,
                                              p->out_params.w,
                                              p->out_params.h, false);
    }
    if (!pool_ok) {
        MP_ERR(vf, "Failed to create hw pool\n");
        return NULL;
    }

    AVFrame *av_frame = av_frame_alloc();
    MP_HANDLE_OOM(av_frame);
    if (av_hwframe_get_buffer(p->hw_pool, av_frame, 0) < 0) {
        MP_ERR(vf, "Failed to allocate frame from hw pool\n");
        av_frame_free(&av_frame);
        return NULL;
    }

    struct mp_image *img = mp_image_from_av_frame(av_frame);
    av_frame_free(&av_frame);
    if (!img) {
        MP_ERR(vf, "Internal error when converting AVFrame\n");
        return NULL;
    }

    return img;
}

// Hand the frames at relative positions rel..rel+r to the engine's history
// (each frame once, in order). Missing futures (end of stream) are simply
// not ingested; the engine clamps the window.
static bool ingest_window(struct mp_filter *vf, int rel)
{
    struct priv *p = vf->priv;
    const int mat = map_matrix(&p->params);
    const int rng = map_range(&p->params);
    const int sit = map_siting(&p->params);
#if HAVE_D3D11
    int staged = 0;
#endif
    for (int k = rel; k <= rel + p->temporal_r; k++) {
        const uint64_t seq = p->seq0 + k;
        if (seq < p->seq_next)
            continue;
        struct mp_image *f = mp_refqueue_get(p->queue, k);
        if (!f)
            break;
        aji_frame fin = {
            .width = f->w, .height = f->h, .format = p->aji_fmt,
            .matrix = mat, .range = rng, .siting = sit,
            .plane = {f->planes[0], f->planes[1], f->planes[2]},
            .stride = {f->stride[0], f->stride[1], f->stride[2]},
        };
        void *stream = p->stream;
#if HAVE_D3D11
        if (p->is_d3d11) {
            // Decoder surfaces aren't shareable: stage each frame like
            // submit_frame does. Earlier frames' stages are retired by the
            // previous emission's ticket wait; a stage reused within this
            // call (start of stream: up to r+1 ingests) must first finish
            // its previous ingest's D3D12 read.
            if (staged && staged % p->d3d_stage_count == 0 &&
                p->api.wait(p->aji, p->api.flush(p->aji, NULL)) != AJI_OK)
                return false;
            ID3D11Texture2D *stage = p->d3d_stage[p->d3d_stage_next];
            p->d3d_stage_next = (p->d3d_stage_next + 1) % p->d3d_stage_count;
            ID3D11DeviceContext_CopySubresourceRegion(p->d3d_ctx,
                (ID3D11Resource *)stage, 0, 0, 0, 0,
                (ID3D11Resource *)f->planes[0], (UINT)(intptr_t)f->planes[1],
                NULL);
            fin.plane[0] = stage;
            fin.plane[1] = 0;
            stream = NULL;
            staged++;
        }
#endif
#if HAVE_VULKAN
        if (p->is_vk) {
            // the decoder's AVVkFrame, read in libaji's submit (its semaphore orders it)
            struct vk_held h;
            if (!vk_take(vf, f, &h))
                return false;
            fin.plane[0] = &h.d;
            fin.plane[1] = NULL;
            const int ret = p->api.ingest(p->aji, &fin, seq, NULL);
            vk_give(&h, true);
            if (ret != AJI_OK)
                return false;
            p->seq_next = seq + 1;
            continue;
        }
#endif
        if (p->api.ingest(p->aji, &fin, seq, stream) != AJI_OK)
            return false;
        p->seq_next = seq + 1;
    }
    return true;
}

// aji_infer_rife with the endpoints' ids and the scene decision, through the newest entry point the
// library has.
static int call_rife(struct priv *p, const aji_frame *a, int64_t a_id, const aji_frame *b, int64_t b_id,
                     double t, const aji_frame *out, void *stream)
{
    if (p->api.infer_rife_ids)
        return p->api.infer_rife_ids(p->aji, a, a_id, b, b_id, t, out, stream, p->scene_decision);
    return p->scene_decision != -1 && p->api.infer_rife_with_scene
        ? p->api.infer_rife_with_scene(p->aji, a, b, t, out, stream, p->scene_decision)
        : p->api.infer_rife(p->aji, a, b, t, out, stream);
}

#if HAVE_VULKAN
// One frame through aji_create_vk's context.
static bool infer_vulkan(struct mp_filter *vf, struct mp_image *in,
                         struct mp_image *out, int mat, int rng, int sit)
{
    struct priv *p = vf->priv;
    struct vk_held hi, ho;
    if (!vk_take(vf, in, &hi))
        return false;
    if (!vk_take(vf, out, &ho)) {
        vk_give(&hi, false);
        return false;
    }
    const aji_frame fin = {
        .width = in->w, .height = in->h, .format = p->aji_fmt,
        .matrix = mat, .range = rng, .siting = sit,
        .plane = {&hi.d},
    };
    const aji_frame fout = {
        .width = p->out_params.w, .height = p->out_params.h, .format = p->out_fmt,
        .matrix = mat, .range = rng, .siting = sit,
        .plane = {&ho.d},
    };
    const bool ok = p->api.infer(p->aji, &fin, &fout, NULL) == AJI_OK;
    vk_give(&ho, ok);
    vk_give(&hi, ok);
    return ok;
}

// A temporal chain's frame `seq` (ingested) into out.
static bool infer_seq_vulkan(struct mp_filter *vf, uint64_t seq, struct mp_image *out,
                             int mat, int rng, int sit)
{
    struct priv *p = vf->priv;
    struct vk_held ho;
    if (!vk_take(vf, out, &ho))
        return false;
    const aji_frame fout = {
        .width = p->out_params.w, .height = p->out_params.h, .format = p->out_fmt,
        .matrix = mat, .range = rng, .siting = sit,
        .plane = {&ho.d},
    };
    const bool ok = p->api.infer_seq(p->aji, seq, &fout, NULL) == AJI_OK;
    vk_give(&ho, true);
    return ok;
}

// RIFE between a and b into out (w x h, format fmt), synchronous like the
// other backends': AJI_OK, AJI_SCENE or an error.
static int rife_vulkan(struct mp_filter *vf, struct mp_image *a, int64_t a_id, struct mp_image *b,
                       int64_t b_id, struct mp_image *out, double t, int w, int h, int fmt,
                       int mat, int rng, int sit)
{
    struct priv *p = vf->priv;
    struct vk_held held[3];
    struct mp_image *imgs[3] = {a, b, out};
    int n = 0;
    for (; n < 3; n++)
        if (!vk_take(vf, imgs[n], &held[n]))
            break;
    int ret = AJI_ERR;
    if (n == 3) {
        aji_frame f[3];
        for (int i = 0; i < 3; i++)
            f[i] = (aji_frame){
                .width = w, .height = h, .format = fmt,
                .matrix = mat, .range = rng, .siting = sit,
                .plane = {&held[i].d},
            };
        ret = call_rife(p, &f[0], a_id, &f[1], b_id, t, &f[2], NULL);
    }
    // libaji updates an image's state only for a submit it made (the scene
    // check reads a and b, then may return before touching out): always take it
    while (n-- > 0)
        vk_give(&held[n], true);
    return ret;
}

// FFmpeg's queue lock, deprecated for VK_KHR_internally_synchronized_queues
// but still what serializes submits on devices without it. Gone with the API
// (the queues then synchronize themselves): nothing to hold.
#if !defined(FF_API_VULKAN_SYNC_QUEUES) || FF_API_VULKAN_SYNC_QUEUES
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
static void vk_lock_queue(void *opaque, uint32_t family, uint32_t index)
{
    AVHWDeviceContext *dc = opaque;
    AVVulkanDeviceContext *vk = dc->hwctx;
    if (vk->lock_queue)
        vk->lock_queue(dc, family, index);
}

static void vk_unlock_queue(void *opaque, uint32_t family, uint32_t index)
{
    AVHWDeviceContext *dc = opaque;
    AVVulkanDeviceContext *vk = dc->hwctx;
    if (vk->unlock_queue)
        vk->unlock_queue(dc, family, index);
}
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#else
static void vk_lock_queue(void *opaque, uint32_t family, uint32_t index)
{
    (void)opaque, (void)family, (void)index;
}

static void vk_unlock_queue(void *opaque, uint32_t family, uint32_t index)
{
    (void)opaque, (void)family, (void)index;
}
#endif

// The decoder's Vulkan device as libaji takes it: a compute queue of its
// families (FFmpeg's list) and its lock, what it was created with.
static bool adopt_vulkan(struct mp_filter *vf, AVHWDeviceContext *dc)
{
    struct priv *p = vf->priv;
    AVVulkanDeviceContext *vk = dc->hwctx;
    int fam = -1;
    for (int i = 0; i < vk->nb_qf && fam < 0; i++)
        if (vk->qf[i].flags & VK_QUEUE_COMPUTE_BIT)
            fam = vk->qf[i].idx;
    if (fam < 0) {
        MP_ERR(vf, "The Vulkan device has no compute queue\n");
        return false;
    }
    PFN_vkGetDeviceQueue2 get_queue2 =
        (PFN_vkGetDeviceQueue2)vk->get_proc_addr(vk->inst, "vkGetDeviceQueue2");
    VkQueue queue = VK_NULL_HANDLE;
    VkDeviceQueueInfo2 qi = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2,
        .queueFamilyIndex = fam,
        .queueIndex = 0,
    };
#if defined(VK_KHR_internally_synchronized_queues) && LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(60, 32, 100)
    // libplacebo creates its queues internally synchronized when lavu takes the
    // flag (video/out/vulkan/context.c): those come back only with it
    qi.flags = VK_DEVICE_QUEUE_CREATE_INTERNALLY_SYNCHRONIZED_BIT_KHR;
    if (get_queue2)
        get_queue2(vk->act_dev, &qi, &queue);
    qi.flags = 0;
#endif
    if (!queue && get_queue2)
        get_queue2(vk->act_dev, &qi, &queue);
    if (!queue) {
        MP_ERR(vf, "Cannot get the Vulkan device's compute queue\n");
        return false;
    }
    p->vkdev = (aji_vk_device){
        .struct_size = sizeof(aji_vk_device),
        .get_instance_proc_addr = (void *)vk->get_proc_addr,
        .instance = vk->inst,
        .physical_device = vk->phys_dev,
        .device = vk->act_dev,
        .queue = queue,
        .queue_family = fam,
        .queue_index = 0,
        .lock_queue = vk_lock_queue,
        .unlock_queue = vk_unlock_queue,
        .lock_opaque = dc,
        .enabled_extensions = vk->enabled_dev_extensions,
        .num_enabled_extensions = vk->nb_enabled_dev_extensions,
        .enabled_features = &vk->device_features,
    };
    return true;
}
#endif

// Enqueue inference for `in` and append it to the in-flight ring without
// waiting for the GPU. `in` stays alive (and the decoder surface pinned)
// through the refqueue until emission waits the entry's ticket. rel is
// `in`'s refqueue position (temporal chains read its neighbors), or -1 for
// a frame that is not in the refqueue.
static bool submit_frame(struct mp_filter *vf, struct mp_image *in, int rel)
{
    struct priv *p = vf->priv;

    mp_assert(p->ring_n < MAX_DEPTH);

    // Held-frame reuse decides every refqueue frame of a plain chain, once
    // and in order (see held_decide).
    const bool held = p->held_on && rel >= 0;
    struct mp_image *stage = NULL;  // D3D11 + reuse: in's staged copy
    if (held && held_decide(vf, in, rel, &stage)) {
        struct mp_image *out = mp_image_new_ref(p->anchor_out);
        MP_HANDLE_OOM(out);
        mp_image_copy_attributes(out, in);
        out->params = p->out_params;
        out->pts = in->pts;
        mp_image_unrefp(&stage);
        p->ring[p->ring_n].src = in;
        p->ring[p->ring_n].out = out;
        p->ring[p->ring_n].stage = NULL;
        p->ring[p->ring_n].ticket = p->anchor_ticket;
        p->ring_n++;
        return true;
    }

    struct mp_image *out = alloc_out(vf);
    if (!out) {
        mp_image_unrefp(&stage);
        if (held)
            held_reset(vf);  // the frame comes back to be decided again
        return false;
    }
    mp_image_copy_attributes(out, in);
    out->params = p->out_params;
    out->pts = in->pts;

    const int mat = map_matrix(&p->params);
    const int rng = map_range(&p->params);
    const int sit = map_siting(&p->params);
    bool ok;
    void *stream = NULL;

#if HAVE_VULKAN
    if (p->is_vk) {
        if (p->temporal_r && rel >= 0)
            ok = ingest_window(vf, rel) && infer_seq_vulkan(vf, p->seq0 + rel, out, mat, rng, sit);
        else
            ok = infer_vulkan(vf, in, out, mat, rng, sit);
    } else
#endif
#if HAVE_D3D11
    if (p->is_d3d11) {
        // Input geometry comes from the frame: the decoder size on the normal
        // path, the work (downscaled) size in rife-first mode.
        aji_frame fin = {
            .width = in->w, .height = in->h,
            .format = p->aji_fmt,
            .matrix = mat, .range = rng, .siting = sit,
        };
        if (p->rife_first || (p->temporal_r && rel >= 0)) {
            // rife-first frames (the downscaled source / interp temps) are
            // already shareable work-res textures; feed them to the shim
            // directly, no decoder staging. Temporal chains stage in
            // ingest_window instead.
            fin.plane[0] = in->planes[0];
            fin.plane[1] = in->planes[1];
        } else if (stage) {
            // held-frame reuse staged it already (pooled, refcounted; the
            // ring keeps it until the ticket was waited)
            fin.plane[0] = stage->planes[0];
            fin.plane[1] = stage->planes[1];
        } else {
            // Stage the decode slice into a shareable texture first: decoder
            // surfaces are neither shareable nor synchronized for other
            // devices, and the same-device copy orders against decode on the
            // immediate context (vf_amf.c precedent).
            ID3D11Texture2D *tex = p->d3d_stage[p->d3d_stage_next];
            p->d3d_stage_next = (p->d3d_stage_next + 1) % p->d3d_stage_count;
            ID3D11DeviceContext_CopySubresourceRegion(p->d3d_ctx,
                (ID3D11Resource *)tex, 0, 0, 0, 0,
                (ID3D11Resource *)in->planes[0], (UINT)(intptr_t)in->planes[1],
                NULL);
            fin.plane[0] = tex;
            fin.plane[1] = 0;
        }
        const aji_frame fout = {
            .width = p->out_params.w, .height = p->out_params.h,
            .format = p->aji_fmt,
            .matrix = mat, .range = rng, .siting = sit,
            .plane = {out->planes[0], out->planes[1]},
        };
        if (p->temporal_r && rel >= 0) {
            ok = ingest_window(vf, rel) &&
                 p->api.infer_seq(p->aji, p->seq0 + rel, &fout,
                                  NULL) == AJI_OK;
        } else {
            ok = p->api.infer(p->aji, &fin, &fout, NULL) == AJI_OK;
        }
    } else
#endif
    {
        const aji_frame fin = {
            // Input geometry from the frame: decoder size normally, work
            // (downscaled) size in rife-first mode.
            .width = in->w, .height = in->h, .format = p->aji_fmt,
            .matrix = mat, .range = rng, .siting = sit,
            // plane[2]/stride[2] carry Cr for 4:4:4 input; ignored by the
            // engine for the 2-plane NV12/P010 formats.
            .plane = {in->planes[0], in->planes[1], in->planes[2]},
            .stride = {in->stride[0], in->stride[1], in->stride[2]},
        };
        const aji_frame fout = {
            .width = p->out_params.w, .height = p->out_params.h,
            .format = p->out_fmt,
            .matrix = mat, .range = rng, .siting = sit,
            .plane = {out->planes[0], out->planes[1], out->planes[2]},
            .stride = {out->stride[0], out->stride[1], out->stride[2]},
        };
        if (!order_after_decode(vf)) {
            talloc_free(out);
            if (held)
                held_reset(vf);
            return false;
        }
        stream = p->stream;
        if (p->temporal_r && rel >= 0) {

            ok = ingest_window(vf, rel) &&
                 p->api.infer_seq(p->aji, p->seq0 + rel, &fout,
                                  p->stream) == AJI_OK;
        } else {
            ok = p->api.infer(p->aji, &fin, &fout, p->stream) == AJI_OK;
        }
    }

    uint64_t ticket = ok ? p->api.flush(p->aji, stream) : 0;
    if (!ticket) {
        MP_ERR(vf, "inference failed: %s\n", p->api.last_error(p->aji));
        // a failed call may still have queued partial device work that
        // references `out`; let it finish before the frame is freed
        p->api.wait(p->aji, p->api.flush(p->aji, stream));
        talloc_free(out);
        mp_image_unrefp(&stage);
        if (held)
            held_reset(vf);
        return false;
    }

    // An upscaled frame becomes the anchor (unless reuse just failed).
    if (held && p->held_on)
        held_set_anchor(vf, in, rel, stage, out, ticket);

    p->ring[p->ring_n].src = in;
    p->ring[p->ring_n].out = out;
    p->ring[p->ring_n].stage = stage;
    p->ring[p->ring_n].ticket = ticket;
    p->ring_n++;
    return true;
}

// Pop the oldest in-flight frame, waiting until its GPU work (writes to
// the output, reads of the decoder surface) completed.
static struct mp_image *pop_ring(struct mp_filter *vf)
{
    struct priv *p = vf->priv;

    mp_assert(p->ring_n > 0);
    struct mp_image *out = p->ring[0].out;
    struct mp_image *stage = p->ring[0].stage;
    uint64_t ticket = p->ring[0].ticket;
    p->ring_n--;
    memmove(&p->ring[0], &p->ring[1], p->ring_n * sizeof(p->ring[0]));

    if (p->api.wait(p->aji, ticket) != AJI_OK) {
        MP_ERR(vf, "inference wait failed: %s\n", p->api.last_error(p->aji));
        talloc_free(out);
        out = NULL;
    }
    mp_image_unrefp(&stage);  // its reads are done (or the device is lost)
    return out;
}

static struct mp_image *render(struct mp_filter *vf)
{
    struct priv *p = vf->priv;
    CUcontext dummy;
    int ret = -1;

    struct mp_image *in = mp_refqueue_get(p->queue, 0);
    if (!in)
        return NULL;

    // Bypass (slot 0 / no matching chain): hand the decoder frame through
    // untouched — exactly as safe as filterless playback. With RIFE
    // active the copy stays: the buffered prev/cur frames (+ outq) would
    // otherwise pin entries of the decoder's fixed surface ring, which is
    // the documented copy-on-arrival rationale.
    if (!p->aji_active && !p->rife_on)
        return mp_image_new_ref(in);

    if (p->aji_active) {
        // Synchronous inference (RIFE chains, queue-depth=1, pipeline
        // fallback): submit and immediately wait.
        if (!submit_frame(vf, in, 0))
            return NULL;
        return pop_ring(vf);
    }

    struct mp_image *out = alloc_out(vf);
    if (!out)
        return NULL;

    mp_image_copy_attributes(out, in);
    out->params = p->out_params;
    out->pts = in->pts;

#if HAVE_VULKAN
    if (p->is_vk) {
        // Passthrough: an exact copy into the output pool's frame
        struct vk_held hi, ho;
        bool ok = false;
        if (vk_take(vf, in, &hi)) {
            if (vk_take(vf, out, &ho)) {
                const aji_frame fsrc = {
                    .width = in->w, .height = in->h, .format = p->aji_fmt,
                    .plane = {&hi.d},
                };
                aji_frame fdst = fsrc;
                fdst.plane[0] = &ho.d;
                ok = p->api.copy_frame(p->aji, &fsrc, &fdst) == AJI_OK;
                vk_give(&ho, true);
            }
            vk_give(&hi, true);
        }
        if (!ok) {
            MP_ERR(vf, "frame copy failed: %s\n", p->api.last_error(p->aji));
            talloc_free(out);
            return NULL;
        }
        return out;
    }
#endif
#if HAVE_D3D11
    if (p->is_d3d11) {
        // Passthrough: same-device copy into the output slice (implicitly
        // synchronized on the immediate context).
        ID3D11DeviceContext_CopySubresourceRegion(p->d3d_ctx,
            (ID3D11Resource *)out->planes[0],
            (UINT)(intptr_t)out->planes[1], 0, 0, 0,
            (ID3D11Resource *)in->planes[0], (UINT)(intptr_t)in->planes[1],
            NULL);
        return out;
    }
#endif

    if (!order_after_decode(vf))
        goto done;

    if (CHECK_CU(p->cu->cuCtxPushCurrent(p->cuda_ctx)) < 0)
        goto done;

    // Passthrough: synchronized copy out of the decoder surface ring.
    for (int n = 0; n < p->layout.fmt.num_planes; n++) {
        CUDA_MEMCPY2D cpy = {
            .srcMemoryType = CU_MEMORYTYPE_DEVICE,
            .srcDevice     = (CUdeviceptr)(uintptr_t)in->planes[n],
            .srcPitch      = in->stride[n],
            .dstMemoryType = CU_MEMORYTYPE_DEVICE,
            .dstDevice     = (CUdeviceptr)(uintptr_t)out->planes[n],
            .dstPitch      = out->stride[n],
            .WidthInBytes  = mp_image_plane_w(&p->layout, n) *
                             p->layout.fmt.bpp[n] / 8,
            .Height        = mp_image_plane_h(&p->layout, n),
        };
        if (CHECK_CU(p->cu->cuMemcpy2DAsync(&cpy, p->stream)) < 0)
            goto pop;
    }

    if (CHECK_CU(p->cu->cuStreamSynchronize(p->stream)) < 0)
        goto pop;

    ret = 0;
pop:
    CHECK_CU(p->cu->cuCtxPopCurrent(&dummy));
done:
    if (ret < 0)
        TA_FREEP(&out);
    return out;
}

// Evaluate once per original pair, independent of the interpolation factor.
// Consume only a completed look-ahead result. Ordinary playback has no worker,
// readback, extra references or waits for addon decisions.
static int scene_pair(struct mp_filter *vf, struct mp_image *a, struct mp_image *b)
{
    struct priv *p = vf->priv;
    if (!ajn_scene_active(p->scene)) return -1;
    bool supported = p->api.infer_rife_with_scene && p->api.rife_scene_supported &&
        p->api.rife_scene_supported(p->aji);
    return ajn_scene_decide(p->scene, a, b, supported);
}

// Interpolate between two upscaled frames at time point t. Returns the
// interpolated image, a new ref of `a` on a scene change (the reference
// pipeline substitutes the left frame), or NULL on error. pts is left for
// the caller to set.
static struct mp_image *render_interp(struct mp_filter *vf,
                                      struct mp_image *a, int64_t a_id, struct mp_image *b,
                                      int64_t b_id, double t)
{
    struct priv *p = vf->priv;
    CUcontext dummy;

    struct mp_image *out = alloc_out(vf);
    if (!out)
        return NULL;
    mp_image_copy_attributes(out, b);

    const int mat = map_matrix(&p->out_params);
    const int rng = map_range(&p->out_params);
    const aji_frame fa = {
        .width = p->out_params.w, .height = p->out_params.h,
        .format = p->out_fmt, .matrix = mat, .range = rng,
        .siting = AJI_SITING_LEFT,
        .plane = {a->planes[0], a->planes[1], a->planes[2]},
        .stride = {a->stride[0], a->stride[1], a->stride[2]},
    };
    aji_frame fb = fa, fout = fa;
    for (int i = 0; i < 3; i++) {
        fb.plane[i] = b->planes[i];
        fb.stride[i] = b->stride[i];
        fout.plane[i] = out->planes[i];
        fout.stride[i] = out->stride[i];
    }

    // b is the just-decoded source frame; order the read after decode (same
    // HAGS race the upscale path guards against).
    if (!order_after_decode(vf)) {
        talloc_free(out);
        return NULL;
    }
    int ret;
#if HAVE_VULKAN
    if (p->is_vk)
        ret = rife_vulkan(vf, a, a_id, b, b_id, out, t, p->out_params.w, p->out_params.h, p->out_fmt, mat, rng,
                          AJI_SITING_LEFT);
    else
#endif
    ret = call_rife(p, &fa, a_id, &fb, b_id, t, &fout, p->stream);
    if (ret == AJI_SCENE) {
        talloc_free(out);
        return mp_image_new_ref(a);
    }
    if (ret != AJI_OK) {
        MP_ERR(vf, "interpolation failed: %s\n", p->api.last_error(p->aji));
        talloc_free(out);
        return NULL;
    }
    bool ok = p->is_d3d11 || p->is_vk;  // DirectML and Vulkan frames complete synchronously
    if (!ok && CHECK_CU(p->cu->cuCtxPushCurrent(p->cuda_ctx)) >= 0) {
        ok = CHECK_CU(p->cu->cuStreamSynchronize(p->stream)) >= 0;
        CHECK_CU(p->cu->cuCtxPopCurrent(&dummy));
    }
    if (!ok)
        TA_FREEP(&out);
    return out;
}

// RIFE-first (CUDA): allocate a source-resolution hwframe for an
// interpolation temp (the upscale models then consume it). Mirrors alloc_out
// but at the input geometry and from a dedicated pool.
static struct mp_image *alloc_src(struct mp_filter *vf)
{
    struct priv *p = vf->priv;

    bool pool_ok;
    if (p->is_d3d11) {
#if HAVE_D3D11
        pool_ok = update_d3d11_src_pool(vf);
#else
        pool_ok = false;
#endif
    } else {
        pool_ok = mp_update_av_hw_frames_pool(&p->src_pool, p->av_device_ref,
                                              p->is_vk ? IMGFMT_VULKAN : IMGFMT_CUDA,
                                              p->params.hw_subfmt,
                                              p->params.w, p->params.h, false);
    }
    if (!pool_ok) {
        MP_ERR(vf, "Failed to create source hw pool\n");
        return NULL;
    }
    AVFrame *av_frame = av_frame_alloc();
    MP_HANDLE_OOM(av_frame);
    if (av_hwframe_get_buffer(p->src_pool, av_frame, 0) < 0) {
        MP_ERR(vf, "Failed to allocate source frame from hw pool\n");
        av_frame_free(&av_frame);
        return NULL;
    }
    struct mp_image *img = mp_image_from_av_frame(av_frame);
    av_frame_free(&av_frame);
    if (!img)
        MP_ERR(vf, "Internal error when converting source AVFrame\n");
    return img;
}

// RIFE-first: allocate a work-resolution frame (interpolation temps and the
// pre-RIFE downscale output). When there is no pre-resize the work res equals
// the source res, so this is just alloc_src.
static struct mp_image *alloc_work(struct mp_filter *vf)
{
    struct priv *p = vf->priv;
    if (!p->has_pre_resize)
        return alloc_src(vf);

    bool pool_ok;
    if (p->is_d3d11) {
#if HAVE_D3D11
        pool_ok = update_d3d11_work_pool(vf);
#else
        pool_ok = false;
#endif
    } else {
        pool_ok = mp_update_av_hw_frames_pool(&p->work_pool, p->av_device_ref,
                                              p->is_vk ? IMGFMT_VULKAN : IMGFMT_CUDA,
                                              p->params.hw_subfmt,
                                              p->work_w, p->work_h, false);
    }
    if (!pool_ok) {
        MP_ERR(vf, "Failed to create work hw pool\n");
        return NULL;
    }
    AVFrame *av_frame = av_frame_alloc();
    MP_HANDLE_OOM(av_frame);
    if (av_hwframe_get_buffer(p->work_pool, av_frame, 0) < 0) {
        MP_ERR(vf, "Failed to allocate work frame from hw pool\n");
        av_frame_free(&av_frame);
        return NULL;
    }
    struct mp_image *img = mp_image_from_av_frame(av_frame);
    av_frame_free(&av_frame);
    if (!img)
        MP_ERR(vf, "Internal error when converting work AVFrame\n");
    return img;
}

#if HAVE_D3D11
// RIFE-first (D3D11): copy a decoder source texture into a fresh shareable
// source-res texture, so the shim's D3D12 device can open it for interpolation
// (decoder textures are neither shareable nor synchronized). The same staged
// copy serves as a rife endpoint for two pairs and as an upscale input.
static struct mp_image *stage_src_d3d11(struct mp_filter *vf,
                                        struct mp_image *src)
{
    struct priv *p = vf->priv;
    struct mp_image *out = alloc_src(vf);
    if (!out)
        return NULL;
    mp_image_copy_attributes(out, src);
    out->params = p->params;
    ID3D11DeviceContext_CopySubresourceRegion(p->d3d_ctx,
        (ID3D11Resource *)out->planes[0], (UINT)(intptr_t)out->planes[1],
        0, 0, 0,
        (ID3D11Resource *)src->planes[0], (UINT)(intptr_t)src->planes[1],
        NULL);
    return out;
}
#endif

// RIFE-first pre-resize: downscale a decoder source frame to the work
// resolution via aji_resize, returning an owned work-res frame (the caller
// interpolates and upscales it). NULL on error. CUDA reads the decoder frame
// directly; D3D11 stages it into the shareable d3d_stage ring (free in
// rife-first mode, since submit feeds the shim's textures directly) first.
static struct mp_image *downscale_src(struct mp_filter *vf,
                                      struct mp_image *src)
{
    struct priv *p = vf->priv;

    struct mp_image *work = alloc_work(vf);
    if (!work)
        return NULL;
    mp_image_copy_attributes(work, src);
    work->params = p->params;
    work->params.w = p->work_w;
    work->params.h = p->work_h;

    const int mat = map_matrix(&p->params);
    const int rng = map_range(&p->params);
    const int sit = map_siting(&p->params);
    int ret;

#if HAVE_D3D11
    if (p->is_d3d11) {
        ID3D11Texture2D *stage = p->d3d_stage[p->d3d_stage_next];
        p->d3d_stage_next = (p->d3d_stage_next + 1) % p->d3d_stage_count;
        ID3D11DeviceContext_CopySubresourceRegion(p->d3d_ctx,
            (ID3D11Resource *)stage, 0, 0, 0, 0,
            (ID3D11Resource *)src->planes[0], (UINT)(intptr_t)src->planes[1],
            NULL);
        const aji_frame fin = {
            .width = p->params.w, .height = p->params.h, .format = p->aji_fmt,
            .matrix = mat, .range = rng, .siting = sit,
            .plane = {stage, 0},
        };
        const aji_frame fout = {
            .width = p->work_w, .height = p->work_h, .format = p->aji_fmt,
            .matrix = mat, .range = rng, .siting = sit,
            .plane = {work->planes[0], work->planes[1]},
        };
        ret = p->api.resize(p->aji, &fin, &fout, NULL);
    } else
#endif
#if HAVE_VULKAN
    if (p->is_vk) {
        struct vk_held hs, hw;
        ret = AJI_ERR;
        if (vk_take(vf, src, &hs)) {
            if (vk_take(vf, work, &hw)) {
                const aji_frame fin = {
                    .width = p->params.w, .height = p->params.h, .format = p->aji_fmt,
                    .matrix = mat, .range = rng, .siting = sit,
                    .plane = {&hs.d},
                };
                const aji_frame fout = {
                    .width = p->work_w, .height = p->work_h, .format = p->aji_fmt,
                    .matrix = mat, .range = rng, .siting = sit,
                    .plane = {&hw.d},
                };
                ret = p->api.resize(p->aji, &fin, &fout, NULL);
                vk_give(&hw, true);
            }
            vk_give(&hs, true);
        }
    } else
#endif
    {
        if (!order_after_decode(vf)) {
            talloc_free(work);
            return NULL;
        }
        const aji_frame fin = {
            .width = p->params.w, .height = p->params.h, .format = p->aji_fmt,
            .matrix = mat, .range = rng, .siting = sit,
            .plane = {src->planes[0], src->planes[1], src->planes[2]},
            .stride = {src->stride[0], src->stride[1], src->stride[2]},
        };
        const aji_frame fout = {
            .width = p->work_w, .height = p->work_h, .format = p->aji_fmt,
            .matrix = mat, .range = rng, .siting = sit,
            .plane = {work->planes[0], work->planes[1], work->planes[2]},
            .stride = {work->stride[0], work->stride[1], work->stride[2]},
        };
        ret = p->api.resize(p->aji, &fin, &fout, p->stream);
    }

    if (ret != AJI_OK) {
        MP_ERR(vf, "pre-RIFE downscale failed: %s\n",
               p->api.last_error(p->aji));
        talloc_free(work);
        return NULL;
    }
    return work;
}

// RIFE-first: interpolate two work-resolution frames at time t into a fresh
// work-resolution frame (the caller upscales it). Without a pre-RIFE downscale
// the work res equals the source res. Returns a new ref of `a` on a scene
// change, NULL on error. pts is left for the caller.
static struct mp_image *interp_source(struct mp_filter *vf,
                                      struct mp_image *a, int64_t a_id, struct mp_image *b,
                                      int64_t b_id, double t)
{
    struct priv *p = vf->priv;
    CUcontext dummy;

    struct mp_image *out = alloc_work(vf);
    if (!out)
        return NULL;
    mp_image_copy_attributes(out, b);
    out->params = p->params;
    out->params.w = p->work_w;
    out->params.h = p->work_h;

    const int mat = map_matrix(&p->params);
    const int rng = map_range(&p->params);
    const int sit = map_siting(&p->params);
    const aji_frame fa = {
        .width = p->work_w, .height = p->work_h, .format = p->aji_fmt,
        .matrix = mat, .range = rng, .siting = sit,
        .plane = {a->planes[0], a->planes[1], a->planes[2]},
        .stride = {a->stride[0], a->stride[1], a->stride[2]},
    };
    aji_frame fb = fa, fout = fa;
    for (int i = 0; i < 3; i++) {
        fb.plane[i] = b->planes[i];
        fb.stride[i] = b->stride[i];
        fout.plane[i] = out->planes[i];
        fout.stride[i] = out->stride[i];
    }

    int ret;
#if HAVE_VULKAN
    if (p->is_vk)
        ret = rife_vulkan(vf, a, a_id, b, b_id, out, t, p->work_w, p->work_h, p->aji_fmt, mat, rng, sit);
    else
#endif
    ret = call_rife(p, &fa, a_id, &fb, b_id, t, &fout, p->stream);
    if (ret == AJI_SCENE) {
        talloc_free(out);
        return mp_image_new_ref(a);
    }
    if (ret != AJI_OK) {
        MP_ERR(vf, "interpolation failed: %s\n", p->api.last_error(p->aji));
        talloc_free(out);
        return NULL;
    }
    bool ok = p->is_d3d11 || p->is_vk;   // DirectML and Vulkan frames complete synchronously
    if (!ok && CHECK_CU(p->cu->cuCtxPushCurrent(p->cuda_ctx)) >= 0) {
        ok = CHECK_CU(p->cu->cuStreamSynchronize(p->stream)) >= 0;
        CHECK_CU(p->cu->cuCtxPopCurrent(&dummy));
    }
    if (!ok)
        TA_FREEP(&out);
    return out;
}

// RIFE-first: upscale one source-resolution frame synchronously (submit +
// wait), used on the source frame and on each interpolation temp.
static struct mp_image *upscale_image(struct mp_filter *vf,
                                      struct mp_image *in)
{
    if (!submit_frame(vf, in, -1))
        return NULL;
    return pop_ring(vf);
}

static void vf_animejanai_process(struct mp_filter *vf)
{
    struct priv *p = vf->priv;

    struct mp_image *in_fmt = mp_refqueue_execute_reinit(p->queue);
    if (in_fmt) {
        if (p->configured || p->aji || p->av_device_ref)
            MP_VERBOSE(vf, "Input reinit: rebuilding AnimeJaNai device state\n");
        release_device_state(vf);

        p->params = in_fmt->params;
        p->fps = in_fmt->nominal_fps;
        if (!p->params.hw_subfmt) {
            MP_ERR(vf, "Unknown hw_subfmt\n");
            mp_filter_internal_mark_failed(vf);
            return;
        }

        // Adopt the device (and thus CUcontext) the incoming frames live on.
        if (!in_fmt->hwctx) {
            MP_ERR(vf, "Input frame has no hw frames context\n");
            mp_filter_internal_mark_failed(vf);
            return;
        }
        AVHWFramesContext *fctx = (void *)in_fmt->hwctx->data;
        AVHWDeviceContext *avhwctx = fctx->device_ctx;
        if (avhwctx->type == AV_HWDEVICE_TYPE_CUDA && p->cu) {
            p->is_d3d11 = false;
            p->av_device_ref = av_buffer_ref(fctx->device_ref);
            MP_HANDLE_OOM(p->av_device_ref);
            AVCUDADeviceContext *cudactx = avhwctx->hwctx;
            p->cuda_ctx = cudactx->cuda_ctx;
            p->stream = cudactx->stream;
            p->decode_stream = cudactx->stream;   // the real decode stream (may be NULL)
            if (!p->stream) {
                // ffmpeg's CUDA device context usually leaves this NULL (the
                // default stream); TensorRT then adds extra stream syncs per
                // enqueue, and CUDA graph capture is impossible. Use our own.
                // (Format re-adoptions reuse it; the device ctx is stable.)
                if (!p->own_stream &&
                    CHECK_CU(p->cu->cuCtxPushCurrent(p->cuda_ctx)) >= 0) {
                    CUcontext dummy;
                    if (CHECK_CU(p->cu->cuStreamCreate(&p->own_stream,
                                            CU_STREAM_NON_BLOCKING)) >= 0) {
                        // A non-blocking stream does not order against the
                        // default stream where decode/copy runs; the events
                        // give that ordering explicitly (order_after_decode).
                        bool evt_ok = true;
                        for (int i = 0; i < MAX_DEPTH; i++)
                            evt_ok &= CHECK_CU(p->cu->cuEventCreate(
                                &p->decode_evt[i],
                                CU_EVENT_DISABLE_TIMING)) >= 0;
                        if (evt_ok) {
                            p->decode_evt_ok = true;
                        } else {
                            // Without the events the private stream cannot be
                            // ordered after decode. Don't keep it and silently
                            // race: drop it and run inference on the decode/
                            // default stream (implicitly ordered, at the cost
                            // of per-enqueue TRT syncs and no CUDA graphs).
                            MP_WARN(vf, "CUDA event setup failed; falling back "
                                    "to the decode stream (no CUDA graphs)\n");
                            for (int i = 0; i < MAX_DEPTH; i++) {
                                if (p->decode_evt[i])
                                    p->cu->cuEventDestroy(p->decode_evt[i]);
                                p->decode_evt[i] = NULL;
                            }
                            p->cu->cuStreamDestroy(p->own_stream);
                            p->own_stream = NULL;
                        }
                    }
                    CHECK_CU(p->cu->cuCtxPopCurrent(&dummy));
                }
                // own_stream is NULL if creation/event setup failed; then
                // p->stream stays the decode (default) stream and inference
                // is ordered implicitly, never racing.
                p->stream = p->own_stream;
            }
#if HAVE_D3D11
        } else if (avhwctx->type == AV_HWDEVICE_TYPE_D3D11VA) {
            p->is_d3d11 = true;
            p->av_device_ref = av_buffer_ref(fctx->device_ref);
            MP_HANDLE_OOM(p->av_device_ref);
            AVD3D11VADeviceContext *d3dctx = avhwctx->hwctx;
            p->d3d_dev = d3dctx->device;
            p->d3d_ctx = d3dctx->device_context;
            p->stream = NULL;
#endif
#if HAVE_VULKAN
        } else if (avhwctx->type == AV_HWDEVICE_TYPE_VULKAN) {
            if (!adopt_vulkan(vf, avhwctx)) {
                mp_filter_internal_mark_failed(vf);
                return;
            }
            p->is_vk = true;
            p->av_device_ref = av_buffer_ref(fctx->device_ref);
            MP_HANDLE_OOM(p->av_device_ref);
            p->stream = NULL;
#endif
        } else {
            MP_ERR(vf, "Input frames are neither CUDA, D3D11 nor Vulkan frames%s\n",
                   avhwctx->type == AV_HWDEVICE_TYPE_CUDA
                       ? " (CUDA driver unavailable)" : "");
            mp_filter_internal_mark_failed(vf);
            return;
        }

        if (p->api.handle && !p->aji) {
            // mpv suboption values cannot contain '=', so the option takes a
            // plain lib dir and the env assignment is composed here.
            char *env = p->path.trtexec_libdir
                ? talloc_asprintf(p, "LD_LIBRARY_PATH=%s", p->path.trtexec_libdir)
                : NULL;
            aji_create_params cp = {
                .api_version = AJI_API_VERSION,
                .cuda_context = p->is_d3d11 ? NULL : p->cuda_ctx,
#if HAVE_D3D11
                .d3d11_device = p->is_d3d11 ? p->d3d_dev : NULL,
#endif
                .conf_path = p->path.conf,
                .model_dir = p->path.model_dir,
                .trtexec = p->path.trtexec,
                .trtexec_env = env,
                .slot = p->pending_slot,
                .rife_model_dir = p->path.rife_model_dir,
                .async_build = p->opts->stream_mode == 1 || p->opts->stream_mode == 3 ? 2 : 1,
                .engine_path = p->path.engine,
                .max_width = p->params.w,
                .max_height = p->params.h,
                .log = aji_log_bridge,
                .log_opaque = vf,
            };
#if HAVE_VULKAN
            if (p->is_vk && !p->api.create_vk) {
                MP_ERR(vf, "This inference shim takes no Vulkan frames (aji_create_vk): "
                       "use hwdec=d3d11va / nvdec, or update it\n");
                mp_filter_internal_mark_failed(vf);
                return;
            }
            p->aji = p->is_vk ? p->api.create_vk(&cp, &p->vkdev) : p->api.create(&cp);
#else
            p->aji = p->api.create(&cp);
#endif
            if (!p->aji) {
                MP_ERR(vf, "Failed to create inference context\n");
                mp_filter_internal_mark_failed(vf);
                return;
            }
        }

        if (p->aji) {
            enum AVPixelFormat sw = imgfmt2pixfmt(p->params.hw_subfmt);
            // x2bgr10 (packed 10-bit RGB) is what mpv hwuploads a 4:4:4 source
            // to on D3D11 (no planar 4:4:4 DXGI format); the DirectML backend
            // ingests it as RGB and round-trips it as RGB.
            p->aji_fmt = sw == AV_PIX_FMT_NV12 ? AJI_FMT_NV12 :
                         sw == AV_PIX_FMT_P010 ? AJI_FMT_P010 :
                         sw == AV_PIX_FMT_YUV444P16 ? AJI_FMT_YUV444P16 :
#ifdef AV_PIX_FMT_YUV444P10MSB
                         sw == AV_PIX_FMT_YUV444P10MSB ? AJI_FMT_YUV444P10MSB :
                         sw == AV_PIX_FMT_YUV444P12MSB ? AJI_FMT_YUV444P12MSB :
#endif
                         sw == AV_PIX_FMT_X2BGR10 ? AJI_FMT_RGB10A2 : 0;
            if (!p->aji_fmt) {
                MP_ERR(vf, "Unsupported sw format %s for inference\n",
                       mp_imgfmt_to_name(p->params.hw_subfmt));
                mp_filter_internal_mark_failed(vf);
                return;
            }
            // 4:4:4 planar ingest is TensorRT/CUDA only (no planar 16-bit
            // 4:4:4 DXGI format); on D3D11 it arrives as x2bgr10 RGB instead.
            if ((p->is_d3d11 || p->is_vk) && aji_format_is_444(p->aji_fmt)) {
                MP_ERR(vf, "planar 4:4:4 input requires the TensorRT/CUDA backend\n");
                mp_filter_internal_mark_failed(vf);
                return;
            }
            // RGB ingest is DirectML/D3D11 only (the CUDA path gets native
            // 4:4:4); reject it cleanly on CUDA rather than mis-decoding.
            if (!p->is_d3d11 && p->aji_fmt == AJI_FMT_RGB10A2) {
                MP_ERR(vf, "x2bgr10 input requires the DirectML backend\n");
                mp_filter_internal_mark_failed(vf);
                return;
            }
        }

#if HAVE_D3D11
        if (p->is_d3d11) {
            const DXGI_FORMAT fmt =
                p->aji_fmt == AJI_FMT_RGB10A2 ? DXGI_FORMAT_R10G10B10A2_UNORM :
                p->aji_fmt == AJI_FMT_P010    ? DXGI_FORMAT_P010 :
                                                DXGI_FORMAT_NV12;
            const int want = MPCLAMP(p->opts->queue_depth, 1, MAX_DEPTH);
            if (p->d3d_stage_count != want || p->d3d_stage_w != p->params.w ||
                p->d3d_stage_h != p->params.h || p->d3d_stage_fmt != fmt) {
                for (int i = 0; i < p->d3d_stage_count; i++)
                    ID3D11Texture2D_Release(p->d3d_stage[i]);
                p->d3d_stage_count = 0;
                p->d3d_stage_next = 0;
                D3D11_TEXTURE2D_DESC desc = {
                    .Width = p->params.w,
                    .Height = p->params.h,
                    .MipLevels = 1,
                    .ArraySize = 1,
                    .Format = fmt,
                    .SampleDesc = { .Count = 1 },
                    .Usage = D3D11_USAGE_DEFAULT,
                    .BindFlags = D3D11_BIND_SHADER_RESOURCE,
                    .MiscFlags = D3D11_RESOURCE_MISC_SHARED |
                                 D3D11_RESOURCE_MISC_SHARED_NTHANDLE,
                };
                for (int i = 0; i < want; i++) {
                    if (FAILED(ID3D11Device_CreateTexture2D(p->d3d_dev,
                                                            &desc, NULL,
                                                            &p->d3d_stage[i]))) {
                        MP_ERR(vf, "Failed to create the shared staging "
                                   "texture\n");
                        mp_filter_internal_mark_failed(vf);
                        return;
                    }
                    p->d3d_stage_count = i + 1;
                }
                p->d3d_stage_w = p->params.w;
                p->d3d_stage_h = p->params.h;
                p->d3d_stage_fmt = fmt;
            }
        }
#endif

        if (!configure_aji(vf)) {
            mp_filter_internal_mark_failed(vf);
            return;
        }
        p->configured = true;

        mp_image_setfmt(&p->layout, p->params.hw_subfmt);
        mp_image_set_size(&p->layout, p->params.w, p->params.h);
        MP_VERBOSE(vf, "Stream: %dx%d@%.3f subfmt=%s -> %dx%d\n", p->params.w,
                   p->params.h, p->fps, mp_imgfmt_to_name(p->params.hw_subfmt),
                   p->out_params.w, p->out_params.h);
    }

    // Runtime slot switch (vf-command): reconfigure on the filter thread.
    if (p->configured && p->aji && p->pending_slot != p->cur_slot) {
        av_buffer_unref(&p->hw_pool);
        if (!configure_aji(vf)) {
            mp_filter_internal_mark_failed(vf);
            return;
        }
    }

    // A background engine build finished: reconfigure, and the chain (or
    // RIFE) activates mid-playback. Until then frames pass through.
    if (p->configured && p->aji && p->api.poll(p->aji)) {
        MP_VERBOSE(vf, "background engine build finished; reconfiguring\n");
        av_buffer_unref(&p->hw_pool);
        if (!configure_aji(vf)) {
            mp_filter_internal_mark_failed(vf);
            return;
        }
    }

    // A preparation/check host polls the filter while it retains its first
    // input. No frame can escape before engines have been checked/prepared.
    if (p->opts->stream_mode >= 2 && p->configured)
        return;

    // Extra RIFE outputs first: they go straight to the out pin without
    // consuming input (the refqueue isn't touched while any are pending,
    // which also keeps EOF behind them).
    if (p->outq_n) {
        if (mp_pin_in_needs_data(vf->ppins[1])) {
            struct mp_image *img = p->outq[p->outq_pos];
            p->outq[p->outq_pos] = NULL;
            p->outq_pos++;
            if (--p->outq_n == 0)
                p->outq_pos = 0;
            mp_pin_in_write(vf->ppins[1], MAKE_FRAME(MP_FRAME_VIDEO, img));
            mp_filter_internal_mark_progress(vf);
        }
        return;
    }

    if (!mp_refqueue_can_output(p->queue))
        return;

    if (p->scene && p->rife_on && p->api.infer_rife_with_scene &&
        p->api.rife_scene_supported && p->api.rife_scene_supported(p->aji)) {
        // Analyze original decoded frames ahead of interpolation. GPU readback
        // and all addon/native work run on the optional worker, never here.
        for (int i = 0; i < 2; i++)
            ajn_scene_submit(p->scene, mp_refqueue_get(p->queue, i),
                             mp_refqueue_get(p->queue, i + 1));
    }

    if (p->rife_first) {
        // RIFE-first: interpolate the source pair, then upscale each emitted
        // frame. Same output-grid cadence as the default path below, but
        // rife_prev holds the previous *source* frame and the heavy upscaler
        // runs per emitted frame instead of RIFE at the output res.
        struct mp_image *src = mp_refqueue_get(p->queue, 0);
        if (!src)
            return;
        const double in_fps = src->nominal_fps;
        // The rife/upscale read this owned work-res source frame. With a
        // pre-RIFE downscale it's source -> work res via aji_resize; otherwise
        // (work == source res) it's a ref of the decoder frame on CUDA, or a
        // shareable staged copy on D3D11 (decoder textures aren't shareable to
        // the shim's D3D12 device).
        struct mp_image *cur_src;
        if (p->has_pre_resize)
            cur_src = downscale_src(vf, src);
        else
#if HAVE_D3D11
        if (p->is_d3d11)
            cur_src = stage_src_d3d11(vf, src);
        else
#endif
            cur_src = mp_image_new_ref(src);
        if (!cur_src) {
            mp_filter_internal_mark_failed(vf);
            return;
        }
        const int64_t cur_id = p->rife_next_id++;  // every new endpoint: a number of its own
        struct mp_image *list[8];
        int n = 0;
        bool failed = false;
        if (p->rife_prev && p->rife_prev->pts != MP_NOPTS_VALUE &&
            src->pts != MP_NOPTS_VALUE && src->pts > p->rife_prev->pts) {
            p->scene_decision = scene_pair(vf, p->rife_prev, cur_src);
            while (p->rife_acc <= p->rife_num && n < 8) {
                struct mp_image *frame;
                double pts;
                if (p->rife_acc == p->rife_num) {
                    frame = upscale_image(vf, cur_src);  // source grid point
                    pts = src->pts;
                } else {
                    double t = (double)p->rife_acc / p->rife_num;
                    struct mp_image *tmp =
                        interp_source(vf, p->rife_prev, p->rife_prev_id, cur_src, cur_id, t);
                    frame = tmp ? upscale_image(vf, tmp) : NULL;
                    mp_image_unrefp(&tmp);
                    pts = p->rife_prev->pts +
                          (src->pts - p->rife_prev->pts) * t;
                }
                if (!frame) {
                    failed = true;
                    break;
                }
                frame->pts = pts;
                if (in_fps > 0)
                    frame->nominal_fps = in_fps * p->rife_num / p->rife_den;
                MP_DBG(vf, "rife-first: %s pts=%f\n",
                       p->rife_acc == p->rife_num ? "source" : "interp", pts);
                list[n++] = frame;
                p->rife_acc += p->rife_den;
            }
            if (failed || !n) {
                for (int i = 0; i < n; i++)
                    mp_image_unrefp(&list[i]);
                n = 0;
            } else {
                p->rife_acc -= p->rife_num;  // rebase into the next pair
            }
        }
        if (!n) {
            // fresh chain (start/seek), invalid pts, or a failed interp: emit
            // the upscaled source as the grid origin and resync.
            p->rife_acc = p->rife_den;
            struct mp_image *u = upscale_image(vf, cur_src);
            if (!u) {
                mp_image_unrefp(&cur_src);
                mp_filter_internal_mark_failed(vf);
                return;
            }
            u->pts = src->pts;
            if (in_fps > 0)
                u->nominal_fps = in_fps * p->rife_num / p->rife_den;
            list[n++] = u;
        }
        mp_image_unrefp(&p->rife_prev);
        p->rife_prev = cur_src;  // previous *source* endpoint (owns the ref)
        p->rife_prev_id = cur_id;
        for (int i = 1; i < n; i++)
            p->outq[p->outq_n++] = list[i];
        p->outq_pos = 0;
        mp_refqueue_write_out_pin(p->queue, list[0]);
        return;
    }

    // Pipelined inference: submit the current frame plus the refqueue's
    // buffered future frames before waiting on the oldest, so the GPU
    // already works on frame N+1 (and N+2, ...) while N's wait blocks.
    // Stream/queue ordering in the shim makes overlapping submissions
    // safe; the refqueue window keeps each input alive until its
    // emission below.
    if (p->aji_active && !p->rife_on && p->depth > 1) {
        for (int rel = p->ring_n; p->ring_n < p->depth; rel++) {
            struct mp_image *src = mp_refqueue_get(p->queue, rel);
            if (!src)
                break;  // EOF tail or fewer futures buffered yet
            if (!submit_frame(vf, src, rel))
                break;  // degrade; the current frame falls back below
        }
    }

    struct mp_image *out;
    if (p->ring_n && p->ring[0].src == mp_refqueue_get(p->queue, 0)) {
        // also drains leftovers after a mid-flight reconfigure (their
        // outputs are valid frames of the pre-reconfigure chain)
        out = pop_ring(vf);
    } else {
        if (p->ring_n) {
            MP_WARN(vf, "pipeline lost sync; rendering synchronously\n");
            clear_ring(vf);
        }
        out = render(vf);
    }

    if (out && p->rife_on) {
        const int64_t out_id = p->rife_next_id++;  // every new endpoint: a number of its own
        if (out->nominal_fps > 0)
            out->nominal_fps = out->nominal_fps * p->rife_num / p->rife_den;
        struct mp_image *first = out;
        bool emitted_cur = true;
        if (p->rife_prev && p->rife_prev->pts != MP_NOPTS_VALUE &&
            out->pts != MP_NOPTS_VALUE && out->pts > p->rife_prev->pts) {
            p->scene_decision = scene_pair(vf, p->rife_prev, out);
            // Emit the output-grid points due within this input pair:
            // offsets acc, acc+den, ... (in 1/num units) while <= num.
            // An offset of exactly num is the right source frame itself;
            // with fractional factors some pairs end on an interpolation
            // and the source frame is never emitted.
            struct mp_image *list[8];
            int n = 0;
            bool failed = false;
            emitted_cur = false;
            while (p->rife_acc <= p->rife_num && n < 8) {
                if (p->rife_acc == p->rife_num) {
                    list[n++] = out;
                    emitted_cur = true;
                    MP_DBG(vf, "rife: source frame pts=%f\n", out->pts);
                } else {
                    double t = (double)p->rife_acc / p->rife_num;
                    struct mp_image *ip =
                        render_interp(vf, p->rife_prev, p->rife_prev_id, out, out_id, t);
                    if (!ip) {
                        failed = true;  // degrade to fewer frames
                        break;
                    }
                    ip->pts = p->rife_prev->pts +
                              (out->pts - p->rife_prev->pts) * t;
                    MP_DBG(vf, "rife: interp t=%f pts=%f\n", t, ip->pts);
                    list[n++] = ip;
                }
                p->rife_acc += p->rife_den;
            }
            if (failed || !n) {
                // resync the grid on the next pair, starting from `out`
                p->rife_acc = p->rife_den;
                if (!emitted_cur) {
                    // drop any interpolations made before the failure and
                    // emit the source frame so playback keeps moving
                    for (int i = 0; i < n; i++)
                        mp_image_unrefp(&list[i]);
                    n = 0;
                    emitted_cur = true;
                }
            } else {
                p->rife_acc -= p->rife_num;  // rebase into the next pair
            }
            if (n) {
                first = list[0];
                for (int i = 1; i < n; i++)
                    p->outq[p->outq_n++] = list[i];
                p->outq_pos = 0;
            }
        } else {
            // fresh pair chain (start of stream or after a seek): `out`
            // is the grid origin and the next point lies den/num in
            p->rife_acc = p->rife_den;
        }
        mp_image_unrefp(&p->rife_prev);
        p->rife_prev = mp_image_new_ref(out);
        p->rife_prev_id = out_id;
        if (!emitted_cur)
            mp_image_unrefp(&out);  // endpoint only; rife_prev keeps it
        p->seq0++;
        mp_refqueue_write_out_pin(p->queue, first);
        return;
    }

    p->seq0++;

    mp_refqueue_write_out_pin(p->queue, out);
    if (p->held_on && mp_time_sec() - p->stats_time >= 1.0)

        write_stats(vf);
}

static bool vf_animejanai_command(struct mp_filter *vf,
                                  struct mp_filter_command *cmd)
{
    struct priv *p = vf->priv;
    if (cmd->type != MP_FILTER_COMMAND_TEXT || !cmd->cmd)
        return false;
    if (strcmp(cmd->cmd, "scene-map") == 0 && cmd->arg) {
        ajn_scene_destroy(p->scene); p->scene = NULL;
        p->scene_builtin = false;
        p->scene_decision = -1;
        bool ok = true;
        if (*cmd->arg) {
            p->scene = ajn_scene_create(vf->log);
            ok = ajn_scene_connect(p->scene, cmd->arg);
            if (!ok) { ajn_scene_destroy(p->scene); p->scene = NULL; }
        }
        configure_scene(vf, p->rife_on);
        update_depth(vf);
        return ok;
    }
    if (strcmp(cmd->cmd, "poll") == 0) {
        // No-op wakeup: while playback is paused no frames flow, so
        // process() (and with it the background-build completion poll)
        // never runs. The engine-build monitor script kicks this so a
        // finished build is noticed and playback can auto-resume.
        mp_filter_wakeup(vf);
        return true;
    }
    if (strcmp(cmd->cmd, "slot") == 0 && cmd->arg) {
        int slot = atoi(cmd->arg);
        // slot 0 = bypass: the filter stays in the chain but passes frames
        // through, so on/off toggling never rebuilds the chain (a rebuild
        // would race with vf-commands sent right after a profile change).
        if (slot < 0 || (slot == 0 && strcmp(cmd->arg, "0") != 0)) {
            MP_ERR(vf, "Invalid slot '%s'\n", cmd->arg);
            return false;
        }
        p->pending_slot = slot;
        mp_filter_wakeup(vf);
        return true;
    }
    return false;
}

static void uninit(struct mp_filter *vf)
{
    struct priv *p = vf->priv;

    flush_frames(vf);
    ajn_scene_destroy(p->scene);
    p->scene = NULL;
    talloc_free(p->queue);
    held_destroy(vf);
    if (p->aji)
        p->api.destroy(&p->aji);
    if (p->api.handle)
        aji_lib_close(p->api.handle);
    if ((p->own_stream || p->diff_stream) && p->cu &&
        p->cu->cuCtxPushCurrent(p->cuda_ctx) == CUDA_SUCCESS) {
        CUcontext dummy;
        for (int i = 0; i < MAX_DEPTH; i++)
            if (p->decode_evt[i])
                p->cu->cuEventDestroy(p->decode_evt[i]);
        if (p->own_stream)
            p->cu->cuStreamDestroy(p->own_stream);
        held_destroy_cuda(vf);
        p->cu->cuCtxPopCurrent(&dummy);
    }
#if HAVE_D3D11
    for (int i = 0; i < p->d3d_stage_count; i++)
        ID3D11Texture2D_Release(p->d3d_stage[i]);
#endif
    av_buffer_unref(&p->hw_pool);
    av_buffer_unref(&p->src_pool);
    av_buffer_unref(&p->work_pool);
    av_buffer_unref(&p->av_device_ref);
    if (p->cu)
        cuda_free_functions(&p->cu);
}

static const struct mp_filter_info vf_animejanai_filter = {
    .name = "animejanai",
    .process = vf_animejanai_process,
    .command = vf_animejanai_command,
    .reset = flush_frames,
    .destroy = uninit,
    .priv_size = sizeof(struct priv),
};

// During hr-seek, frames before the target would be upscaled and then
// discarded by the post-filter framedrop; skip inference on them entirely
// (same approach as the vf_vapoursynth fast-seek patch).
static bool drop_pre_seek_target(void *ctx, struct mp_image *mpi)
{
    struct mp_filter *vf = ctx;
    struct priv *p = vf->priv;

    if (!p->opts->skip_seek_pre_target || mpi->pts == MP_NOPTS_VALUE)
        return false;
    struct mp_stream_info *info = mp_filter_find_stream_info(vf);
    double target;
    if (!(info && info->get_hrseek && info->get_hrseek(info, &target)))
        return false;
    if (mpi->pts >= target - 0.005)
        return false;
    MP_DBG(vf, "drop pre-seek-target frame pts=%f target=%f\n", mpi->pts,
           target);
    return true;
}

static struct mp_filter *vf_animejanai_create(struct mp_filter *parent,
                                              void *options)
{
    struct mp_filter *f = mp_filter_create(parent, &vf_animejanai_filter);
    if (!f) {
        talloc_free(options);
        return NULL;
    }

    mp_filter_add_pin(f, MP_PIN_IN, "in");
    mp_filter_add_pin(f, MP_PIN_OUT, "out");

    struct priv *p = f->priv;
    p->opts = talloc_steal(p, options);
    f->failure_is_fatal = p->opts->stream_mode != 0;
    p->queue = mp_refqueue_alloc(f);
    p->cur_slot = p->pending_slot = p->opts->slot;
    p->held_last_pts = MP_NOPTS_VALUE;

    // Expand mpv path shortcuts (~~/ etc.) so portable configs can point at
    // files relative to the config directory (suboptions are not expanded
    // by the option parser itself). Empty options stay NULL.
    struct { char **dst; char *src; } path_opts[] = {
        {&p->path.conf, p->opts->conf},
        {&p->path.model_dir, p->opts->model_dir},
        {&p->path.trtexec, p->opts->trtexec},
        {&p->path.trtexec_libdir, p->opts->trtexec_libdir},
        {&p->path.stats, p->opts->stats},
        {&p->path.engine, p->opts->engine},
        {&p->path.lib, p->opts->lib},
        {&p->path.rife_model_dir, p->opts->rife_model_dir},
    };
    for (int i = 0; i < MP_ARRAY_SIZE(path_opts); i++) {
        if (path_opts[i].src && path_opts[i].src[0])
            *path_opts[i].dst = mp_normalize_user_path(p, f->global,
                                                       path_opts[i].src);
    }

    // Not fatal: on non-NVIDIA machines the D3D11/DirectML path carries
    // the filter; CUDA frames are rejected at reinit instead.
    if (cuda_load_functions(&p->cu, NULL) < 0) {
        p->cu = NULL;
        MP_VERBOSE(f, "CUDA driver API unavailable (D3D11 input only)\n");
    }

    bool want_aji = p->path.engine || p->path.conf;
    if (want_aji) {
        // Strict C ABI boundary: the inference backend (TensorRT) is loaded
        // at runtime and never linked.
        const char *lib = p->path.lib ? p->path.lib : AJI_DEFAULT_LIB;
        p->api.handle = aji_lib_open(lib);
        if (!p->api.handle) {
            MP_ERR(f, "Failed to load inference shim '%s': %s\n", lib,
                   aji_lib_error());
            goto fail;
        }
        p->api.create = aji_lib_sym(p->api.handle, "aji_create");
        p->api.create_vk = aji_lib_sym(p->api.handle, "aji_create_vk");
        p->api.set_slot = aji_lib_sym(p->api.handle, "aji_set_slot");
        p->api.configure = aji_lib_sym(p->api.handle, "aji_configure");
        p->api.infer = aji_lib_sym(p->api.handle, "aji_infer");
        p->api.flush = aji_lib_sym(p->api.handle, "aji_flush");
        p->api.done = aji_lib_sym(p->api.handle, "aji_done");
        p->api.wait = aji_lib_sym(p->api.handle, "aji_wait");
        p->api.rife_factor = aji_lib_sym(p->api.handle, "aji_rife_factor");
        // Optional (added without an API_VERSION bump); NULL on older shims,
        // which then run the default upscale-then-RIFE order.
        p->api.rife_before_upscale =
            aji_lib_sym(p->api.handle, "aji_rife_before_upscale");
        // Optional pre-RIFE downscale (rife-first). NULL on older shims, which
        // then run RIFE at the source resolution (no pre-resize).
        p->api.pre_resize = aji_lib_sym(p->api.handle, "aji_pre_resize");
        p->api.resize = aji_lib_sym(p->api.handle, "aji_resize");
        p->api.copy_frame = aji_lib_sym(p->api.handle, "aji_copy_frame");
        p->api.infer_rife = aji_lib_sym(p->api.handle, "aji_infer_rife");
        p->api.rife_scene_supported = aji_lib_sym(p->api.handle, "aji_rife_scene_supported");
        p->api.rife_enhanced_scene = aji_lib_sym(p->api.handle, "aji_rife_enhanced_scene");
        p->api.infer_rife_with_scene = aji_lib_sym(p->api.handle, "aji_infer_rife_with_scene");
        p->api.infer_rife_ids = aji_lib_sym(p->api.handle, "aji_infer_rife_ids");
        p->api.poll = aji_lib_sym(p->api.handle, "aji_poll");
        p->api.temporal_radius =
            aji_lib_sym(p->api.handle, "aji_temporal_radius");
        p->api.ingest = aji_lib_sym(p->api.handle, "aji_ingest");
        p->api.infer_seq = aji_lib_sym(p->api.handle, "aji_infer_seq");
        p->api.temporal_reset =
            aji_lib_sym(p->api.handle, "aji_temporal_reset");
        p->api.current_log = aji_lib_sym(p->api.handle, "aji_current_log");
        if (p->opts->stream_mode) {
            int (*version)(void) = aji_lib_sym(p->api.handle, "aji_stream_policy_version");
            int (*configured_version)(const char *) =
                aji_lib_sym(p->api.handle, "aji_stream_policy_version_for_config");
            int policy = configured_version ? configured_version(p->path.conf) : version ? version() : 0;
            if (policy != AJN_STREAM_VERSION) {
                MP_ERR(f, "AJN_STREAM_POLICY_UNAVAILABLE\n");
                goto fail;
            }
        }
        p->api.last_error = aji_lib_sym(p->api.handle, "aji_last_error");
        p->api.destroy = aji_lib_sym(p->api.handle, "aji_destroy");
        // Held-frame reuse (libaji 0.30): all or nothing, never required.
        p->api.reuse_for_config =
            aji_lib_sym(p->api.handle, "aji_reuse_held_frames_for_config");
        p->api.held_create = aji_lib_sym(p->api.handle, "aji_held_create");
        p->api.held_set_trace = aji_lib_sym(p->api.handle, "aji_held_set_trace");
        p->api.held_decide = aji_lib_sym(p->api.handle, "aji_held_decide");
        p->api.held_set_anchor =
            aji_lib_sym(p->api.handle, "aji_held_set_anchor");
        p->api.held_reset = aji_lib_sym(p->api.handle, "aji_held_reset");
        p->api.held_get_stats = aji_lib_sym(p->api.handle, "aji_held_get_stats");
        p->api.held_destroy = aji_lib_sym(p->api.handle, "aji_held_destroy");
        if (!p->api.reuse_for_config || !p->api.held_create ||
            !p->api.held_decide || !p->api.held_set_anchor ||
            !p->api.held_reset || !p->api.held_get_stats ||
            !p->api.held_destroy) {
            p->api.reuse_for_config = NULL;
            p->api.held_create = NULL;
        }
        if (!p->api.create || !p->api.set_slot || !p->api.configure ||
            !p->api.infer || !p->api.flush || !p->api.done || !p->api.wait ||
            !p->api.rife_factor || !p->api.infer_rife ||
            !p->api.poll || !p->api.temporal_radius || !p->api.ingest ||
            !p->api.infer_seq || !p->api.temporal_reset ||
            !p->api.current_log || !p->api.last_error || !p->api.destroy) {
            MP_ERR(f, "Inference shim '%s' is missing aji_* symbols "
                      "(ABI version mismatch?)\n", lib);
            goto fail;
        }
        MP_VERBOSE(f, "Loaded inference shim: %s\n", lib);
    }

    // The CUDA device is adopted from the first input frame's
    // AVHWFramesContext (see process()), so the filter works regardless of
    // whether the VO provides a hwdec device (e.g. with --vo=null).

    // Register the exact surface sub-formats the engine can ingest (mirrors the
    // whitelist in process() and the aji_trt/aji_dml input guards). Passing the
    // real sub-formats - instead of 0 ("any") - lets autoconvert constrain its
    // hw-upload target to one of these, so a software-decoded/planar source is
    // converted to an ingestible format before upload instead of arriving in a
    // format the filter would reject and disable itself over. Backend gating:
    // yuv444p16 is TensorRT/CUDA-only; x2bgr10 (packed 10-bit RGB) is
    // DirectML/D3D11-only.
    mp_refqueue_add_in_format(p->queue, IMGFMT_CUDA, pixfmt2imgfmt(AV_PIX_FMT_NV12));
    mp_refqueue_add_in_format(p->queue, IMGFMT_CUDA, pixfmt2imgfmt(AV_PIX_FMT_P010));
    mp_refqueue_add_in_format(p->queue, IMGFMT_CUDA, pixfmt2imgfmt(AV_PIX_FMT_YUV444P16));
#ifdef AV_PIX_FMT_YUV444P10MSB
    // NVDEC's MSB-aligned 4:4:4 output; older FFmpeg (e.g. the Linux bundle's
    // n7.1) has neither the formats nor a decoder that produces them
    mp_refqueue_add_in_format(p->queue, IMGFMT_CUDA, pixfmt2imgfmt(AV_PIX_FMT_YUV444P10MSB));
    mp_refqueue_add_in_format(p->queue, IMGFMT_CUDA, pixfmt2imgfmt(AV_PIX_FMT_YUV444P12MSB));
#endif
#if HAVE_VULKAN
    mp_refqueue_add_in_format(p->queue, IMGFMT_VULKAN, pixfmt2imgfmt(AV_PIX_FMT_NV12));
    mp_refqueue_add_in_format(p->queue, IMGFMT_VULKAN, pixfmt2imgfmt(AV_PIX_FMT_P010));
#endif
#if HAVE_D3D11
    mp_refqueue_add_in_format(p->queue, IMGFMT_D3D11, pixfmt2imgfmt(AV_PIX_FMT_NV12));
    mp_refqueue_add_in_format(p->queue, IMGFMT_D3D11, pixfmt2imgfmt(AV_PIX_FMT_P010));
    mp_refqueue_add_in_format(p->queue, IMGFMT_D3D11, pixfmt2imgfmt(AV_PIX_FMT_X2BGR10));
#endif
    mp_refqueue_set_refs(p->queue, 0, 0);
    mp_refqueue_set_mode(p->queue, 0);
    mp_refqueue_set_drop_check(p->queue, drop_pre_seek_target, f);

    return f;

fail:
    talloc_free(f);
    return NULL;
}

#define OPT_BASE_STRUCT struct opts
static const m_option_t vf_opts_fields[] = {
    {"conf", OPT_STRING(conf), .flags = M_OPT_FILE},
    {"model-dir", OPT_STRING(model_dir), .flags = M_OPT_FILE},
    {"trtexec", OPT_STRING(trtexec), .flags = M_OPT_FILE},
    {"trtexec-libdir", OPT_STRING(trtexec_libdir), .flags = M_OPT_FILE},
    {"slot", OPT_INT(slot), M_RANGE(0, 9999)},
    {"stats", OPT_STRING(stats), .flags = M_OPT_FILE},
    {"engine", OPT_STRING(engine), .flags = M_OPT_FILE},
    {"lib", OPT_STRING(lib), .flags = M_OPT_FILE},
    {"rife-model-dir", OPT_STRING(rife_model_dir), .flags = M_OPT_FILE},
    {"passthrough", OPT_BOOL(passthrough)},
    {"skip-seek-pre-target", OPT_BOOL(skip_seek_pre_target)},
    {"output-444", OPT_BOOL(output_444)},
    {"queue-depth", OPT_INT(queue_depth), M_RANGE(1, MAX_DEPTH)},
    {"stream-mode", OPT_CHOICE(stream_mode, {"normal", 0}, {"required", 1}, {"prepare", 2}, {"check", 3})},
    {"reuse-held", OPT_CHOICE(reuse_held, {"auto", 0}, {"no", 1})},

    {0}
};

const struct mp_user_filter_entry vf_animejanai = {
    .desc = {
        .description = "AnimeJaNai AI upscaling filter (CUDA/D3D11)",
        .name = "animejanai",
        .priv_size = sizeof(OPT_BASE_STRUCT),
        .priv_defaults = &(const OPT_BASE_STRUCT) {
            .passthrough = true,
            .slot = 1,
            .skip_seek_pre_target = true,
            .output_444 = true,
            .queue_depth = 3,
        },
        .options = vf_opts_fields,
    },
    .create = vf_animejanai_create,
};
