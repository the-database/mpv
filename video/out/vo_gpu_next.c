/*
 * Copyright (C) 2021 Niklas Haas
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

#include <sys/stat.h>
#include <time.h>

#include <libplacebo/colorspace.h>
#include <libplacebo/options.h>
#include <libplacebo/renderer.h>
#include <libplacebo/shaders/lut.h>
#include <libplacebo/shaders/icc.h>
#include <libplacebo/shaders/custom.h>
#include <libplacebo/utils/libav.h>
#include <libplacebo/utils/frame_queue.h>

#include "config.h"
#include "common/common.h"
#include "common/stats.h"
#include "misc/io_utils.h"
#include "options/m_config.h"
#include "options/options.h"
#include "options/path.h"
#include "osdep/io.h"
#include "osdep/threads.h"
#include "stream/stream.h"
#include "sub/draw_bmp.h"
#include "video/fmt-conversion.h"
#include "video/mp_image.h"
#include "video/out/placebo/ra_pl.h"
#include "placebo/utils.h"
#include "gpu/context.h"
#include "gpu/hwdec.h"
#include "gpu/utils.h"
#include "gpu/video.h"
#include "gpu/video_shaders.h"
#include "sub/osd.h"
#include "gpu_next/context.h"

#if HAVE_GL && defined(PL_HAVE_OPENGL)
#include <libplacebo/opengl.h>
#include "video/out/opengl/ra_gl.h"
#endif

#if HAVE_D3D11 && defined(PL_HAVE_D3D11)
#include <libplacebo/d3d11.h>
#include "video/out/d3d11/ra_d3d11.h"
#include "osdep/windows_utils.h"
#endif


struct osd_entry {
    pl_tex tex;
    pl_tex blur_tex, tmp_tex; // deferred-blur scratch (see osd_blur_part)
    struct pl_overlay_part *parts;
    int num_parts;
    pl_tex result_tex;        // deferred-composite result atlas (compose_glyph_runs)
    struct pl_overlay_part *run_parts;
    int num_run_parts;
};

// Ring of streaming upload buffers cycled across uploads, so a buffer is never
// reused while its previous async upload is still in flight (which would make
// pl_buf_write block). Sized well above the in-flight depth.
#define NUM_OVERLAY_BUFS 16

struct osd_state {
    struct osd_entry entries[MAX_OSD_PARTS];
    struct pl_overlay overlays[MAX_OSD_PARTS];
};

struct scaler_params {
    struct pl_filter_config config;
};

struct user_hook {
    char *path;
    const struct pl_hook *hook;
};

struct user_lut {
    char *opt;
    char *path;
    int type;
    struct pl_custom_lut *lut;
};

struct frame_info {
    int count;
    struct pl_dispatch_info info[VO_PASS_PERF_MAX];
};

struct cache {
    struct mp_log *log;
    struct mpv_global *global;
    char *dir;
    const char *name;
    size_t size_limit;
    pl_cache cache;
};

struct priv {
    struct mp_log *log;
    struct mpv_global *global;
    struct stats_ctx *stats;
    struct ra_ctx *ra_ctx;
    struct gpu_ctx *context;
    struct ra_hwdec_ctx hwdec_ctx;
    struct ra_hwdec_mapper *hwdec_mapper;
    struct timer_pool *hwdec_timer;
    struct mp_pass_perf hwdec_perf;
    struct timer_pool *sw_upload_timer;
    struct mp_pass_perf sw_upload_perf;

    // Allocated DR buffers
    mp_mutex dr_lock;
    pl_buf *dr_buffers;
    int num_dr_buffers;

    pl_log pllog;
    pl_gpu gpu;
    pl_renderer rr;
    pl_dispatch osd_dp; // for the deferred-blur compute passes
    bool osd_blur_unsupported; // logged once if the format can't be blurred
    pl_queue queue;
    pl_swapchain sw;
    pl_fmt osd_fmt[SUBBITMAP_COUNT];
    pl_tex *sub_tex;
    int num_sub_tex;
    pl_buf overlay_bufs[NUM_OVERLAY_BUFS]; // ring; see NUM_OVERLAY_BUFS
    unsigned overlay_buf_idx;
    pl_tex *sub_scratch;       // recycled blur scratch textures (blur_tex/tmp_tex)
    int num_sub_scratch;

    // Deferred composite (SUBBITMAP_LIBASS_GLYPHS): r32f combine accumulator
    // format + reusable scratch. The composited per-run coverage goes into the
    // per-entry result atlas (entry->result_tex).
    pl_fmt osd_acc_fmt;        // r32f (storable) for the (legacy) per-region accumulator
    pl_tex run_acc;            // r32f combine accumulator (sized to max run)
    pl_tex run_tmp;            // r8 blur temp
    pl_tex run_cov_f, run_cov_b; // r8 per-run fill/border coverage (pre-copy)

    // Persistent GPU glyph cache (Stage B): each unique glyph (keyed by libass
    // cache_id) is uploaded once into glyph_atlas, so the per-frame upload is
    // just the cache misses instead of the whole packed atlas.
    pl_tex glyph_atlas;
    struct gcache_slot { uint64_t id; int ax, ay, w, h; } *gcache;
    int gcache_cap, gcache_count;            // open-addressed table (cap = pow2)
    int gatlas_w, gatlas_h, gsx, gsy, growh; // skyline allocator cursor
    pl_buf glyph_stage[3];                   // async upload staging ring (no VO stall)
    unsigned glyph_stage_idx;
    uint8_t *gstage_cpu; size_t gstage_cpu_sz; // tight-repack scratch for misses
    pl_tex edge_tex;                         // outline mode: rgba32f edge list (x0,y0,x1,y1)
    float *ebuf; int ebuf_cap;               // CPU edge scratch (in vec4 units)
    pl_tex hdr_tex;                          // per-(glyph,Y-band) (offset,count) into edge_tex
    float *hbuf; int hbuf_cap;               // CPU header scratch (in vec4 units)
    int *bscratch; int bscratch_cap;         // per-glyph band counting-sort scratch
    pl_tex work_tex;                         // batched raster: one 16x16 tile per workgroup
    float *wbuf; int wbuf_cap;               // CPU work-list scratch (in vec4 units)
    pl_tex result_acc;                       // r32f atlas-wide combine accumulator
    pl_tex result_tmp;                       // r8 atlas-wide blur intermediate
    pl_tex blurwork_tex;                     // batched blur: one 16x16 tile per workgroup
    float *bwbuf; int bwbuf_cap;             // CPU blur work-list scratch (vec4 units)
    pl_tex combwork_tex;                     // batched single-part combine work-list
    float *cwbuf; int cwbuf_cap;             // CPU combine work-list scratch

    struct mp_rect src, dst;
    struct mp_osd_res osd_res;
    struct osd_state osd_state;

    uint64_t last_id;
    uint64_t osd_sync;
    double last_pts;
    bool is_interpolated;
    bool want_reset;
    bool flush_cache;
    bool frame_pending;
    bool paused;

    pl_options pars;
    struct m_config_cache *opts_cache;
    struct m_config_cache *next_opts_cache;
    struct gl_next_opts *next_opts;
    struct cache shader_cache, icc_cache;
    struct mp_csp_equalizer_state *video_eq;
    struct scaler_params scalers[SCALER_COUNT];
    const struct pl_hook **hooks; // storage for `params.hooks`
    enum pl_color_levels output_levels;

    struct pl_icc_params icc_params;
    char *icc_path;
    pl_icc_object icc_profile;

    // Cached shaders, preserved across options updates
    struct user_hook *user_hooks;
    int num_user_hooks;

    // Performance data of last frame
    struct frame_info perf_fresh;
    struct frame_info perf_redraw;

    struct mp_image_params target_params;
};

static void update_render_options(struct vo *vo);
static void update_lut(struct priv *p, struct user_lut *lut);

struct gl_next_opts {
    bool delayed_peak;
    int sub_hdr_peak;
    int image_subs_hdr_peak;
    int border_background;
    float background_blur_radius;
    float corner_rounding;
    bool inter_preserve;
    struct user_lut lut;
    struct user_lut image_lut;
    struct user_lut target_lut;
    int target_hint;
    int target_hint_mode;
    bool target_hint_strict;
    char **raw_opts;
};

const struct m_opt_choice_alternatives lut_types[] = {
    {"auto",        PL_LUT_UNKNOWN},
    {"native",      PL_LUT_NATIVE},
    {"normalized",  PL_LUT_NORMALIZED},
    {"conversion",  PL_LUT_CONVERSION},
    {0}
};

#define OPT_BASE_STRUCT struct gl_next_opts
const struct m_sub_options gl_next_conf = {
    .opts = (const struct m_option[]) {
        {"sub-hdr-peak", OPT_CHOICE(sub_hdr_peak, {"sdr", PL_COLOR_SDR_WHITE}),
            M_RANGE(10, 10000)},
        {"image-subs-hdr-peak", OPT_CHOICE(image_subs_hdr_peak, {"sdr", PL_COLOR_SDR_WHITE},
            {"video", -1}, {"video-static", -2}, {"video-dynamic", -3}),  M_RANGE(10, 10000)},
        {"allow-delayed-peak-detect", OPT_BOOL(delayed_peak)},
        {"border-background", OPT_CHOICE(border_background,
            {"none",  BACKGROUND_NONE},
            {"color", BACKGROUND_COLOR},
            {"tiles", BACKGROUND_TILES}
            ,{"blur", BACKGROUND_BLUR})},
        {"background-blur-radius", OPT_FLOAT(background_blur_radius)},
        {"corner-rounding", OPT_FLOAT(corner_rounding), M_RANGE(0, 1)},
        {"interpolation-preserve", OPT_BOOL(inter_preserve)},
        {"lut", OPT_STRING(lut.opt), .flags = M_OPT_FILE},
        {"lut-type", OPT_CHOICE_C(lut.type, lut_types)},
        {"image-lut", OPT_STRING(image_lut.opt), .flags = M_OPT_FILE},
        {"image-lut-type", OPT_CHOICE_C(image_lut.type, lut_types)},
        {"target-lut", OPT_STRING(target_lut.opt), .flags = M_OPT_FILE},
        {"target-colorspace-hint", OPT_CHOICE(target_hint, {"auto", -1}, {"no", 0}, {"yes", 1})},
        {"target-colorspace-hint-mode", OPT_CHOICE(target_hint_mode, {"target", 0}, {"source", 1}, {"source-dynamic", 2})},
        {"target-colorspace-hint-strict", OPT_BOOL(target_hint_strict)},
        // No `target-lut-type` because we don't support non-RGB targets
        {"libplacebo-opts", OPT_KEYVALUELIST(raw_opts)},
        {0},
    },
    .defaults = &(struct gl_next_opts) {
        .border_background = BACKGROUND_COLOR,
        .background_blur_radius = 16.0f,
        .inter_preserve = true,
        .sub_hdr_peak = PL_COLOR_SDR_WHITE,
        .image_subs_hdr_peak = 1000,
        .target_hint = -1,
        .target_hint_strict = true,
    },
    .size = sizeof(struct gl_next_opts),
    .change_flags = UPDATE_VIDEO,
};

static pl_buf get_dr_buf(struct priv *p, const uint8_t *ptr)
{
    mp_mutex_lock(&p->dr_lock);

    for (int i = 0; i < p->num_dr_buffers; i++) {
        pl_buf buf = p->dr_buffers[i];
        if (ptr >= buf->data && ptr < buf->data + buf->params.size) {
            mp_mutex_unlock(&p->dr_lock);
            return buf;
        }
    }

    mp_mutex_unlock(&p->dr_lock);
    return NULL;
}

static void free_dr_buf(void *opaque, uint8_t *data)
{
    struct priv *p = opaque;
    mp_mutex_lock(&p->dr_lock);

    for (int i = 0; i < p->num_dr_buffers; i++) {
        if (p->dr_buffers[i]->data == data) {
            pl_buf_destroy(p->gpu, &p->dr_buffers[i]);
            MP_TARRAY_REMOVE_AT(p->dr_buffers, p->num_dr_buffers, i);
            mp_mutex_unlock(&p->dr_lock);
            return;
        }
    }

    MP_ASSERT_UNREACHABLE();
}

static struct mp_image *get_image(struct vo *vo, int imgfmt, int w, int h,
                                  int stride_align, int flags)
{
    struct priv *p = vo->priv;
    pl_gpu gpu = p->gpu;
    if (!gpu->limits.thread_safe || !gpu->limits.max_mapped_size)
        return NULL;

    if ((flags & VO_DR_FLAG_HOST_CACHED) && !gpu->limits.host_cached)
        return NULL;

    stride_align = mp_lcm(stride_align, gpu->limits.align_tex_xfer_pitch);
    stride_align = mp_lcm(stride_align, gpu->limits.align_tex_xfer_offset);
    int size = mp_image_get_alloc_size(imgfmt, w, h, stride_align);
    if (size < 0)
        return NULL;

    pl_buf buf = pl_buf_create(gpu, &(struct pl_buf_params) {
        .memory_type = PL_BUF_MEM_HOST,
        .host_mapped = true,
        .size = size + stride_align,
    });

    if (!buf)
        return NULL;

    struct mp_image *mpi = mp_image_from_buffer(imgfmt, w, h, stride_align,
                                                buf->data, buf->params.size,
                                                p, free_dr_buf);
    if (!mpi) {
        pl_buf_destroy(gpu, &buf);
        return NULL;
    }

    mp_mutex_lock(&p->dr_lock);
    MP_TARRAY_APPEND(p, p->dr_buffers, p->num_dr_buffers, buf);
    mp_mutex_unlock(&p->dr_lock);

    return mpi;
}

// Separable gaussian over one atlas sub-region [ox,oy,rw,rh], clamped to that
// region (so a blurred part can't read/write its atlas neighbours). sigma == 0
// degenerates to a copy. Validated against a CPU reference via lavapipe.
// Separable gaussian with strided sampling: taps are spaced `stride` px apart
// so the work is O(1) in the radius (bounded tap count), not O(radius). For wide
// blurs the gaussian is smooth enough that ~stride<=sigma/2 spacing reconstructs
// it accurately; small blurs use stride==1 (exact).
static const char *const osd_blur_body_h =
    "ivec2 g = ivec2(gl_GlobalInvocationID.xy);\n"
    "if (g.x < rw && g.y < rh) {\n"
    "    int px = g.x + ox, py = g.y + oy;\n"
    "    float acc = 0.0, wsum = 0.0;\n"
    "    for (int t = -ntaps; t <= ntaps; t++) {\n"
    "        int d = t * stride;\n"
    "        int q = px + d;\n"
    "        if (q >= ox && q < ox+rw) {\n"
    "            float w = sigma > 0.0 ? exp(-0.5*float(d*d)/(sigma*sigma)) : float(d==0);\n"
    "            acc += w * texelFetch(src, ivec2(q, py), 0).r; wsum += w;\n"
    "        }\n"
    "    }\n"
    "    imageStore(dst, ivec2(px, py), vec4(acc / wsum, 0.0, 0.0, 0.0));\n"
    "}\n";
static const char *const osd_blur_body_v =
    "ivec2 g = ivec2(gl_GlobalInvocationID.xy);\n"
    "if (g.x < rw && g.y < rh) {\n"
    "    int px = g.x + ox, py = g.y + oy;\n"
    "    float acc = 0.0, wsum = 0.0;\n"
    "    for (int t = -ntaps; t <= ntaps; t++) {\n"
    "        int d = t * stride;\n"
    "        int q = py + d;\n"
    "        if (q >= oy && q < oy+rh) {\n"
    "            float w = sigma > 0.0 ? exp(-0.5*float(d*d)/(sigma*sigma)) : float(d==0);\n"
    "            acc += w * texelFetch(src, ivec2(px, q), 0).r; wsum += w;\n"
    "        }\n"
    "    }\n"
    "    imageStore(dst, ivec2(px, py), vec4(acc / wsum, 0.0, 0.0, 0.0));\n"
    "}\n";

static void osd_blur_part(struct priv *p, pl_tex src, pl_tex dst,
                          int ox, int oy, int rw, int rh, float sigma,
                          const char *body)
{
    int radius = (int)(3.0f * sigma + 0.999f); // ~ceil(3*sigma); 0 when sigma==0
    // Bound the tap count (~17 max) by striding wide kernels; this makes the
    // blur O(1) in radius. Spacing stays <= sigma/2 so the gaussian is well
    // sampled. Small kernels keep stride 1 (identical to a dense gaussian).
    int stride = (radius + 7) / 8;
    if (stride < 1) stride = 1;
    int ntaps = (radius + stride - 1) / stride;
    pl_shader sh = pl_dispatch_begin(p->osd_dp);
    struct pl_shader_desc descs[] = {
        { .desc = { .name="src", .type=PL_DESC_SAMPLED_TEX, .binding=0 },
          .binding = { .object=src } },
        { .desc = { .name="dst", .type=PL_DESC_STORAGE_IMG, .binding=1,
                    .access=PL_DESC_ACCESS_WRITEONLY },
          .binding = { .object=dst } },
    };
    struct pl_shader_var vars[] = {
        { .var = pl_var_int("ox"),     .data=&ox },
        { .var = pl_var_int("oy"),     .data=&oy },
        { .var = pl_var_int("rw"),     .data=&rw },
        { .var = pl_var_int("rh"),     .data=&rh },
        { .var = pl_var_int("stride"), .data=&stride },
        { .var = pl_var_int("ntaps"),  .data=&ntaps },
        { .var = pl_var_float("sigma"),.data=&sigma },
    };
    struct pl_custom_shader cs = {
        .input = PL_SHADER_SIG_NONE, .output = PL_SHADER_SIG_NONE,
        .compute = true, .compute_group_size = {16, 16},
        .descriptors = descs, .num_descriptors = 2,
        .variables = vars, .num_variables = 7,
        .body = body,
    };
    if (pl_shader_custom(sh, &cs)) {
        pl_dispatch_compute(p->osd_dp, pl_dispatch_compute_params(
            .shader = &sh, .dispatch_size = { (rw+15)/16, (rh+15)/16, 1 }));
    } else {
        pl_dispatch_abort(p->osd_dp, &sh);
    }
}

// ---- Deferred composite (SUBBITMAP_LIBASS_GLYPHS) GPU passes ----------------
// libass emits uncombined per-glyph coverage; we reproduce its per-run pipeline
// on the GPU: combine (saturating add) -> blur -> fix_outline -> alpha-over,
// integer-exact, so the result matches the CPU-combined path bit for bit.

// Add one glyph's atlas coverage into the run's f32 accumulator. Read-add-write;
// pl_dispatch serializes overlapping parts via its texture-dependency tracking,
// so this is the (associative, commutative) sum the CPU computes -- the final
// clamp happens once at resolve, which is identical to the CPU's per-glyph
// min(dst+src,255) because coverage is non-negative and saturation is sticky.
static const char *const osd_combine_body =
    "ivec2 g = ivec2(gl_GlobalInvocationID.xy);\n"
    "if (g.x < rw && g.y < rh) {\n"
    "    float cov = texelFetch(src, ivec2(sx+g.x, sy+g.y), 0).r;\n"
    "    ivec2 d = ivec2(dx+g.x, dy+g.y);\n"
    "    float a = imageLoad(acc, d).r;\n"
    "    imageStore(acc, d, vec4(a + cov, 0.0, 0.0, 0.0));\n"
    "}\n";
// Resolve the f32 sum to r8 coverage with a single saturating clamp.
static const char *const osd_resolve_body =
    "ivec2 g = ivec2(gl_GlobalInvocationID.xy);\n"
    "if (g.x < rw && g.y < rh)\n"
    "    imageStore(dst, g, vec4(min(imageLoad(acc, g).r, 1.0), 0.0, 0.0, 0.0));\n";
// fix_outline: border = border>fill ? border - fill/2 : 0, in 0..255 integers
// (matching ass_fix_outline's integer floor division), in place on B.
static const char *const osd_fixoutline_body MP_UNUSED =
    "ivec2 g = ivec2(gl_GlobalInvocationID.xy);\n"
    "if (g.x < rw && g.y < rh) {\n"
    "    float bb = floor(imageLoad(bord, g).r * 255.0 + 0.5);\n"
    "    float ff = floor(texelFetch(fill, g, 0).r * 255.0 + 0.5);\n"
    "    float o = bb > ff ? bb - floor(ff * 0.5) : 0.0;\n"
    "    imageStore(bord, g, vec4(o / 255.0, 0.0, 0.0, 0.0));\n"
    "}\n";

static void osd_combine_part(struct priv *p, pl_tex atlas, pl_tex acc,
                             int sx, int sy, int dx, int dy, int rw, int rh)
{
    pl_shader sh = pl_dispatch_begin(p->osd_dp);
    struct pl_shader_desc descs[] = {
        { .desc = { .name="src", .type=PL_DESC_SAMPLED_TEX, .binding=0 },
          .binding = { .object=atlas } },
        { .desc = { .name="acc", .type=PL_DESC_STORAGE_IMG, .binding=1,
                    .access=PL_DESC_ACCESS_READWRITE },
          .binding = { .object=acc } },
    };
    struct pl_shader_var vars[] = {
        { .var = pl_var_int("sx"), .data=&sx }, { .var = pl_var_int("sy"), .data=&sy },
        { .var = pl_var_int("dx"), .data=&dx }, { .var = pl_var_int("dy"), .data=&dy },
        { .var = pl_var_int("rw"), .data=&rw }, { .var = pl_var_int("rh"), .data=&rh },
    };
    struct pl_custom_shader cs = {
        .compute = true, .compute_group_size = {16, 16},
        .descriptors = descs, .num_descriptors = 2,
        .variables = vars, .num_variables = 6, .body = osd_combine_body,
    };
    if (pl_shader_custom(sh, &cs))
        pl_dispatch_compute(p->osd_dp, pl_dispatch_compute_params(
            .shader = &sh, .dispatch_size = { (rw+15)/16, (rh+15)/16, 1 }));
    else
        pl_dispatch_abort(p->osd_dp, &sh);
}

// One r/w-storage + one sampled/storage helper for resolve and fix_outline.
static void osd_unop(struct priv *p, pl_tex a, pl_tex b, int rw, int rh,
                     const char *aname, bool a_sampled, const char *bname,
                     const char *body)
{
    pl_shader sh = pl_dispatch_begin(p->osd_dp);
    struct pl_shader_desc descs[] = {
        { .desc = { .name=aname,
                    .type = a_sampled ? PL_DESC_SAMPLED_TEX : PL_DESC_STORAGE_IMG,
                    .binding=0,
                    .access = a_sampled ? 0 : PL_DESC_ACCESS_READWRITE },
          .binding = { .object=a } },
        { .desc = { .name=bname, .type=PL_DESC_STORAGE_IMG, .binding=1,
                    .access = PL_DESC_ACCESS_READWRITE },
          .binding = { .object=b } },
    };
    struct pl_shader_var vars[] = {
        { .var = pl_var_int("rw"), .data=&rw }, { .var = pl_var_int("rh"), .data=&rh },
    };
    struct pl_custom_shader cs = {
        .compute = true, .compute_group_size = {16, 16},
        .descriptors = descs, .num_descriptors = 2,
        .variables = vars, .num_variables = 2, .body = body,
    };
    if (pl_shader_custom(sh, &cs))
        pl_dispatch_compute(p->osd_dp, pl_dispatch_compute_params(
            .shader = &sh, .dispatch_size = { (rw+15)/16, (rh+15)/16, 1 }));
    else
        pl_dispatch_abort(p->osd_dp, &sh);
}

// Copy src[0..rw,0..rh] -> dst[dx..,dy..] (r8 isn't blittable everywhere, so a
// compute copy instead of pl_tex_blit).
static const char *const osd_copy_body =
    "ivec2 g = ivec2(gl_GlobalInvocationID.xy);\n"
    "if (g.x < rw && g.y < rh)\n"
    "    imageStore(dst, ivec2(dx+g.x, dy+g.y), vec4(texelFetch(src, g, 0).r,0.0,0.0,0.0));\n";

static void MP_UNUSED osd_copy(struct priv *p, pl_tex src, pl_tex dst, int dx, int dy,
                     int rw, int rh)
{
    pl_shader sh = pl_dispatch_begin(p->osd_dp);
    struct pl_shader_desc descs[] = {
        { .desc = { .name="src", .type=PL_DESC_SAMPLED_TEX, .binding=0 },
          .binding = { .object=src } },
        { .desc = { .name="dst", .type=PL_DESC_STORAGE_IMG, .binding=1,
                    .access=PL_DESC_ACCESS_WRITEONLY },
          .binding = { .object=dst } },
    };
    struct pl_shader_var vars[] = {
        { .var = pl_var_int("dx"), .data=&dx }, { .var = pl_var_int("dy"), .data=&dy },
        { .var = pl_var_int("rw"), .data=&rw }, { .var = pl_var_int("rh"), .data=&rh },
    };
    struct pl_custom_shader cs = {
        .compute = true, .compute_group_size = {16, 16},
        .descriptors = descs, .num_descriptors = 2,
        .variables = vars, .num_variables = 4, .body = osd_copy_body,
    };
    if (pl_shader_custom(sh, &cs))
        pl_dispatch_compute(p->osd_dp, pl_dispatch_compute_params(
            .shader = &sh, .dispatch_size = { (rw+15)/16, (rh+15)/16, 1 }));
    else
        pl_dispatch_abort(p->osd_dp, &sh);
}

// Zero a storage image region via compute (pl_tex_clear needs a blittable
// texture, which r32f/r8 storage scratch isn't on all backends).
static const char *const osd_clear_body =
    "ivec2 g = ivec2(gl_GlobalInvocationID.xy);\n"
    "if (g.x < rw && g.y < rh) imageStore(acc, g, vec4(0.0));\n";

static void osd_clear(struct priv *p, pl_tex acc, int rw, int rh)
{
    pl_shader sh = pl_dispatch_begin(p->osd_dp);
    struct pl_shader_desc descs[] = {
        { .desc = { .name="acc", .type=PL_DESC_STORAGE_IMG, .binding=0,
                    .access=PL_DESC_ACCESS_WRITEONLY }, .binding = { .object=acc } },
    };
    struct pl_shader_var vars[] = {
        { .var = pl_var_int("rw"), .data=&rw }, { .var = pl_var_int("rh"), .data=&rh },
    };
    struct pl_custom_shader cs = {
        .compute = true, .compute_group_size = {16, 16},
        .descriptors = descs, .num_descriptors = 1,
        .variables = vars, .num_variables = 2, .body = osd_clear_body,
    };
    if (pl_shader_custom(sh, &cs))
        pl_dispatch_compute(p->osd_dp, pl_dispatch_compute_params(
            .shader = &sh, .dispatch_size = { (rw+15)/16, (rh+15)/16, 1 }));
    else
        pl_dispatch_abort(p->osd_dp, &sh);
}

static bool gc_ensure(pl_gpu gpu, pl_tex *t, pl_fmt fmt, int w, int h,
                      bool storable, bool sampleable, bool blit_src,
                      bool blit_dst, bool host_writable)
{
    w = MPMAX(w, *t ? (*t)->params.w : 0);
    h = MPMAX(h, *t ? (*t)->params.h : 0);
    // Respect the GPU's max 2D texture size; bail (caller skips) rather than
    // hit a libplacebo validation crash. At 8K the coverage atlas can stack
    // taller than this.
    if (w > gpu->limits.max_tex_2d_dim || h > gpu->limits.max_tex_2d_dim)
        return false;
    return pl_tex_recreate(gpu, t, pl_tex_params(
        .format = fmt, .w = w, .h = h,
        .storable = storable, .sampleable = sampleable,
        .blit_src = blit_src, .blit_dst = blit_dst, .host_writable = host_writable));
}

// Match libass's deferred-blur expand padding (ass_blur_expand_only): the blur
// runs over coverage padded by this many pixels per side, and osd_blur_part's
// edge renormalization depends on it, so the margin must equal it for a
// bit-exact match. Mirrors find_best_method() + the shift formula.
static int blur_expand_pad(float sigma)
{
    double r2 = (double) sigma * sigma;
    if (r2 <= 0.001)
        return 0;
    int level, radius;
    if (r2 < 0.5) {
        level = 0; radius = 4;
    } else {
        double frac = frexp(sqrt(0.11569 * r2 + 0.20591047), &level);
        double mul = pow(0.25, level);
        radius = 8 - (int)((10.1525 + 0.8335 * mul) * (1 - frac));
        if (radius < 4) radius = 4;
    }
    return ((radius + 4) << level) - 4;
}

struct gpos { int ax, ay; };   // a glyph's position in the persistent atlas

static struct gcache_slot *gcache_find(struct priv *p, uint64_t id)
{
    if (!p->gcache_cap)
        return NULL;
    uint64_t h = id & (p->gcache_cap - 1);
    while (p->gcache[h].id) {
        if (p->gcache[h].id == id)
            return &p->gcache[h];
        h = (h + 1) & (p->gcache_cap - 1);
    }
    return NULL;
}

static void gcache_reset(struct priv *p)
{
    if (p->gcache)
        memset(p->gcache, 0, p->gcache_cap * sizeof(p->gcache[0]));
    p->gcache_count = 0;
    p->gsx = p->gsy = p->growh = 0;
}

// Locate a glyph in the persistent atlas, uploading it from `src` on a miss
// (skyline-packed, 1px pad). Returns false if it can't be placed this frame.
// Reserve a glyph's slot in the persistent atlas. Cache hit: *upload stays
// false. Miss: a skyline slot is allocated + the id inserted, *upload set true
// (the caller uploads the pixels asynchronously, batched, to avoid a VO stall).
// Returns false only if the glyph can't be placed this frame (atlas full).
static bool gcache_reserve(struct priv *p, uint64_t id, int w, int h,
                           struct gpos *out, bool *upload)
{
    *upload = false;
    struct gcache_slot *s = gcache_find(p, id);
    if (s && s->w == w && s->h == h) {
        out->ax = s->ax; out->ay = s->ay;
        return true;
    }
    int nw = w + 1, nh = h + 1;     // 1px pad against bilinear bleed
    if (nw > p->gatlas_w || nh > p->gatlas_h)
        return false;               // glyph larger than the whole atlas: skip (no OOB)
    if (p->gsx + nw > p->gatlas_w) { p->gsy += p->growh; p->gsx = 0; p->growh = 0; }
    if (p->gsy + nh > p->gatlas_h)
        return false;               // atlas full this frame (reset happens at frame start)
    int gx = p->gsx, gy = p->gsy;
    p->gsx += nw;
    if (nh > p->growh) p->growh = nh;
    uint64_t hh = id & (p->gcache_cap - 1);
    while (p->gcache[hh].id) hh = (hh + 1) & (p->gcache_cap - 1);
    p->gcache[hh] = (struct gcache_slot){ id, gx, gy, w, h };
    p->gcache_count++;
    out->ax = gx; out->ay = gy;
    *upload = true;
    return true;
}

// One grouped composite unit: a deferred run (run_id != 0) carries its fill and
// border glyph lists separately and gets combine+blur+fix_outline; a fallback
// singleton (glyph_id == 0) is already-combined coverage that only needs blur.
struct gc_region {
    int x0, y0, x1, y1;          // screen bbox of all member parts (pre-margin)
    int margin;                  // blur halo padding added on every side
    float blur_f, blur_b;        // fill / border gaussian std-dev (0 = no blur)
    uint32_t run_flags;
    int *fill, nfill;            // part indices (layer 0 / singleton)
    int *bord, nbord;            // part indices (layer 1)
    uint32_t fill_color, bord_color;
    int fill_ax, fill_ay, bord_ax, bord_ay;  // result-atlas positions
    uint8_t single_layer;        // 0xff for a run; else the singleton's layer
    uint32_t clip_id;            // vector \clip mask to multiply by (0 = none)
    int rcx0, rcy0, rcx1, rcy1;  // rectangular \clip; visible screen rect
    uint32_t fill_color2;        // \kf wipe: fill colour right of wipe_x
    int wipe_x;                  // \kf wipe boundary (screen x); used iff KF_WIPE
    int be;                      // \be: iterations of the [1,2,1]/4 box blur
    int fill_gbase, fill_gcount; // multi-glyph run: glyph block in combwork_tex
    int bord_gbase, bord_gcount;
};

// Combine a region's glyph list (saturating add) into run_acc at run-local
// coords and resolve to cov. (Blur happens later -- after fix_outline -- to
// match libass's combine -> expand -> fix_outline -> blur order.)
static void MP_UNUSED gc_build_cov(struct priv *p, const struct sub_bitmaps *item,
                         struct gc_region *r, int *parts, int n,
                         pl_tex cov, int bw, int bh, struct gpos *cpos)
{
    osd_clear(p, p->run_acc, bw, bh);
    for (int k = 0; k < n; k++) {
        const struct sub_bitmap *b = &item->parts[parts[k]];
        int dx = b->x - r->x0 + r->margin, dy = b->y - r->y0 + r->margin;
        osd_combine_part(p, p->glyph_atlas, p->run_acc, cpos[parts[k]].ax,
                         cpos[parts[k]].ay, dx, dy, b->w, b->h);
    }
    osd_unop(p, p->run_acc, cov, bw, bh, "acc", false, "dst", osd_resolve_body);
}

static void MP_UNUSED gc_blur(struct priv *p, pl_tex cov, int bw, int bh, float sigma)
{
    osd_blur_part(p, cov, p->run_tmp, 0, 0, bw, bh, sigma, osd_blur_body_h);
    osd_blur_part(p, p->run_tmp, cov, 0, 0, bw, bh, sigma, osd_blur_body_v);
}

#define BLURWORK_W 8192
// Batched separable blur: one 16x16 workgroup per slot-tile. Each tile carries
// its result-atlas slot position, bounds and sigma -- so all regions blur in
// just two dispatches (h, v) instead of two per region.
static const char *const osd_blur_batch_pre =
    "int _wi; vec4 _t0,_t1; int _ax,_ay,_bw,_bh,_px,_py; float _sg;\n"
    "bool _bload() {\n"
    "  _wi = int(gl_WorkGroupID.y)*gridx + int(gl_WorkGroupID.x);\n"
    "  if (_wi >= ntiles) return false;\n"
    "  int w0 = 2*_wi;\n"
    "  _t0 = texelFetch(work, ivec2(w0%ww, w0/ww), 0);\n"
    "  _t1 = texelFetch(work, ivec2((w0+1)%ww, (w0+1)/ww), 0);\n"
    "  _ax=int(_t0.x); _ay=int(_t0.y); _bw=int(_t1.x); _bh=int(_t1.y); _sg=_t1.z;\n"
    "  int lx=int(_t0.z)+int(gl_LocalInvocationID.x), ly=int(_t0.w)+int(gl_LocalInvocationID.y);\n"
    "  if (lx>=_bw || ly>=_bh) return false;\n"
    "  _px=_ax+lx; _py=_ay+ly; return true;\n"
    "}\n"
    "void _taps(out int s, out int n){ int r=int(3.0*_sg+0.999); s=(r+7)/8; if(s<1)s=1; n=(r+s-1)/s; }\n";
static const char *const osd_blur_batch_h =
    "if (_bload()) {\n"
    "  int st,nt; _taps(st,nt); float a=0.0,ws=0.0;\n"
    "  for (int t=-nt;t<=nt;t++){ int d=t*st, q=_px+d;\n"
    "    if (q>=_ax && q<_ax+_bw){ float w=_sg>0.0?exp(-0.5*float(d*d)/(_sg*_sg)):float(d==0);\n"
    "      a+=w*texelFetch(src,ivec2(q,_py),0).r; ws+=w; } }\n"
    "  imageStore(dst, ivec2(_px,_py), vec4(a/ws,0.0,0.0,0.0));\n"
    "}\n";
static const char *const osd_blur_batch_v =
    "if (_bload()) {\n"
    "  int st,nt; _taps(st,nt); float a=0.0,ws=0.0;\n"
    "  for (int t=-nt;t<=nt;t++){ int d=t*st, q=_py+d;\n"
    "    if (q>=_ay && q<_ay+_bh){ float w=_sg>0.0?exp(-0.5*float(d*d)/(_sg*_sg)):float(d==0);\n"
    "      a+=w*texelFetch(src,ivec2(_px,q),0).r; ws+=w; } }\n"
    "  imageStore(dst, ivec2(_px,_py), vec4(a/ws,0.0,0.0,0.0));\n"
    "}\n";
static void gc_blur_batch(struct priv *p, pl_tex src, pl_tex dst, int ntiles,
                          const char *body)
{
    int ww = BLURWORK_W, gridx = MPMIN(ntiles, 32768), gridy = (ntiles+gridx-1)/gridx;
    pl_shader sh = pl_dispatch_begin(p->osd_dp);
    struct pl_shader_desc descs[] = {
        { .desc = { .name="src", .type=PL_DESC_SAMPLED_TEX, .binding=0 }, .binding={.object=src} },
        { .desc = { .name="dst", .type=PL_DESC_STORAGE_IMG, .binding=1,
                    .access=PL_DESC_ACCESS_WRITEONLY }, .binding={.object=dst} },
        { .desc = { .name="work", .type=PL_DESC_SAMPLED_TEX, .binding=2 },
          .binding={.object=p->blurwork_tex} },
    };
    struct pl_shader_var vars[] = {
        { .var = pl_var_int("ntiles"), .data=&ntiles }, { .var = pl_var_int("gridx"), .data=&gridx },
        { .var = pl_var_int("ww"), .data=&ww },
    };
    struct pl_custom_shader cs = {
        .compute = true, .compute_group_size = {16, 16},
        .descriptors = descs, .num_descriptors = 3,
        .variables = vars, .num_variables = 3,
        .prelude = osd_blur_batch_pre, .body = body,
    };
    if (pl_shader_custom(sh, &cs))
        pl_dispatch_compute(p->osd_dp, pl_dispatch_compute_params(
            .shader = &sh, .dispatch_size = { gridx, gridy, 1 }));
    else
        pl_dispatch_abort(p->osd_dp, &sh);
}

// Batched single-part combine: one dispatch over a part-tile work-list, plain
// imageStore (single-part slots never overlap, so no read-add needed).
static const char *const osd_combine_batch_body =
    "int wi = int(gl_WorkGroupID.y)*gridx + int(gl_WorkGroupID.x);\n"
    "if (wi >= ntiles) return;\n"
    "int w0 = 2*wi;\n"
    "vec4 t0 = texelFetch(cw, ivec2(w0%ww, w0/ww), 0);\n"
    "vec4 t1 = texelFetch(cw, ivec2((w0+1)%ww, (w0+1)/ww), 0);\n"
    "int sax=int(t0.x), say=int(t0.y), dx=int(t1.x), dy=int(t1.y), w=int(t1.z), h=int(t1.w);\n"
    "int lx=int(t0.z)+int(gl_LocalInvocationID.x), ly=int(t0.w)+int(gl_LocalInvocationID.y);\n"
    "if (lx>=w || ly>=h) return;\n"
    "float cov = texelFetch(src, ivec2(sax+lx, say+ly), 0).r;\n"
    "imageStore(acc, ivec2(dx+lx, dy+ly), vec4(cov, 0.0, 0.0, 0.0));\n";
static void gc_combine_batch(struct priv *p, pl_tex atlas, pl_tex acc, int ntiles)
{
    int ww = BLURWORK_W, gridx = MPMIN(ntiles, 32768), gridy = (ntiles+gridx-1)/gridx;
    pl_shader sh = pl_dispatch_begin(p->osd_dp);
    struct pl_shader_desc descs[] = {
        { .desc = { .name="src", .type=PL_DESC_SAMPLED_TEX, .binding=0 }, .binding={.object=atlas} },
        { .desc = { .name="acc", .type=PL_DESC_STORAGE_IMG, .binding=1,
                    .access=PL_DESC_ACCESS_WRITEONLY }, .binding={.object=acc} },
        { .desc = { .name="cw", .type=PL_DESC_SAMPLED_TEX, .binding=2 },
          .binding={.object=p->combwork_tex} },
    };
    struct pl_shader_var vars[] = {
        { .var = pl_var_int("ntiles"), .data=&ntiles }, { .var = pl_var_int("gridx"), .data=&gridx },
        { .var = pl_var_int("ww"), .data=&ww },
    };
    struct pl_custom_shader cs = {
        .compute = true, .compute_group_size = {16, 16},
        .descriptors = descs, .num_descriptors = 3,
        .variables = vars, .num_variables = 3, .body = osd_combine_batch_body,
    };
    if (pl_shader_custom(sh, &cs))
        pl_dispatch_compute(p->osd_dp, pl_dispatch_compute_params(
            .shader = &sh, .dispatch_size = { gridx, gridy, 1 }));
    else
        pl_dispatch_abort(p->osd_dp, &sh);
}
// Per-run combine: ONE dispatch per multi-glyph run slot. Each output pixel loops
// over the run's glyphs (block [gbase, gbase+gcount) in the glyph texture) and
// sums their coverage -- one write per pixel, no atomics (replaces gcount
// serialized read-add dispatches; the recording cost that spiked dense frames).
static const char *const osd_combine_run_body =
    "ivec2 g = ivec2(gl_GlobalInvocationID.xy);\n"
    "if (g.x >= bw || g.y >= bh) return;\n"
    "float acc = 0.0;\n"
    "for (int i = 0; i < gcount; i++) {\n"
    "  int e = 2*(gbase+i);\n"
    "  vec4 q0 = texelFetch(gl, ivec2(e%ww, e/ww), 0);\n"
    "  vec4 q1 = texelFetch(gl, ivec2((e+1)%ww, (e+1)/ww), 0);\n"
    "  int sax=int(q0.x), say=int(q0.y), dox=int(q0.z), doy=int(q0.w), w=int(q1.x), h=int(q1.y);\n"
    "  int lx=g.x-dox, ly=g.y-doy;\n"
    "  if (lx>=0 && ly>=0 && lx<w && ly<h) acc += texelFetch(src, ivec2(sax+lx, say+ly), 0).r;\n"
    "}\n"
    "imageStore(dst, ivec2(ax+g.x, ay+g.y), vec4(acc, 0.0, 0.0, 0.0));\n";
static void gc_combine_run(struct priv *p, pl_tex atlas, pl_tex acc, int ax, int ay,
                           int bw, int bh, int gbase, int gcount)
{
    int ww = BLURWORK_W;
    pl_shader sh = pl_dispatch_begin(p->osd_dp);
    struct pl_shader_desc descs[] = {
        { .desc = { .name="src", .type=PL_DESC_SAMPLED_TEX, .binding=0 }, .binding={.object=atlas} },
        { .desc = { .name="dst", .type=PL_DESC_STORAGE_IMG, .binding=1,
                    .access=PL_DESC_ACCESS_WRITEONLY }, .binding={.object=acc} },
        { .desc = { .name="gl", .type=PL_DESC_SAMPLED_TEX, .binding=2 },
          .binding={.object=p->combwork_tex} },
    };
    struct pl_shader_var vars[] = {
        { .var = pl_var_int("ax"), .data=&ax }, { .var = pl_var_int("ay"), .data=&ay },
        { .var = pl_var_int("bw"), .data=&bw }, { .var = pl_var_int("bh"), .data=&bh },
        { .var = pl_var_int("gbase"), .data=&gbase }, { .var = pl_var_int("gcount"), .data=&gcount },
        { .var = pl_var_int("ww"), .data=&ww },
    };
    struct pl_custom_shader cs = {
        .compute = true, .compute_group_size = {16, 16},
        .descriptors = descs, .num_descriptors = 3,
        .variables = vars, .num_variables = 7, .body = osd_combine_run_body,
    };
    if (pl_shader_custom(sh, &cs))
        pl_dispatch_compute(p->osd_dp, pl_dispatch_compute_params(
            .shader = &sh, .dispatch_size = { (bw+15)/16, (bh+15)/16, 1 }));
    else
        pl_dispatch_abort(p->osd_dp, &sh);
}

// fix_outline between two slots of the same result atlas (border -= fill/2).
static const char *const osd_fixoutline_slot_body =
    "ivec2 g = ivec2(gl_GlobalInvocationID.xy);\n"
    "if (g.x < rw && g.y < rh) {\n"
    "    ivec2 bp = ivec2(box+g.x, boy+g.y), fp = ivec2(fox+g.x, foy+g.y);\n"
    "    float bb = floor(imageLoad(res, bp).r*255.0+0.5);\n"
    "    float ff = floor(imageLoad(res, fp).r*255.0+0.5);\n"
    "    float o = bb > ff ? bb - floor(ff*0.5) : 0.0;\n"
    "    imageStore(res, bp, vec4(o/255.0, 0.0, 0.0, 0.0));\n"
    "}\n";
static void gc_fixoutline_slot(struct priv *p, pl_tex res, int box, int boy,
                               int fox, int foy, int rw, int rh)
{
    pl_shader sh = pl_dispatch_begin(p->osd_dp);
    struct pl_shader_desc descs[] = {
        { .desc = { .name="res", .type=PL_DESC_STORAGE_IMG, .binding=0,
                    .access=PL_DESC_ACCESS_READWRITE }, .binding={.object=res} },
    };
    struct pl_shader_var vars[] = {
        { .var = pl_var_int("box"), .data=&box }, { .var = pl_var_int("boy"), .data=&boy },
        { .var = pl_var_int("fox"), .data=&fox }, { .var = pl_var_int("foy"), .data=&foy },
        { .var = pl_var_int("rw"), .data=&rw }, { .var = pl_var_int("rh"), .data=&rh },
    };
    struct pl_custom_shader cs = {
        .compute = true, .compute_group_size = {16, 16},
        .descriptors = descs, .num_descriptors = 1,
        .variables = vars, .num_variables = 6, .body = osd_fixoutline_slot_body,
    };
    if (pl_shader_custom(sh, &cs))
        pl_dispatch_compute(p->osd_dp, pl_dispatch_compute_params(
            .shader = &sh, .dispatch_size = { (rw+15)/16, (rh+15)/16, 1 }));
    else
        pl_dispatch_abort(p->osd_dp, &sh);
}

// \be edge-blur: one separable [1,2,1]/4 box pass over a result slot (outside the
// slot reads 0, matching libass's be_blur edges). horiz!=0 = x pass, else y.
static const char *const osd_be_body =
    "ivec2 g = ivec2(gl_GlobalInvocationID.xy);\n"
    "if (g.x < bw && g.y < bh) {\n"
    "    ivec2 b = ivec2(ax, ay);\n"
    "    float c = imageLoad(src, b + g).r;\n"
    "    float l, r;\n"
    "    if (horiz != 0) {\n"
    "        l = g.x > 0    ? imageLoad(src, b + ivec2(g.x-1, g.y)).r : 0.0;\n"
    "        r = g.x < bw-1 ? imageLoad(src, b + ivec2(g.x+1, g.y)).r : 0.0;\n"
    "    } else {\n"
    "        l = g.y > 0    ? imageLoad(src, b + ivec2(g.x, g.y-1)).r : 0.0;\n"
    "        r = g.y < bh-1 ? imageLoad(src, b + ivec2(g.x, g.y+1)).r : 0.0;\n"
    "    }\n"
    "    imageStore(dst, b + g, vec4((l + 2.0*c + r) * 0.25, 0.0, 0.0, 0.0));\n"
    "}\n";
static void gc_be_pass(struct priv *p, pl_tex src, pl_tex dst,
                       int ax, int ay, int bw, int bh, int horiz)
{
    pl_shader sh = pl_dispatch_begin(p->osd_dp);
    struct pl_shader_desc descs[] = {
        { .desc = { .name="src", .type=PL_DESC_STORAGE_IMG, .binding=0,
                    .access=PL_DESC_ACCESS_READONLY }, .binding={.object=src} },
        { .desc = { .name="dst", .type=PL_DESC_STORAGE_IMG, .binding=1,
                    .access=PL_DESC_ACCESS_WRITEONLY }, .binding={.object=dst} },
    };
    struct pl_shader_var vars[] = {
        { .var = pl_var_int("ax"), .data=&ax }, { .var = pl_var_int("ay"), .data=&ay },
        { .var = pl_var_int("bw"), .data=&bw }, { .var = pl_var_int("bh"), .data=&bh },
        { .var = pl_var_int("horiz"), .data=&horiz },
    };
    struct pl_custom_shader cs = {
        .compute = true, .compute_group_size = {16, 16},
        .descriptors = descs, .num_descriptors = 2,
        .variables = vars, .num_variables = 5, .body = osd_be_body,
    };
    if (pl_shader_custom(sh, &cs))
        pl_dispatch_compute(p->osd_dp, pl_dispatch_compute_params(
            .shader = &sh, .dispatch_size = { (bw+15)/16, (bh+15)/16, 1 }));
    else
        pl_dispatch_abort(p->osd_dp, &sh);
}
// `be` iterations of the box blur on a result slot, ping-ponging through tmp.
static void gc_be_blur(struct priv *p, pl_tex res, pl_tex tmp,
                       int ax, int ay, int bw, int bh, int be)
{
    for (int i = 0; i < be; i++) {
        gc_be_pass(p, res, tmp, ax, ay, bw, bh, 1);   // x pass: res -> tmp
        gc_be_pass(p, tmp, res, ax, ay, bw, bh, 0);   // y pass: tmp -> res
    }
}

// A vector-\clip mask, rasterized into the glyph atlas (like a glyph) at (ax,ay)
// and corresponding to screen origin (sx,sy); multiply a run's coverage by it.
struct clipmask { uint32_t id; int ax, ay, sx, sy, w, h, inv; };

static const char *const osd_clipmult_body =
    "ivec2 g = ivec2(gl_GlobalInvocationID.xy);\n"
    "if (g.x >= bw || g.y >= bh) return;\n"
    "int mlx = ox + g.x - csx, mly = oy + g.y - csy;\n"   // clip-mask-local coord
    "float m = 0.0;\n"
    "if (mlx >= 0 && mlx < cw && mly >= 0 && mly < ch)\n"
    "    m = texelFetch(mask, ivec2(cax + mlx, cay + mly), 0).r;\n"
    "float f = inv != 0 ? (1.0 - m) : m;\n"
    "ivec2 sp = ivec2(sox + g.x, soy + g.y);\n"
    "float c = imageLoad(cov, sp).r;\n"
    "imageStore(cov, sp, vec4(c * f, 0.0, 0.0, 0.0));\n";

// Multiply cov (at slot (sox,soy), screen origin (ox,oy)) by the clip mask.
static void gc_clip_mult(struct priv *p, pl_tex cov, int bw, int bh,
                         int sox, int soy, int ox, int oy,
                         const struct clipmask *cm)
{
    int cax = cm->ax, cay = cm->ay, csx = cm->sx, csy = cm->sy,
        cw = cm->w, ch = cm->h, inv = cm->inv;
    pl_shader sh = pl_dispatch_begin(p->osd_dp);
    struct pl_shader_desc descs[] = {
        { .desc = { .name = "cov", .type = PL_DESC_STORAGE_IMG, .binding = 0,
                    .access = PL_DESC_ACCESS_READWRITE }, .binding = { .object = cov } },
        { .desc = { .name = "mask", .type = PL_DESC_SAMPLED_TEX, .binding = 1 },
          .binding = { .object = p->glyph_atlas } },
    };
    struct pl_shader_var vars[] = {
        { .var = pl_var_int("bw"), .data = &bw }, { .var = pl_var_int("bh"), .data = &bh },
        { .var = pl_var_int("sox"), .data = &sox }, { .var = pl_var_int("soy"), .data = &soy },
        { .var = pl_var_int("ox"), .data = &ox }, { .var = pl_var_int("oy"), .data = &oy },
        { .var = pl_var_int("cax"), .data = &cax }, { .var = pl_var_int("cay"), .data = &cay },
        { .var = pl_var_int("csx"), .data = &csx }, { .var = pl_var_int("csy"), .data = &csy },
        { .var = pl_var_int("cw"), .data = &cw }, { .var = pl_var_int("ch"), .data = &ch },
        { .var = pl_var_int("inv"), .data = &inv },
    };
    struct pl_custom_shader cs = {
        .compute = true, .compute_group_size = { 16, 16 },
        .descriptors = descs, .num_descriptors = 2,
        .variables = vars, .num_variables = 13, .body = osd_clipmult_body,
    };
    if (pl_shader_custom(sh, &cs))
        pl_dispatch_compute(p->osd_dp, pl_dispatch_compute_params(
            .shader = &sh, .dispatch_size = { (bw + 15) / 16, (bh + 15) / 16, 1 }));
    else
        pl_dispatch_abort(p->osd_dp, &sh);
}

static void gc_apply_clip(struct priv *p, pl_tex cov, int bw, int bh, int sox, int soy,
                          struct gc_region *r, struct clipmask *clips, int nclips)
{
    if (!r->clip_id)
        return;
    for (int c = 0; c < nclips; c++)
        if (clips[c].id == r->clip_id) {
            gc_clip_mult(p, cov, bw, bh, sox, soy,
                         r->x0 - r->margin, r->y0 - r->margin, &clips[c]);
            return;
        }
}

// --- GPU glyph rasterizer (SUBBITMAP_LIBASS_OUTLINES) -----------------------
// Exact-area analytic coverage from a glyph's line segments, written into the
// glyph atlas slot. Matches libass's CPU raster to <=9/255 (validated, edge-AA
// only). Antiderivative of clamp(t,0,1) for the exact clamped-ramp integral.
#define EDGE_TEX_W 8192   // edge_tex/hdr_tex/work_tex width; element i is at (i%W, i/W).
#define HDR_TEX_W  8192   // Wide so the height (count/W) stays under max_tex_2d_dim at 8K.
#define WORK_TEX_W 8192
#define RASTER_BAND_H 8   // Y-band height (px): a pixel only tests edges crossing its band.
#define TILE_EXPORT_W 11  // libass tile-export: tx,ty,ng,group0[4],group1[4] (matches ass_rasterizer.h)
#define SEG_EXPORT_W  8   // libass tile-export: a,b,c,flags,xmin,ymin,ymax,_ (2 rgba32f texels)
#define RUN_FLAG_CLIP_MASK    0x2  // libass ABI: this part is a vector-clip mask
#define RUN_FLAG_CLIP_INVERSE 0x4  // the clip mask is inverse (\iclip)
#define RUN_FLAG_KF_WIPE      0x8  // \kf fill: fill_color left of wipe_x, fill_color2 right
#define RUN_FLAG_RECT_INVERSE 0x10 // \iclip rect: the clip rect is EXCLUDED, not visible
#define RUN_FLAG_SHADOW       0x20 // drop shadow: draw behind border+fill
// Faithful float port of libass update_border_line (rasterizer_template.h): the
// per-pixel coverage of a partial sub-pixel row span [up,dn] (1/64 units),
// FULL_VALUE=1024, TILE_ORDER=4. Used by the res/winding filler below so the GPU
// matches libass's analytic AA (incl. filling stroke self-overlaps).
static const char *const osd_raster_prelude =
    "float ubl(int px, float abs_a, float a, float b, float abs_b, float c, float up, float dn){\n"
    "  float size = dn - up;\n"
    "  float w = min(1024.0 + size*16.0 - abs_a, 1024.0) * 8.0;\n"
    "  float dc_b = abs_b*size/64.0;\n"
    "  float dc = (min(abs_a, dc_b)+2.0)/4.0;\n"
    "  float base = b*(up+dn)/128.0;\n"
    "  float offs1 = size - (base+dc)*w/65536.0;\n"
    "  float offs2 = size - (base-dc)*w/65536.0;\n"
    "  float size2 = size*2.0;\n"
    "  float cw = (c - a*float(px))*w/65536.0;\n"
    "  float c1 = clamp(cw+offs1, 0.0, size2);\n"
    "  float c2 = clamp(cw+offs2, 0.0, size2);\n"
    "  return c1+c2;\n"
    "}\n";
// One 16x16 workgroup per work-list tile. Each tile carries the glyph's atlas
// origin, the tile offset/size, and 1-2 groups of clipped segments + winding
// (from libass's tile-split, ass_outline_to_tiles). Per pixel: run libass's
// generic_tile filler (res = unsigned 2-sample coverage; cur = winding via the
// SEGFLAG_DN delta; |res+cur|) for each group and max-merge them -- matching
// libass CPU exactly, including stroke self-intersections.
static const char *const osd_raster_body =
    "int wi = int(gl_WorkGroupID.y) * gridx + int(gl_WorkGroupID.x);\n"
    "if (wi >= ntiles) return;\n"
    "int w0 = 4 * wi;\n"
    "vec4 A  = texelFetch(work, ivec2(w0 % ww, w0 / ww), 0);\n"            // ax,ay,tx,ty
    "vec4 WH = texelFetch(work, ivec2((w0+1) % ww, (w0+1) / ww), 0);\n"    // w,h,ng,_
    "vec4 GG[2];\n"
    "GG[0] = texelFetch(work, ivec2((w0+2) % ww, (w0+2) / ww), 0);\n"
    "GG[1] = texelFetch(work, ivec2((w0+3) % ww, (w0+3) / ww), 0);\n"
    "int ax=int(A.x), ay=int(A.y), tx=int(A.z), ty=int(A.w);\n"
    "int gw=int(WH.x), gh=int(WH.y), ng=int(WH.z);\n"
    "int lx=int(gl_LocalInvocationID.x), ly=int(gl_LocalInvocationID.y);\n"
    "if (tx+lx >= gw || ty+ly >= gh) return;\n"
    "float cov = 0.0;\n"
    "for (int gi = 0; gi < 2; gi++) {\n"
    "  if (gi >= ng) break;\n"
    "  vec4 G = GG[gi];\n"
    "  int type=int(G.x); float wind=G.y; int soff=int(G.z); int scnt=int(G.w);\n"
    "  float v = 0.0;\n"
    "  if (type == 0) { v = wind != 0.0 ? 255.0 : 0.0; }\n"               // solid
    "  else if (type == 1) {\n"                                           // halfplane
    "    int e=soff*2; vec4 s0=texelFetch(edges, ivec2(e % ew, e / ew), 0);\n"
    "    float aa=s0.x, bb=s0.y;\n"
    "    float cc = s0.z + 512.0 - (aa+bb)*0.5 - bb*float(ly);\n"
    "    float dl = (min(abs(aa),abs(bb))+2.0)/4.0;\n"
    "    float c1=clamp(cc-aa*float(lx)-dl,0.0,1024.0), c2=clamp(cc-aa*float(lx)+dl,0.0,1024.0);\n"
    "    v = min((c1+c2)/8.0, 255.0);\n"
    "  } else {\n"                                                        // generic
    "    float res=0.0, cur=256.0*wind;\n"
    "    for (int i=0;i<scnt;i++){\n"
    "      int e=(soff+i)*2;\n"
    "      vec4 s0=texelFetch(edges, ivec2(e % ew, e / ew), 0);\n"
    "      vec4 s1=texelFetch(edges, ivec2((e+1) % ew, (e+1) / ew), 0);\n"
    "      float a=s0.x, b=s0.y, c0=s0.z; int flags=int(s0.w);\n"
    "      int xmin=int(s1.x), ymn=int(s1.y), ymx=int(s1.z);\n"
    "      float upd = (flags&1)!=0 ? 4.0 : 0.0, dnd=upd;\n"
    "      if (xmin==0 && (flags&4)!=0) dnd=4.0-dnd;\n"
    "      if ((flags&2)!=0){ float t=upd; upd=dnd; dnd=t; }\n"
    "      int up=ymn>>6, dn=ymx>>6; float upp=float(ymn&63), dnp=float(ymx&63);\n"
    "      if (up   <= ly) cur-=upd*64.0-upd*upp;\n"
    "      if (up+1 <= ly) cur-=upd*upp;\n"
    "      if (dn   <= ly) cur+=dnd*64.0-dnd*dnp;\n"
    "      if (dn+1 <= ly) cur+=dnd*dnp;\n"
    "      if (ymn==ymx) continue;\n"
    "      float abs_a=abs(a), abs_b=abs(b);\n"
    "      float dc=(min(abs_a,abs_b)+2.0)/4.0;\n"
    "      float base=512.0 - b*0.5;\n"
    "      float c=c0 - a*0.5 - b*float(up); int rup=up;\n"
    "      if (upp!=0.0){\n"
    "        if (dn==up){ if(ly==up) res+=ubl(lx,abs_a,a,b,abs_b,c,upp,dnp); continue; }\n"
    "        if (ly==up) res+=ubl(lx,abs_a,a,b,abs_b,c,upp,64.0); rup=up+1; c-=b;\n"
    "      }\n"
    "      if (ly>=rup && ly<dn){\n"
    "        float cy=c - b*float(ly-rup);\n"
    "        float c1=clamp(cy-a*float(lx)+base+dc,0.0,1024.0), c2=clamp(cy-a*float(lx)+base-dc,0.0,1024.0);\n"
    "        res+=(c1+c2)/8.0;\n"
    "      }\n"
    "      if (dnp!=0.0 && ly==dn){ float cy=c - b*float(dn-rup); res+=ubl(lx,abs_a,a,b,abs_b,cy,0.0,dnp); }\n"
    "    }\n"
    "    float val=res+cur; v = min(max(val,-val), 255.0);\n"
    "  }\n"
    "  cov = gi==0 ? v : max(cov, v);\n"
    "}\n"
    "imageStore(dst, ivec2(ax+tx+lx, ay+ty+ly), vec4(cov/255.0, 0.0, 0.0, 0.0));\n";

// Rasterize ALL collected glyphs in one dispatch: ntiles 16x16 work-list tiles.
static void gc_raster_batch(struct priv *p, int ntiles)
{
    int ew = EDGE_TEX_W, ww = WORK_TEX_W;
    int gridx = MPMIN(ntiles, 32768), gridy = (ntiles + gridx - 1) / gridx;
    pl_shader sh = pl_dispatch_begin(p->osd_dp);
    struct pl_shader_desc descs[] = {
        { .desc = { .name = "dst", .type = PL_DESC_STORAGE_IMG, .binding = 0,
                    .access = PL_DESC_ACCESS_WRITEONLY }, .binding = { .object = p->glyph_atlas } },
        { .desc = { .name = "edges", .type = PL_DESC_SAMPLED_TEX, .binding = 1 },
          .binding = { .object = p->edge_tex } },
        { .desc = { .name = "work", .type = PL_DESC_SAMPLED_TEX, .binding = 2 },
          .binding = { .object = p->work_tex } },
    };
    struct pl_shader_var vars[] = {
        { .var = pl_var_int("ntiles"), .data = &ntiles }, { .var = pl_var_int("gridx"), .data = &gridx },
        { .var = pl_var_int("ew"), .data = &ew }, { .var = pl_var_int("ww"), .data = &ww },
    };
    struct pl_custom_shader cs = {
        .compute = true, .compute_group_size = { 16, 16 },
        .descriptors = descs, .num_descriptors = 3,
        .variables = vars, .num_variables = 4,
        .prelude = osd_raster_prelude, .body = osd_raster_body,
    };
    if (pl_shader_custom(sh, &cs))
        pl_dispatch_compute(p->osd_dp, pl_dispatch_compute_params(
            .shader = &sh, .dispatch_size = { gridx, gridy, 1 }));
    else
        pl_dispatch_abort(p->osd_dp, &sh);
}

// Composite a SUBBITMAP_LIBASS_GLYPHS item: reproduce libass's per-run combine
// + blur + fix_outline on the GPU into entry->tex (a result atlas), then emit a
// single monochrome overlay whose parts reference each run's coverage region.
static void compose_glyph_runs(struct priv *p, const struct sub_bitmaps *item,
                               struct osd_entry *entry, struct pl_frame *frame,
                               struct osd_state *state, enum pl_overlay_coords coords,
                               struct mp_image *src)
{
    pl_gpu gpu = p->gpu;
    pl_fmt r8 = p->osd_fmt[SUBBITMAP_LIBASS];
    if (!r8 || !p->osd_acc_fmt || !(p->osd_acc_fmt->caps & PL_FMT_CAP_STORABLE) ||
        !(r8->caps & PL_FMT_CAP_STORABLE))
        return;

    // Persistent glyph cache: ensure the atlas + map, reset if near-full, then
    // upload only the cache-miss glyphs (the per-frame bandwidth win).
    if (!p->glyph_atlas) {
        // As large as the GPU allows: a dense 8K frame's glyphs + clip masks must
        // mostly fit at once or later ones get dropped (flashing). 16384^2 = 256MB.
        p->gatlas_w = p->gatlas_h = MPMIN(gpu->limits.max_tex_2d_dim, 16384);
        // storable too: the GPU rasterizer (outline mode) writes coverage here.
        if (!gc_ensure(gpu, &p->glyph_atlas, r8, p->gatlas_w, p->gatlas_h,
                       true, true, false, false, true))
            return;
        p->gcache_cap = 1 << 15;
        p->gcache = talloc_zero_array(p, struct gcache_slot, p->gcache_cap);
    }
    if (p->gsy > p->gatlas_h - 256 || p->gcache_count * 10 > p->gcache_cap * 7)
        gcache_reset(p);

    void *tmp = talloc_new(NULL);

    // Resolve every deferred glyph to its atlas position, collecting cache
    // misses to upload in one async batch (a synchronous per-glyph .ptr upload
    // stalls the VO thread on d3d11; see the staging-buffer note below).
    bool is_outline = item->format == SUBBITMAP_LIBASS_OUTLINES;
    struct gpos *cpos = talloc_array(tmp, struct gpos, item->num_parts);
    struct gmiss { int ax, ay, w, h; const uint8_t *src; };
    struct gmiss *miss = NULL;
    int nmiss = 0;
    size_t miss_bytes = 0;
    // outline raster jobs: per glyph, its atlas pos + size + the tile-export data
    struct rjob { int ax, ay, w, h, sbase, nt; const float *tiles; } *rjobs = NULL;
    int nrjobs = 0, ne = 0, nh = 0;          // ne = exported segments (2 texels each), nh unused
    // Vector-\clip masks found in this item: rasterized like glyphs, then used to
    // multiply each clipped run's coverage.
    struct clipmask *clips = NULL;
    int nclips = 0;
    ptrdiff_t pstride = is_outline ? 0 : item->packed->stride[0];
    // The persistent glyph atlas accumulates glyphs across frames; a dense frame
    // can exhaust the remaining space mid-frame, dropping its later glyphs/clip
    // masks (flashing). Nothing is rasterized until after this loop, so if a
    // reservation fails, reset the atlas once and rebuild from empty.
    bool gc_retried = false;
gc_restart:
    nmiss = nrjobs = nclips = ne = nh = 0;
    miss_bytes = 0;
    for (int i = 0; i < item->num_parts; i++) {
        const struct sub_bitmap *b = &item->parts[i];
        cpos[i] = (struct gpos){0, 0};
        if (b->libass.glyph_id == 0)
            continue;
        bool up;
        if (!gcache_reserve(p, b->libass.glyph_id, b->w, b->h, &cpos[i], &up)) {
            if (!gc_retried) { gcache_reset(p); gc_retried = true; goto gc_restart; }
            continue;   // already rebuilt from empty: this frame genuinely overflows
        }
        if (b->libass.run_flags & RUN_FLAG_CLIP_MASK)   // record (still rasterized below)
            MP_TARRAY_APPEND(tmp, clips, nclips, ((struct clipmask){
                b->libass.run_id, cpos[i].ax, cpos[i].ay, b->x, b->y, b->w, b->h,
                !!(b->libass.run_flags & RUN_FLAG_CLIP_INVERSE) }));
        if (up && is_outline) {
            // libass tile-export blob: [n_tiles, n_segs, tiles(11f), segs(8f)].
            // Append this glyph's segments to the shared seg buffer (2 rgba32f
            // texels each) and record the glyph's tiles for the work-list below.
            const int32_t *blob = b->libass.outline;
            if (!blob || b->libass.n_outline < 2) continue;
            int nt = blob[0], ns = blob[1];
            const float *gtiles = (const float *)(blob + 2);
            const float *gsegs  = (const float *)(blob + 2 + (size_t) nt * TILE_EXPORT_W);
            if (p->ebuf_cap < (ne + ns) * SEG_EXPORT_W) {
                p->ebuf_cap = MPMAX((ne + ns) * SEG_EXPORT_W * 2, 8192);
                p->ebuf = talloc_realloc(p, p->ebuf, float, p->ebuf_cap);
            }
            memcpy(p->ebuf + (size_t) ne * SEG_EXPORT_W, gsegs,
                   (size_t) ns * SEG_EXPORT_W * sizeof(float));
            MP_TARRAY_APPEND(tmp, rjobs, nrjobs,
                ((struct rjob){ cpos[i].ax, cpos[i].ay, b->w, b->h, ne, nt, gtiles }));
            ne += ns;
        } else if (up) {
            const uint8_t *gsrc = (const uint8_t *) item->packed->planes[0]
                                + (ptrdiff_t) b->src_y * pstride + b->src_x;
            MP_TARRAY_APPEND(tmp, miss, nmiss,
                ((struct gmiss){ cpos[i].ax, cpos[i].ay, b->w, b->h, gsrc }));
            miss_bytes += (size_t) b->w * b->h;
        }
    }

    // Upload the segment pool once, then dispatch one workgroup per tile.
    if (ne) {
        int seg_texels = ne * 2;                                // 2 rgba32f texels per segment
        int eh = (seg_texels + EDGE_TEX_W - 1) / EDGE_TEX_W;
        size_t eneed = (size_t) EDGE_TEX_W * eh * 4;
        if ((size_t) p->ebuf_cap < eneed) {
            p->ebuf_cap = eneed; p->ebuf = talloc_realloc(p, p->ebuf, float, p->ebuf_cap);
        }
        // Build the per-tile work-list (one 16x16 tile = one workgroup). 4 texels/tile:
        // [ax,ay,tx,ty] [w,h,ng,_] group0[type,wind,segoff,cnt] group1[...] (g1.type<0 = none)
        int ntiles = 0;
        for (int j = 0; j < nrjobs; j++) ntiles += rjobs[j].nt;
        int wh = (4 * ntiles + WORK_TEX_W - 1) / WORK_TEX_W;
        size_t wneed = (size_t) WORK_TEX_W * wh * 4;
        if ((size_t) p->wbuf_cap < wneed) {
            p->wbuf_cap = wneed; p->wbuf = talloc_realloc(p, p->wbuf, float, p->wbuf_cap);
        }
        int ti = 0;
        for (int j = 0; j < nrjobs; j++) {
            struct rjob *r = &rjobs[j];
            for (int t = 0; t < r->nt; t++) {
                const float *T = r->tiles + (size_t) t * TILE_EXPORT_W;  // tx,ty,ng,g0[4],g1[4]
                float *w0 = &p->wbuf[(4 * ti) * 4];
                int ng = (int) T[2];
                w0[0]=r->ax; w0[1]=r->ay; w0[2]=T[0]; w0[3]=T[1];        // ax,ay,tx,ty
                w0[4]=r->w;  w0[5]=r->h;  w0[6]=T[2]; w0[7]=0;           // w,h,ng
                w0[8]=T[3];  w0[9]=T[4];  w0[10]=T[5]+r->sbase; w0[11]=T[6];   // group0
                if (ng >= 2) { w0[12]=T[7]; w0[13]=T[8]; w0[14]=T[9]+r->sbase; w0[15]=T[10]; }
                else { w0[12]=-1; w0[13]=w0[14]=w0[15]=0; }
                ti++;
            }
        }
        pl_fmt ef = pl_find_named_fmt(gpu, "rgba32f");
        if (ef && gc_ensure(gpu, &p->edge_tex, ef, EDGE_TEX_W, eh, false, true, false, false, true) &&
                  gc_ensure(gpu, &p->work_tex, ef, WORK_TEX_W, wh, false, true, false, false, true)) {
            pl_tex_upload(gpu, pl_tex_transfer_params(.tex = p->edge_tex,
                .rc = { .x1 = EDGE_TEX_W, .y1 = eh },
                .row_pitch = (size_t) EDGE_TEX_W * 4 * sizeof(float), .ptr = p->ebuf));
            pl_tex_upload(gpu, pl_tex_transfer_params(.tex = p->work_tex,
                .rc = { .x1 = WORK_TEX_W, .y1 = wh },
                .row_pitch = (size_t) WORK_TEX_W * 4 * sizeof(float), .ptr = p->wbuf));
            gc_raster_batch(p, ntiles);
        }
    }

    // Flush the misses: tight-repack into a CPU scratch, one async buffer write,
    // then per-glyph async texture uploads (buffer-backed = no VO-thread stall).
    if (nmiss) {
        if (p->gstage_cpu_sz < miss_bytes) {
            p->gstage_cpu = talloc_realloc(p, p->gstage_cpu, uint8_t, miss_bytes);
            p->gstage_cpu_sz = miss_bytes;
        }
        pl_buf *ring = &p->glyph_stage[p->glyph_stage_idx++ % 3];
        bool buf_ok = (*ring) && (*ring)->params.size >= miss_bytes;
        if (!buf_ok) {
            size_t want = (miss_bytes + (1u << 20) - 1) & ~(size_t)((1u << 20) - 1);
            buf_ok = pl_buf_recreate(gpu, ring,
                                     pl_buf_params(.size = want, .host_writable = true));
        }
        size_t off = 0;
        size_t *offs = talloc_array(tmp, size_t, nmiss);
        for (int m = 0; m < nmiss; m++) {           // repack tight into the scratch
            struct gmiss *g = &miss[m];
            offs[m] = off;
            for (int row = 0; row < g->h; row++)
                memcpy(p->gstage_cpu + off + (size_t) row * g->w,
                       g->src + (ptrdiff_t) row * pstride, g->w);
            off += (size_t) g->w * g->h;
        }
        if (buf_ok) {
            pl_buf_write(gpu, *ring, 0, p->gstage_cpu, miss_bytes);
            for (int m = 0; m < nmiss; m++) {
                struct gmiss *g = &miss[m];
                pl_tex_upload(gpu, pl_tex_transfer_params(.tex = p->glyph_atlas,
                    .rc = { .x0 = g->ax, .y0 = g->ay, .x1 = g->ax + g->w, .y1 = g->ay + g->h },
                    .row_pitch = g->w, .buf = *ring, .buf_offset = offs[m]));
            }
        } else {                                    // fallback: synchronous
            for (int m = 0; m < nmiss; m++) {
                struct gmiss *g = &miss[m];
                pl_tex_upload(gpu, pl_tex_transfer_params(.tex = p->glyph_atlas,
                    .rc = { .x0 = g->ax, .y0 = g->ay, .x1 = g->ax + g->w, .y1 = g->ay + g->h },
                    .row_pitch = g->w, .ptr = p->gstage_cpu + offs[m]));
            }
        }
    }

    int max_run = 0;
    for (int i = 0; i < item->num_parts; i++)
        max_run = MPMAX(max_run, (int) item->parts[i].libass.run_id);
    int *idx = talloc_array(tmp, int, max_run + 1);
    for (int i = 0; i <= max_run; i++) idx[i] = -1;

    struct gc_region *regs = NULL;
    int nregs = 0;
    for (int i = 0; i < item->num_parts; i++) {
        const struct sub_bitmap *b = &item->parts[i];
        struct gc_region *r;
        if (b->libass.glyph_id == 0)
            continue;   // already-combined fallback parts go via the legacy path
        if (b->libass.run_flags & RUN_FLAG_CLIP_MASK)
            continue;   // clip masks are multiplied in, not drawn as runs
        int ri = idx[b->libass.run_id];
        if (ri < 0) {
            MP_TARRAY_GROW(tmp, regs, nregs);
            ri = idx[b->libass.run_id] = nregs++;
            regs[ri] = (struct gc_region){ .x0 = b->x, .y0 = b->y,
                .x1 = b->x + b->dw, .y1 = b->y + b->dh,
                .run_flags = b->libass.run_flags, .single_layer = 0xff,
                .clip_id = b->libass.clip_id,
                .rcx0 = b->libass.clip_rx0, .rcy0 = b->libass.clip_ry0,
                .rcx1 = b->libass.clip_rx1, .rcy1 = b->libass.clip_ry1,
                .be = b->libass.be };
        }
        r = &regs[ri];
        r->x0 = MPMIN(r->x0, b->x); r->y0 = MPMIN(r->y0, b->y);
        r->x1 = MPMAX(r->x1, b->x + b->dw); r->y1 = MPMAX(r->y1, b->y + b->dh);
        if (b->libass.layer == 1) {
            r->bord_color = b->libass.color;
            r->blur_b = b->libass.blur_x;
            MP_TARRAY_APPEND(tmp, r->bord, r->nbord, i);
        } else {
            r->fill_color = b->libass.color;
            r->blur_f = b->libass.blur_x;
            r->fill_color2 = b->libass.color2;
            r->wipe_x = b->libass.wipe_x;
            // KF_WIPE lives on the fill part; the region's run_flags came from
            // whichever part appeared first (the border), so OR it in here.
            r->run_flags |= b->libass.run_flags & RUN_FLAG_KF_WIPE;
            MP_TARRAY_APPEND(tmp, r->fill, r->nfill, i);
        }
    }

    if (!nregs) { talloc_free(tmp); return; }   // no deferred runs in this item

    // Pack the result atlas as wide as the GPU allows so it stays short enough
    // to fit max_tex_2d_dim in height (at 8K many wide runs stack very tall).
    const int AW = MPMIN(gpu->limits.max_tex_2d_dim, 16384);
    int shelf_x = 0, shelf_y = 0, shelf_h = 0, max_w = 0;
    int run_acc_w = 0, run_acc_h = 0;
    #define ALLOC_REGION(AX, AY, RW, RH) do {                               \
        if (shelf_x + (RW) > AW) { shelf_y += shelf_h; shelf_x = 0; shelf_h = 0; } \
        (AX) = shelf_x; (AY) = shelf_y; shelf_x += (RW);                    \
        shelf_h = MPMAX(shelf_h, (RH)); max_w = MPMAX(max_w, shelf_x);      \
        run_acc_w = MPMAX(run_acc_w, (RW)); run_acc_h = MPMAX(run_acc_h, (RH)); \
    } while (0)
    for (int i = 0; i < nregs; i++) {
        struct gc_region *r = &regs[i];
        // Deferred runs are raw (unexpanded) per-glyph coverage, so they need
        // the blur halo padding (libass's exact expand amount).
        r->margin = blur_expand_pad(MPMAX(r->blur_f, r->blur_b)) + r->be;
        int rw = r->x1 - r->x0 + 2 * r->margin, rh = r->y1 - r->y0 + 2 * r->margin;
        if (r->nfill) ALLOC_REGION(r->fill_ax, r->fill_ay, rw, rh);
        if (r->nbord) ALLOC_REGION(r->bord_ax, r->bord_ay, rw, rh);
    }
    int atlas_w = MPMAX(max_w, 1), atlas_h = MPMAX(shelf_y + shelf_h, 1);

    if (!gc_ensure(gpu, &entry->result_tex, r8, atlas_w, atlas_h, true, true, false, false, false) ||
        !gc_ensure(gpu, &p->result_acc, p->osd_acc_fmt, atlas_w, atlas_h, true, false, false, false, false) ||
        !gc_ensure(gpu, &p->result_tmp, r8, atlas_w, atlas_h, true, true, false, false, false)) {
        talloc_free(tmp);
        return;
    }

    // Batched back-half: instead of ~6 GPU dispatches PER region (which is
    // thousands on dense frames), combine every region's glyphs into its result
    // slot, then resolve / fix_outline / blur / clip in a handful of batched
    // dispatches over a slot-tile work-list.
    #define REG_BWBH struct gc_region *r = &regs[i]; \
        int bw = r->x1 - r->x0 + 2*r->margin, bh = r->y1 - r->y0 + 2*r->margin
    // 1. clear the accumulator. Single-part slots batch into ONE imageStore
    //    dispatch (no overlap). Multi-glyph runs get ONE per-run dispatch each
    //    that sums their glyphs per pixel (no atomics, no per-part dispatches).
    //    Both layouts share combwork_tex: [single-part tiles][run glyph entries].
    osd_clear(p, p->result_acc, atlas_w, atlas_h);
    // count single-part tiles + total multi-glyph parts (each = one tex pair).
    int ntiles_sp = 0, nglyph = 0;
    for (int i = 0; i < nregs; i++) {
        struct gc_region *r = &regs[i];
        int nf = r->nfill, nb = r->nbord;
        if (nf == 1) { const struct sub_bitmap *b = &item->parts[r->fill[0]];
            ntiles_sp += ((b->w+15)/16)*((b->h+15)/16); } else nglyph += nf;
        if (nb == 1) { const struct sub_bitmap *b = &item->parts[r->bord[0]];
            ntiles_sp += ((b->w+15)/16)*((b->h+15)/16); } else nglyph += nb;
    }
    int total_pairs = ntiles_sp + nglyph;
    if (total_pairs) {
        int cwh = (2*total_pairs + BLURWORK_W - 1) / BLURWORK_W;
        size_t need = (size_t) BLURWORK_W * cwh * 4;
        if ((size_t) p->cwbuf_cap < need) {
            p->cwbuf_cap = need; p->cwbuf = talloc_realloc(p, p->cwbuf, float, p->cwbuf_cap);
        }
        // [0, ntiles_sp): single-part tiles. [ntiles_sp, total): run glyph entries.
        int ci = 0;
        #define ADD_TILE(SAX, SAY, DX, DY, W, H) do {                          \
            int txs=((W)+15)/16, tys=((H)+15)/16;                              \
            for (int ty=0; ty<tys; ty++) for (int tx=0; tx<txs; tx++) {        \
                float *w0 = &p->cwbuf[(2*ci)*4];                              \
                w0[0]=(SAX); w0[1]=(SAY); w0[2]=tx*16; w0[3]=ty*16;            \
                w0[4]=(DX);  w0[5]=(DY);  w0[6]=(W);   w0[7]=(H); ci++; } } while (0)
        for (int i = 0; i < nregs; i++) {
            struct gc_region *r = &regs[i];
            if (r->nfill == 1) { const struct sub_bitmap *b = &item->parts[r->fill[0]];
                ADD_TILE(cpos[r->fill[0]].ax, cpos[r->fill[0]].ay,
                         r->fill_ax + b->x - r->x0 + r->margin,
                         r->fill_ay + b->y - r->y0 + r->margin, b->w, b->h); }
            if (r->nbord == 1) { const struct sub_bitmap *b = &item->parts[r->bord[0]];
                ADD_TILE(cpos[r->bord[0]].ax, cpos[r->bord[0]].ay,
                         r->bord_ax + b->x - r->x0 + r->margin,
                         r->bord_ay + b->y - r->y0 + r->margin, b->w, b->h); }
        }
        #undef ADD_TILE
        // glyph entries for multi-glyph runs; record each run's [gbase, gcount).
        int gi = ntiles_sp;
        #define ADD_GLYPH(SAX, SAY, DOX, DOY, W, H) do {                       \
            float *w0 = &p->cwbuf[(2*gi)*4];                                  \
            w0[0]=(SAX); w0[1]=(SAY); w0[2]=(DOX); w0[3]=(DOY);               \
            w0[4]=(W);   w0[5]=(H);   w0[6]=0;     w0[7]=0; gi++; } while (0)
        for (int i = 0; i < nregs; i++) {
            struct gc_region *r = &regs[i];
            if (r->nfill > 1) { r->fill_gbase = gi; r->fill_gcount = r->nfill;
                for (int k = 0; k < r->nfill; k++) { const struct sub_bitmap *b = &item->parts[r->fill[k]];
                    ADD_GLYPH(cpos[r->fill[k]].ax, cpos[r->fill[k]].ay,
                              b->x - r->x0 + r->margin, b->y - r->y0 + r->margin, b->w, b->h); } }
            if (r->nbord > 1) { r->bord_gbase = gi; r->bord_gcount = r->nbord;
                for (int k = 0; k < r->nbord; k++) { const struct sub_bitmap *b = &item->parts[r->bord[k]];
                    ADD_GLYPH(cpos[r->bord[k]].ax, cpos[r->bord[k]].ay,
                              b->x - r->x0 + r->margin, b->y - r->y0 + r->margin, b->w, b->h); } }
        }
        #undef ADD_GLYPH
        pl_fmt ef = pl_find_named_fmt(gpu, "rgba32f");
        if (ef && gc_ensure(gpu, &p->combwork_tex, ef, BLURWORK_W, cwh, false, true, false, false, true)) {
            pl_tex_upload(gpu, pl_tex_transfer_params(.tex = p->combwork_tex,
                .rc = { .x1 = BLURWORK_W, .y1 = cwh },
                .row_pitch = (size_t) BLURWORK_W * 4 * sizeof(float), .ptr = p->cwbuf));
            if (ntiles_sp)
                gc_combine_batch(p, p->glyph_atlas, p->result_acc, ntiles_sp);
            for (int i = 0; i < nregs; i++) {
                REG_BWBH;
                if (r->nfill > 1)
                    gc_combine_run(p, p->glyph_atlas, p->result_acc, r->fill_ax, r->fill_ay,
                                   bw, bh, r->fill_gbase, r->fill_gcount);
                if (r->nbord > 1)
                    gc_combine_run(p, p->glyph_atlas, p->result_acc, r->bord_ax, r->bord_ay,
                                   bw, bh, r->bord_gbase, r->bord_gcount);
            }
        }
    }
    // 2. resolve the whole accumulator to r8 coverage in one dispatch.
    osd_unop(p, p->result_acc, entry->result_tex, atlas_w, atlas_h,
             "acc", false, "dst", osd_resolve_body);
    // 3. fix_outline per bordered region (few), on the result slots.
    for (int i = 0; i < nregs; i++) {
        REG_BWBH;
        if (r->nfill && r->nbord && (r->run_flags & 1))
            gc_fixoutline_slot(p, entry->result_tex, r->bord_ax, r->bord_ay,
                               r->fill_ax, r->fill_ay, bw, bh);
    }
    // 4. batched separable blur: one slot-tile work-list, two dispatches.
    int ntiles = 0;
    for (int i = 0; i < nregs; i++) {
        REG_BWBH; int tx = (bw+15)/16, ty = (bh+15)/16;
        if (r->nfill) ntiles += tx*ty;
        if (r->nbord) ntiles += tx*ty;
    }
    if (ntiles) {
        int bwh = (2*ntiles + BLURWORK_W - 1) / BLURWORK_W;
        size_t need = (size_t) BLURWORK_W * bwh * 4;
        if ((size_t) p->bwbuf_cap < need) {
            p->bwbuf_cap = need; p->bwbuf = talloc_realloc(p, p->bwbuf, float, p->bwbuf_cap);
        }
        int ti = 0;
        #define ADD_SLOT(AX, AY, SG) do { int tx=(bw+15)/16, ty=(bh+15)/16; \
            for (int yy=0; yy<ty; yy++) for (int xx=0; xx<tx; xx++) { \
                float *w0 = &p->bwbuf[(2*ti)*4]; \
                w0[0]=(AX); w0[1]=(AY); w0[2]=xx*16; w0[3]=yy*16; \
                w0[4]=bw; w0[5]=bh; w0[6]=(SG); w0[7]=0; ti++; } } while (0)
        for (int i = 0; i < nregs; i++) {
            REG_BWBH;
            if (r->nfill) ADD_SLOT(r->fill_ax, r->fill_ay, r->blur_f);
            if (r->nbord) ADD_SLOT(r->bord_ax, r->bord_ay, r->blur_b);
        }
        #undef ADD_SLOT
        pl_fmt ef = pl_find_named_fmt(gpu, "rgba32f");
        if (ef && gc_ensure(gpu, &p->blurwork_tex, ef, BLURWORK_W, bwh, false, true, false, false, true)) {
            pl_tex_upload(gpu, pl_tex_transfer_params(.tex = p->blurwork_tex,
                .rc = { .x1 = BLURWORK_W, .y1 = bwh },
                .row_pitch = (size_t) BLURWORK_W * 4 * sizeof(float), .ptr = p->bwbuf));
            gc_blur_batch(p, entry->result_tex, p->result_tmp, ntiles, osd_blur_batch_h);
            gc_blur_batch(p, p->result_tmp, entry->result_tex, ntiles, osd_blur_batch_v);
        }
    }
    // 4b. \be edge-blur per be-region, on the result slots (after the gaussian,
    // matching libass's gaussian-then-be order). Rare, so per-region not batched.
    for (int i = 0; i < nregs; i++) {
        REG_BWBH;
        if (r->be <= 0) continue;
        if (r->nfill) gc_be_blur(p, entry->result_tex, p->result_tmp, r->fill_ax, r->fill_ay, bw, bh, r->be);
        if (r->nbord) gc_be_blur(p, entry->result_tex, p->result_tmp, r->bord_ax, r->bord_ay, bw, bh, r->be);
    }
    // 5. vector clip per clipped region, on the result slots (after blur).
    for (int i = 0; i < nregs; i++) {
        REG_BWBH;
        if (!r->clip_id) continue;
        if (r->nbord) gc_apply_clip(p, entry->result_tex, bw, bh, r->bord_ax, r->bord_ay, r, clips, nclips);
        if (r->nfill) gc_apply_clip(p, entry->result_tex, bw, bh, r->fill_ax, r->fill_ay, r, clips, nclips);
    }
    #undef REG_BWBH

    // Emit one monochrome overlay preserving libass's z-order: regions in
    // emission order (each event's shadow region precedes its text region), and
    // within a region border under fill. (Global border-then-fill passes would
    // mislayer overlapping shadowed events -- the shadow must stay behind its
    // own border yet a later event's shadow above an earlier event's fill.)
    entry->num_run_parts = 0;
    for (int i = 0; i < nregs; i++) {
        struct gc_region *r = &regs[i];
        bool is_shadow = r->run_flags & RUN_FLAG_SHADOW;
        int dx0 = r->x0 - r->margin, dy0 = r->y0 - r->margin;
        int dx1 = r->x1 + r->margin, dy1 = r->y1 + r->margin;
        // Rectangular \clip on the screen dst (no shader, a plain rect crop):
        // normal \clip keeps one visible rect (the intersection); inverse
        // \iclip subtracts the excluded rect, leaving up to 4 visible strips.
        int vr[4][4]; int nvr = 0;
        if (r->run_flags & RUN_FLAG_RECT_INVERSE) {
            int ex0 = MPMAX(dx0, r->rcx0), ey0 = MPMAX(dy0, r->rcy0);
            int ex1 = MPMIN(dx1, r->rcx1), ey1 = MPMIN(dy1, r->rcy1);
            if (ex0 >= ex1 || ey0 >= ey1) {     // dst clear of the hole
                vr[nvr][0]=dx0; vr[nvr][1]=dy0; vr[nvr][2]=dx1; vr[nvr][3]=dy1; nvr++;
            } else {
                if (dy0 < ey0) { vr[nvr][0]=dx0; vr[nvr][1]=dy0; vr[nvr][2]=dx1; vr[nvr][3]=ey0; nvr++; }
                if (ey1 < dy1) { vr[nvr][0]=dx0; vr[nvr][1]=ey1; vr[nvr][2]=dx1; vr[nvr][3]=dy1; nvr++; }
                if (dx0 < ex0) { vr[nvr][0]=dx0; vr[nvr][1]=ey0; vr[nvr][2]=ex0; vr[nvr][3]=ey1; nvr++; }
                if (ex1 < dx1) { vr[nvr][0]=ex1; vr[nvr][1]=ey0; vr[nvr][2]=dx1; vr[nvr][3]=ey1; nvr++; }
            }
        } else {
            int cx0 = MPMAX(dx0, r->rcx0), cy0 = MPMAX(dy0, r->rcy0);
            int cx1 = MPMIN(dx1, r->rcx1), cy1 = MPMIN(dy1, r->rcy1);
            if (cx0 < cx1 && cy0 < cy1) { vr[0][0]=cx0; vr[0][1]=cy0; vr[0][2]=cx1; vr[0][3]=cy1; nvr=1; }
        }
        // border (under) then fill (over); a shadow region is fill-only.
        for (int layer = 0; layer < 2; layer++) {
            int ax, ay; uint32_t c;
            if (layer == 0) { if (!r->nbord) continue; ax = r->bord_ax; ay = r->bord_ay; c = r->bord_color; }
            else            { if (!r->nfill) continue; ax = r->fill_ax; ay = r->fill_ay; c = r->fill_color; }
            for (int v = 0; v < nvr; v++) {
                int cx0 = vr[v][0], cy0 = vr[v][1], cx1 = vr[v][2], cy1 = vr[v][3];
                // \kf karaoke: split the fill at wipe_x into sung (fill_color,
                // left) and unsung (fill_color2, right). Else one segment.
                int sx0[2] = { cx0, 0 }, sx1[2] = { cx1, 0 };
                uint32_t sc[2] = { c, 0 }; int nseg = 1;
                if (layer == 1 && !is_shadow && (r->run_flags & RUN_FLAG_KF_WIPE)) {
                    int w = MPMAX(cx0, MPMIN(cx1, r->wipe_x));
                    sx0[0] = cx0; sx1[0] = w;   sc[0] = r->fill_color;
                    sx0[1] = w;   sx1[1] = cx1; sc[1] = r->fill_color2;
                    nseg = 2;
                }
                for (int s = 0; s < nseg; s++) {
                    if (sx0[s] >= sx1[s]) continue;
                    uint32_t sg = sc[s];
                    struct pl_overlay_part part = {
                        .src = { ax + (sx0[s] - dx0), ay + (cy0 - dy0),
                                 ax + (sx1[s] - dx0), ay + (cy1 - dy0) },
                        .dst = { sx0[s], cy0, sx1[s], cy1 },
                        .color = { (sg >> 24) / 255.0f, ((sg >> 16) & 0xFF) / 255.0f,
                                   ((sg >> 8) & 0xFF) / 255.0f, (255 - (sg & 0xFF)) / 255.0f },
                    };
                    MP_TARRAY_APPEND(p, entry->run_parts, entry->num_run_parts, part);
                }
            }
        }
    }
    talloc_free(tmp);

    struct pl_overlay *ol = &state->overlays[frame->num_overlays++];
    *ol = (struct pl_overlay){
        .tex = entry->result_tex, .parts = entry->run_parts,
        .num_parts = entry->num_run_parts,
        .mode = PL_OVERLAY_MONOCHROME, .coords = coords,
        .color = pl_color_space_srgb, .repr.alpha = PL_ALPHA_INDEPENDENT,
    };
    if (src && item->video_color_space && !pl_color_space_is_hdr(&src->params.color))
        ol->color = src->params.color;
}

static void update_overlays(struct vo *vo, struct mp_osd_res res,
                            int flags, enum pl_overlay_coords coords,
                            struct osd_state *state, struct pl_frame *frame,
                            struct mp_image *src, int stereo_mode)
{
    struct priv *p = vo->priv;
    double pts = src ? src->pts : 0;
    int div[2];
    mp_get_3d_side_by_side(stereo_mode, div);
    res.w /= div[0];
    res.h /= div[1];
    // Advertise the deferred-composite format too; sd_ass only emits it when
    // --sub-gpu-composite is set, otherwise it falls back to plain LIBASS.
    static const bool gpu_sub_formats[SUBBITMAP_COUNT] = {
        [SUBBITMAP_LIBASS] = true, [SUBBITMAP_BGRA] = true,
        [SUBBITMAP_LIBASS_GLYPHS] = true, [SUBBITMAP_LIBASS_OUTLINES] = true,
    };
    struct sub_bitmap_list *subs = osd_render(vo->osd, res, pts, flags, gpu_sub_formats);

    frame->overlays = state->overlays;
    frame->num_overlays = 0;

    // --- TEMP instrumentation: split the overlay cost (upload/blur/total) ---
    int64_t dbg_t0 = mp_time_ns();
    int64_t dbg_upload = 0, dbg_blur = 0;
    int64_t dbg_area = 0, dbg_parts = 0, dbg_aw = 0, dbg_ah = 0;

    for (int n = 0; n < subs->num_items; n++) {
        const struct sub_bitmaps *item = subs->items[n];
        // Outline mode has no packed atlas (the GPU rasterizes from segments).
        if (!item->num_parts ||
            (!item->packed && item->format != SUBBITMAP_LIBASS_OUTLINES))
            continue;
        struct osd_entry *entry = &state->entries[item->render_index];
        if (item->format == SUBBITMAP_LIBASS_OUTLINES) {
            compose_glyph_runs(p, item, entry, frame, state, coords, src);
            continue;   // no legacy fallback path (would need a packed atlas)
        }
        if (item->format == SUBBITMAP_LIBASS_GLYPHS) {
            compose_glyph_runs(p, item, entry, frame, state, coords, src);
            // Already-combined fallback parts (shadow/karaoke runs, glyph_id 0)
            // go through the legacy path below; skip it if there are none.
            bool has_fallback = false;
            for (int i = 0; i < item->num_parts; i++)
                if (item->parts[i].libass.glyph_id == 0) { has_fallback = true; break; }
            if (!has_fallback)
                continue;
        }
        pl_fmt tex_fmt = p->osd_fmt[item->format];
        if (!entry->tex)
            MP_TARRAY_POP(p->sub_tex, p->num_sub_tex, &entry->tex);
        // Round the OSD texture up and grow it monotonically so it isn't
        // reallocated every frame as the atlas grows through a dense scene
        // (each realloc stalls the display thread).
        int want_w = (item->packed_w + 255) & ~255;
        int want_h = (item->packed_h + 255) & ~255;
        bool ok = pl_tex_recreate(p->gpu, &entry->tex, &(struct pl_tex_params) {
            .format = tex_fmt,
            .w = MPMAX(want_w, entry->tex ? entry->tex->params.w : 0),
            .h = MPMAX(want_h, entry->tex ? entry->tex->params.h : 0),
            .host_writable = true,
            .sampleable = true,
        });
        if (!ok) {
            MP_ERR(vo, "Failed recreating OSD texture!\n");
            break;
        }
        struct pl_tex_transfer_params upload_params = {
            .tex        = entry->tex,
            .rc         = { .x1 = item->packed_w, .y1 = item->packed_h, },
            .row_pitch  = item->packed->stride[0],
        };
        // Upload the (large, per-frame) subtitle atlas via a streaming buffer:
        // libplacebo guarantees buffer transfers are asynchronous even on
        // backends without upload callbacks (e.g. d3d11), so this no longer
        // blocks the VO thread the way a synchronous `ptr` upload does.
        size_t buf_size = (size_t) upload_params.row_pitch * item->packed_h;
        pl_buf *ring = &p->overlay_bufs[p->overlay_buf_idx++ % NUM_OVERLAY_BUFS];
        int64_t dbg_u0 = mp_time_ns();
        // Reuse the staging buffer whenever it's already big enough; only
        // (re)allocate on growth, rounded up, so it stops being reallocated as
        // the atlas grows frame-to-frame through a dense scene (that realloc was
        // the ~187ms VO-thread stall).
        bool buf_ok = (*ring) && (*ring)->params.size >= buf_size;
        if (!buf_ok) {
            size_t want = (buf_size + (4u << 20) - 1) & ~(size_t)((4u << 20) - 1);
            buf_ok = pl_buf_recreate(p->gpu, ring,
                                     pl_buf_params(.size = want, .host_writable = true));
        }
        if (buf_ok)
        {
            pl_buf_write(p->gpu, *ring, 0, item->packed->planes[0], buf_size);
            upload_params.buf = *ring;
            ok = pl_tex_upload(p->gpu, &upload_params);
        } else {
            // Fallback to a direct upload if the buffer can't be allocated.
            upload_params.ptr = item->packed->planes[0];
            if (p->gpu->limits.callbacks) {
                upload_params.callback = talloc_free;
                upload_params.priv = mp_image_new_ref(item->packed);
            }
            ok = pl_tex_upload(p->gpu, &upload_params);
        }
        dbg_upload += mp_time_ns() - dbg_u0;
        dbg_aw = MPMAX(dbg_aw, item->packed_w); dbg_ah = MPMAX(dbg_ah, item->packed_h);
        for (int i = 0; i < item->num_parts; i++) { dbg_parts++;
            dbg_area += (int64_t)item->parts[i].w * item->parts[i].h; }
        if (!ok) {
            MP_ERR(vo, "Failed uploading OSD texture!\n");
            talloc_free(upload_params.priv);
            break;
        }

        entry->num_parts = 0;
        for (int i = 0; i < item->num_parts; i++) {
            const struct sub_bitmap *b = &item->parts[i];
            if (b->dw == 0 || b->dh == 0)
                continue;
            if (item->format == SUBBITMAP_LIBASS_GLYPHS && b->libass.glyph_id)
                continue;   // deferred runs are handled by compose_glyph_runs
            uint32_t c = b->libass.color;
            struct pl_overlay_part part = {
                .src = { b->src_x, b->src_y, b->src_x + b->w, b->src_y + b->h },
                .dst = { b->x, b->y, b->x + b->dw, b->y + b->dh },
                .color = {
                    (c >> 24) / 255.0f,
                    ((c >> 16) & 0xFF) / 255.0f,
                    ((c >> 8) & 0xFF) / 255.0f,
                    (255 - (c & 0xFF)) / 255.0f,
                }
            };
            MP_TARRAY_APPEND(p, entry->parts, entry->num_parts, part);
        }

        // Deferred-blur (see ass_set_blur_deferred): libass emitted unblurred
        // coverage in pre-expanded bounds plus a per-part gaussian std-dev; do
        // the blur here on the GPU instead of on the CPU display path.
        pl_tex overlay_tex = entry->tex;
        if (item->format == SUBBITMAP_LIBASS ||
            item->format == SUBBITMAP_LIBASS_GLYPHS) {
            bool any_blur = false;
            for (int i = 0; i < item->num_parts; i++) {
                if (item->format == SUBBITMAP_LIBASS_GLYPHS && item->parts[i].libass.glyph_id)
                    continue;   // deferred parts blurred by compose_glyph_runs
                if (item->parts[i].libass.blur_x > 0 || item->parts[i].libass.blur_y > 0) {
                    any_blur = true;
                    break;
                }
            }
            if (any_blur && !(tex_fmt->caps & PL_FMT_CAP_STORABLE)) {
                if (!p->osd_blur_unsupported) {
                    MP_WARN(vo, "GPU subtitle blur needs a storable OSD format; "
                               "falling back to unblurred. Disable --sub-gpu-blur.\n");
                    p->osd_blur_unsupported = true;
                }
                any_blur = false;
            }
            if (any_blur) {
                int tw = entry->tex->params.w, th = entry->tex->params.h;
                struct pl_tex_params bp = { .format=tex_fmt, .w=tw, .h=th,
                                            .sampleable=true, .storable=true };
                int64_t dbg_b0 = mp_time_ns();
                if (!entry->blur_tex)
                    MP_TARRAY_POP(p->sub_scratch, p->num_sub_scratch, &entry->blur_tex);
                if (!entry->tmp_tex)
                    MP_TARRAY_POP(p->sub_scratch, p->num_sub_scratch, &entry->tmp_tex);
                if (pl_tex_recreate(p->gpu, &entry->blur_tex, &bp) &&
                    pl_tex_recreate(p->gpu, &entry->tmp_tex, &bp))
                {
                    for (int i = 0; i < item->num_parts; i++) {
                        const struct sub_bitmap *b = &item->parts[i];
                        if (b->w < 1 || b->h < 1)
                            continue;
                        if (item->format == SUBBITMAP_LIBASS_GLYPHS && b->libass.glyph_id)
                            continue;
                        // separable: H with blur_x (atlas->tmp), V with blur_y (tmp->blur)
                        osd_blur_part(p, entry->tex, entry->tmp_tex,
                                      b->src_x, b->src_y, b->w, b->h,
                                      b->libass.blur_x, osd_blur_body_h);
                        osd_blur_part(p, entry->tmp_tex, entry->blur_tex,
                                      b->src_x, b->src_y, b->w, b->h,
                                      b->libass.blur_y, osd_blur_body_v);
                    }
                    overlay_tex = entry->blur_tex;
                }
                dbg_blur += mp_time_ns() - dbg_b0;
            }
        }

        struct pl_overlay *ol = &state->overlays[frame->num_overlays++];
        *ol = (struct pl_overlay) {
            .tex = overlay_tex,
            .parts = entry->parts,
            .num_parts = entry->num_parts,
            .color = pl_color_space_srgb,
            .coords = coords,
        };

        switch (item->format) {
        case SUBBITMAP_BGRA:
            ol->mode = PL_OVERLAY_NORMAL;
            ol->repr.alpha = PL_ALPHA_PREMULTIPLIED;
            // Infer bitmap colorspace from source
            if (src) {
                ol->color = src->params.color;
                if (pl_color_transfer_is_hdr(ol->color.transfer)) {
                    bool use_static = p->next_opts->image_subs_hdr_peak == -2;
                    if (use_static || p->next_opts->image_subs_hdr_peak == -3) {
                        float max;
                        pl_color_space_nominal_luma_ex(pl_nominal_luma_params(
                            .color      = &ol->color,
                            .metadata   = use_static ? PL_HDR_METADATA_HDR10 : PL_HDR_METADATA_ANY,
                            .scaling    = PL_HDR_NITS,
                            .out_max    = &max,
                        ));
                        ol->color.hdr = (struct pl_hdr_metadata) {
                            .max_luma = max,
                        };
                    } else if (p->next_opts->image_subs_hdr_peak != -1) {
                        ol->color.hdr = (struct pl_hdr_metadata) {
                            .max_luma = p->next_opts->image_subs_hdr_peak,
                        };
                    }
                }
            }
            break;
        case SUBBITMAP_LIBASS_GLYPHS:   // the fallback singletons, same as LIBASS
        case SUBBITMAP_LIBASS:
            if (src && item->video_color_space && !pl_color_space_is_hdr(&src->params.color))
                ol->color = src->params.color;
            if (src && pl_color_transfer_is_hdr(frame->color.transfer)) {
                ol->color.hdr = (struct pl_hdr_metadata) {
                    .max_luma = p->next_opts->sub_hdr_peak,
                };
            }
            ol->mode = PL_OVERLAY_MONOCHROME;
            ol->repr.alpha = PL_ALPHA_INDEPENDENT;
            break;
        }

        // Duplicate overlay parts for each eye in stereo 3D modes
        if (div[0] > 1 || div[1] > 1) {
            int orig_num = entry->num_parts;
            for (int x = 0; x < div[0]; x++) {
                for (int y = 0; y < div[1]; y++) {
                    if (x == 0 && y == 0)
                        continue;
                    float off_x = res.w * x;
                    float off_y = res.h * y;
                    for (int i = 0; i < orig_num; i++) {
                        struct pl_overlay_part duped = entry->parts[i];
                        duped.dst.x0 += off_x;
                        duped.dst.x1 += off_x;
                        duped.dst.y0 += off_y;
                        duped.dst.y1 += off_y;
                        MP_TARRAY_APPEND(p, entry->parts, entry->num_parts, duped);
                    }
                }
            }
            ol->parts = entry->parts;
            ol->num_parts = entry->num_parts;
        }
    }

    // --- overlay timing emit (only for non-trivial overlay frames) ---
    int64_t dbg_total = mp_time_ns() - dbg_t0;
    if (dbg_total > 2000000) {
        MP_STATS(vo, "value %f osd-total-ms",  dbg_total  / 1e6);
        MP_STATS(vo, "value %f osd-upload-ms", dbg_upload / 1e6);
        MP_STATS(vo, "value %f osd-blur-ms",   dbg_blur   / 1e6);
        MP_STATS(vo, "value %f osd-parts",     (double) dbg_parts);
        MP_STATS(vo, "value %f osd-atlas-mpx", (double) dbg_aw * dbg_ah / 1e6);
    }

    talloc_free(subs);
}

struct frame_priv {
    struct vo *vo;
    struct osd_state subs;
    uint64_t osd_sync;
    struct ra_hwdec *hwdec;
};

static int plane_data_from_imgfmt(struct pl_plane_data out_data[4],
                                  struct pl_bit_encoding *out_bits,
                                  enum mp_imgfmt imgfmt, bool use_uint)
{
    struct mp_imgfmt_desc desc = mp_imgfmt_get_desc(imgfmt);
    if (!desc.num_planes || !(desc.flags & MP_IMGFLAG_HAS_COMPS))
        return 0;

    if (desc.flags & MP_IMGFLAG_HWACCEL)
        return 0; // HW-accelerated frames need to be mapped differently

    if (!(desc.flags & MP_IMGFLAG_NE))
        return 0; // GPU endianness follows the host's

    if (desc.flags & MP_IMGFLAG_PAL)
        return 0; // Palette formats (currently) not supported in libplacebo

    if ((desc.flags & MP_IMGFLAG_TYPE_FLOAT) && (desc.flags & MP_IMGFLAG_YUV))
        return 0; // Floating-point YUV (currently) unsupported

    bool has_bits = false;
    bool any_padded = false;

    for (int p = 0; p < desc.num_planes; p++) {
        struct pl_plane_data *data = &out_data[p];
        struct mp_imgfmt_comp_desc sorted[MP_NUM_COMPONENTS];
        int num_comps = 0;
        if (desc.bpp[p] % 8)
            return 0; // Pixel size is not byte-aligned

        for (int c = 0; c < mp_imgfmt_desc_get_num_comps(&desc); c++) {
            if (desc.comps[c].plane != p)
                continue;

            data->component_map[num_comps] = c;
            sorted[num_comps] = desc.comps[c];
            num_comps++;

            // Sort components by offset order, while keeping track of the
            // semantic mapping in `data->component_map`
            for (int i = num_comps - 1; i > 0; i--) {
                if (sorted[i].offset >= sorted[i - 1].offset)
                    break;
                MPSWAP(struct mp_imgfmt_comp_desc, sorted[i], sorted[i - 1]);
                MPSWAP(int, data->component_map[i], data->component_map[i - 1]);
            }
        }

        uint64_t total_bits = 0;

        // Fill in the pl_plane_data fields for each component
        memset(data->component_size, 0, sizeof(data->component_size));
        for (int c = 0; c < num_comps; c++) {
            data->component_size[c] = sorted[c].size;
            data->component_pad[c] = sorted[c].offset - total_bits;
            total_bits += data->component_pad[c] + data->component_size[c];
            any_padded |= sorted[c].pad;

            // Ignore bit encoding of alpha channel
            if (!out_bits || data->component_map[c] == PL_CHANNEL_A)
                continue;

            struct pl_bit_encoding bits = {
                .sample_depth = data->component_size[c],
                .color_depth = sorted[c].size - abs(sorted[c].pad),
                .bit_shift = MPMAX(sorted[c].pad, 0),
            };

            if (!has_bits) {
                *out_bits = bits;
                has_bits = true;
            } else {
                if (!pl_bit_encoding_equal(out_bits, &bits)) {
                    // Bit encoding differs between components/planes,
                    // cannot handle this
                    *out_bits = (struct pl_bit_encoding) {0};
                    out_bits = NULL;
                }
            }
        }

        data->pixel_stride = desc.bpp[p] / 8;
        data->type = (desc.flags & MP_IMGFLAG_TYPE_FLOAT)
                            ? PL_FMT_FLOAT
                            : (use_uint ? PL_FMT_UINT : PL_FMT_UNORM);
    }

    if (any_padded && !out_bits)
        return 0; // can't handle padded components without `pl_bit_encoding`

    return desc.num_planes;
}

static bool hwdec_reconfig(struct priv *p, struct ra_hwdec *hwdec,
                           const struct mp_image_params *par)
{
    if (p->hwdec_mapper) {
        if (mp_image_params_static_equal(par, &p->hwdec_mapper->src_params)) {
            p->hwdec_mapper->src_params.repr.dovi = par->repr.dovi;
            p->hwdec_mapper->dst_params.repr.dovi = par->repr.dovi;
            p->hwdec_mapper->src_params.color.hdr = par->color.hdr;
            p->hwdec_mapper->dst_params.color.hdr = par->color.hdr;
            return p->hwdec_mapper;
        } else {
            ra_hwdec_mapper_free(&p->hwdec_mapper);
            timer_pool_destroy(p->hwdec_timer);
            p->hwdec_timer = NULL;
        }
    }

    p->hwdec_mapper = ra_hwdec_mapper_create(hwdec, par);
    if (!p->hwdec_mapper) {
        MP_ERR(p, "Initializing texture for hardware decoding failed.\n");
        return NULL;
    }
    p->hwdec_timer = timer_pool_create(p->ra_ctx->ra);

    return p->hwdec_mapper;
}

// For RAs not based on ra_pl, this creates a new pl_tex wrapper
static pl_tex hwdec_get_tex(struct priv *p, int n)
{
    struct ra_tex *ratex = p->hwdec_mapper->tex[n];
    struct ra *ra = p->hwdec_mapper->ra;
    if (ra_pl_get(ra))
        return (pl_tex) ratex->priv;

#if HAVE_GL && defined(PL_HAVE_OPENGL)
    if (ra_is_gl(ra) && pl_opengl_get(p->gpu)) {
        struct pl_opengl_wrap_params par = {
            .width = ratex->params.w,
            .height = ratex->params.h,
        };

        ra_gl_get_format(ratex->params.format, &par.iformat,
                         &(GLenum){0}, &(GLenum){0});
        ra_gl_get_raw_tex(ra, ratex, &par.texture, &par.target);
        return pl_opengl_wrap(p->gpu, &par);
    }
#endif

#if HAVE_D3D11 && defined(PL_HAVE_D3D11)
    if (ra_is_d3d11(ra)) {
        int array_slice = 0;
        ID3D11Resource *res = ra_d3d11_get_raw_tex(ra, ratex, &array_slice);
        pl_tex tex = pl_d3d11_wrap(p->gpu, pl_d3d11_wrap_params(
            .tex = res,
            .array_slice = array_slice,
            .fmt = ra_d3d11_get_format(ratex->params.format),
            .w = ratex->params.w,
            .h = ratex->params.h,
        ));
        SAFE_RELEASE(res);
        return tex;
    }
#endif

    MP_ERR(p, "Failed mapping hwdec frame? Open a bug!\n");
    return NULL;
}

static bool hwdec_acquire(pl_gpu gpu, struct pl_frame *frame)
{
    struct mp_image *mpi = frame->user_data;
    struct frame_priv *fp = mpi->priv;
    struct priv *p = fp->vo->priv;
    if (!hwdec_reconfig(p, fp->hwdec, &mpi->params))
        return false;

    stats_time_start(p->stats, "hwdec-map");
    timer_pool_start(p->hwdec_timer);
    if (ra_hwdec_mapper_map(p->hwdec_mapper, mpi) < 0) {
        MP_ERR(p, "Mapping hardware decoded surface failed.\n");
        timer_pool_stop(p->hwdec_timer);
        stats_time_end(p->stats, "hwdec-map");
        return false;
    }

    for (int n = 0; n < frame->num_planes; n++) {
        if (!(frame->planes[n].texture = hwdec_get_tex(p, n))) {
            timer_pool_stop(p->hwdec_timer);
            stats_time_end(p->stats, "hwdec-map");
            return false;
        }
    }

    timer_pool_stop(p->hwdec_timer);
    p->hwdec_perf = timer_pool_measure(p->hwdec_timer);
    stats_time_end(p->stats, "hwdec-map");

    return true;
}

static void hwdec_release(pl_gpu gpu, struct pl_frame *frame)
{
    struct mp_image *mpi = frame->user_data;
    struct frame_priv *fp = mpi->priv;
    struct priv *p = fp->vo->priv;
    if (!ra_pl_get(p->hwdec_mapper->ra)) {
        for (int n = 0; n < frame->num_planes; n++)
            pl_tex_destroy(p->gpu, &frame->planes[n].texture);
    }

    ra_hwdec_mapper_unmap(p->hwdec_mapper);
}

static bool format_supported(struct vo *vo, int format, bool use_uint)
{
    struct priv *p = vo->priv;
    struct pl_bit_encoding bits;
    struct pl_plane_data data[4] = {0};
    int planes = plane_data_from_imgfmt(data, &bits, format, use_uint);
    if (!planes)
        return false;

    for (int i = 0; i < planes; i++) {
        if (!pl_plane_find_fmt(p->gpu, NULL, &data[i]))
            return false;
    }

    return true;
}

static bool map_frame(pl_gpu gpu, pl_tex *tex, const struct pl_source_frame *src,
                      struct pl_frame *frame)
{
    struct mp_image *mpi = src->frame_data;
    struct mp_image_params par = mpi->params;
    struct frame_priv *fp = mpi->priv;
    struct vo *vo = fp->vo;
    struct priv *p = vo->priv;

    fp->hwdec = ra_hwdec_get(&p->hwdec_ctx, mpi->imgfmt);
    if (fp->hwdec) {
        // Note: We don't actually need the mapper to map the frame yet, we
        // only reconfig the mapper here (potentially creating it) to access
        // `dst_params`. In practice, though, this should not matter unless the
        // image format changes mid-stream.
        if (!hwdec_reconfig(p, fp->hwdec, &mpi->params)) {
            talloc_free(mpi);
            return false;
        }

        par = p->hwdec_mapper->dst_params;
    }

    mp_image_params_guess_csp(&par);

    *frame = (struct pl_frame) {
        .color = par.color,
        .repr = par.repr,
        .profile = {
            .data = mpi->icc_profile ? mpi->icc_profile->data : NULL,
            .len = mpi->icc_profile ? mpi->icc_profile->size : 0,
        },
        .rotation = par.rotate / 90,
        .user_data = mpi,
    };

    const struct gl_video_opts *opts = p->opts_cache->opts;
    if (opts->hdr_reference_white && !pl_color_transfer_is_hdr(frame->color.transfer))
        frame->color.hdr.max_luma = opts->hdr_reference_white;


    if (opts->treat_srgb_as_power22 & 1 && frame->color.transfer == PL_COLOR_TRC_SRGB) {
        // The sRGB EOTF is a pure gamma 2.2 function. See reference display in
        // IEC 61966-2-1-1999. Linearize sRGB to display light.
        frame->color.transfer = PL_COLOR_TRC_GAMMA22;
    }

    if (fp->hwdec) {
        p->sw_upload_perf.count = 0;

        struct mp_imgfmt_desc desc = mp_imgfmt_get_desc(par.imgfmt);
        frame->acquire = hwdec_acquire;
        frame->release = hwdec_release;
        frame->num_planes = desc.num_planes;
        for (int n = 0; n < frame->num_planes; n++) {
            struct pl_plane *plane = &frame->planes[n];
            int *map = plane->component_mapping;
            for (int c = 0; c < mp_imgfmt_desc_get_num_comps(&desc); c++) {
                if (desc.comps[c].plane != n)
                    continue;

                // Sort by component offset
                uint8_t offset = desc.comps[c].offset;
                int index = plane->components++;
                while (index > 0 && desc.comps[map[index - 1]].offset > offset) {
                    map[index] = map[index - 1];
                    index--;
                }
                map[index] = c;
            }
        }

    } else { // swdec
        p->hwdec_perf.count = 0;

        if (!p->sw_upload_timer)
            p->sw_upload_timer = timer_pool_create(p->ra_ctx->ra);

        struct pl_plane_data data[4] = {0};
        bool use_uint = false;

        // At this point, we know that the format is supported, query_format()
        // makes sure of that. Just check if we should use UINT as a fallback.
        if (!format_supported(vo, mpi->imgfmt, false))
            use_uint = true;

        frame->num_planes = plane_data_from_imgfmt(data, &frame->repr.bits, mpi->imgfmt, use_uint);
        stats_time_start(p->stats, "swdec-upload");
        timer_pool_start(p->sw_upload_timer);
        for (int n = 0; n < frame->num_planes; n++) {
            struct pl_plane *plane = &frame->planes[n];
            data[n].width = mp_image_plane_w(mpi, n);
            data[n].height = mp_image_plane_h(mpi, n);
            if (mpi->stride[n] < 0) {
                data[n].pixels = mpi->planes[n] + (data[n].height - 1) * mpi->stride[n];
                data[n].row_stride = -mpi->stride[n];
                plane->flipped = true;
            } else {
                data[n].pixels = mpi->planes[n];
                data[n].row_stride = mpi->stride[n];
            }

            pl_buf buf = get_dr_buf(p, data[n].pixels);
            if (buf) {
                data[n].buf = buf;
                data[n].buf_offset = (uint8_t *) data[n].pixels - buf->data;
                data[n].pixels = NULL;
            }
            // Keep the image alive until it's fully read.
            if (gpu->limits.callbacks) {
                mp_assert(!data[n].callback);
                data[n].callback = talloc_free;
                mp_assert(!data[n].priv);
                data[n].priv = mp_image_new_ref(mpi);
            }

            if (!pl_upload_plane(gpu, plane, &tex[n], &data[n])) {
                MP_ERR(vo, "Failed uploading frame!\n");
                timer_pool_stop(p->sw_upload_timer);
                stats_time_end(p->stats, "swdec-upload");
                talloc_free(data[n].priv);
                talloc_free(mpi);
                return false;
            }

            // Without async callback support, we have to poll...
            if (!gpu->limits.callbacks && data[n].buf)
                while (pl_buf_poll(gpu, data[n].buf, UINT64_MAX));
        }
        timer_pool_stop(p->sw_upload_timer);
        p->sw_upload_perf = timer_pool_measure(p->sw_upload_timer);
        stats_time_end(p->stats, "swdec-upload");

    }

    // Update chroma location, must be done after initializing planes
    pl_frame_set_chroma_location(frame, par.chroma_location);

    if (mpi->film_grain)
        pl_film_grain_from_av(&frame->film_grain, (AVFilmGrainParams *) mpi->film_grain->data);

    // Compute a unique signature for any attached ICC profile. Wasteful in
    // theory if the ICC profile is the same for multiple frames, but in
    // practice ICC profiles are overwhelmingly going to be attached to
    // still images so it shouldn't matter.
    pl_icc_profile_compute_signature(&frame->profile);

    // Update LUT attached to this frame
    update_lut(p, &p->next_opts->image_lut);
    frame->lut = p->next_opts->image_lut.lut;
    frame->lut_type = p->next_opts->image_lut.type;
    return true;
}

static void unmap_frame(pl_gpu gpu, struct pl_frame *frame,
                        const struct pl_source_frame *src)
{
    struct mp_image *mpi = src->frame_data;
    struct frame_priv *fp = mpi->priv;
    struct priv *p = fp->vo->priv;
    for (int i = 0; i < MP_ARRAY_SIZE(fp->subs.entries); i++) {
        pl_tex tex = fp->subs.entries[i].tex;
        if (tex)
            MP_TARRAY_APPEND(p, p->sub_tex, p->num_sub_tex, tex);
        pl_tex bl = fp->subs.entries[i].blur_tex;
        if (bl)
            MP_TARRAY_APPEND(p, p->sub_scratch, p->num_sub_scratch, bl);
        pl_tex tm = fp->subs.entries[i].tmp_tex;
        if (tm)
            MP_TARRAY_APPEND(p, p->sub_scratch, p->num_sub_scratch, tm);
    }
    talloc_free(mpi);
}

static void discard_frame(const struct pl_source_frame *src)
{
    struct mp_image *mpi = src->frame_data;
    talloc_free(mpi);
}

static void info_callback(void *priv, const struct pl_render_info *info)
{
    struct vo *vo = priv;
    struct priv *p = vo->priv;
    if (info->index >= VO_PASS_PERF_MAX)
        return; // silently ignore clipped passes, whatever

    struct frame_info *frame;
    switch (info->stage) {
    case PL_RENDER_STAGE_FRAME: frame = &p->perf_fresh; break;
    case PL_RENDER_STAGE_BLEND: frame = &p->perf_redraw; break;
    default: abort();
    }

    frame->count = info->index + 1;
    pl_dispatch_info_move(&frame->info[info->index], info->pass);
}

static void update_options(struct vo *vo)
{
    struct priv *p = vo->priv;
    pl_options pars = p->pars;
    bool changed = m_config_cache_update(p->opts_cache);
    changed = m_config_cache_update(p->next_opts_cache) || changed;
    if (changed)
        update_render_options(vo);

    update_lut(p, &p->next_opts->lut);
    pars->params.lut = p->next_opts->lut.lut;
    pars->params.lut_type = p->next_opts->lut.type;

    // Update equalizer state
    struct mp_csp_params cparams = MP_CSP_PARAMS_DEFAULTS;
    const struct gl_video_opts *opts = p->opts_cache->opts;
    mp_csp_equalizer_state_get(p->video_eq, &cparams);
    pars->color_adjustment.brightness = cparams.brightness;
    pars->color_adjustment.contrast = cparams.contrast;
    pars->color_adjustment.hue = cparams.hue;
    pars->color_adjustment.saturation = cparams.saturation;
    pars->color_adjustment.gamma = cparams.gamma * opts->gamma;
    p->output_levels = cparams.levels_out;

    for (char **kv = p->next_opts->raw_opts; kv && kv[0]; kv += 2)
        pl_options_set_str(pars, kv[0], kv[1]);
}

static void apply_target_contrast(struct priv *p, struct pl_color_space *color, float min_luma)
{
    const struct gl_video_opts *opts = p->opts_cache->opts;

    // Auto mode, use target value if available
    if (!opts->target_contrast) {
        color->hdr.min_luma = min_luma;
        return;
    }

    // Infinite contrast
    if (opts->target_contrast == -1) {
        color->hdr.min_luma = 1e-7;
        mp_assert(color->hdr.min_luma > 0);
        return;
    }

    // Infer max_luma for current pl_color_space
    pl_color_space_nominal_luma_ex(pl_nominal_luma_params(
        .color = color,
        // with HDR10 meta to respect value if already set
        .metadata = PL_HDR_METADATA_HDR10,
        .scaling = PL_HDR_NITS,
        .out_max = &color->hdr.max_luma
    ));

    color->hdr.min_luma = color->hdr.max_luma / opts->target_contrast;
}

static void apply_target_options(struct priv *p, struct pl_frame *target,
                                 float min_luma, bool hint)
{
    update_lut(p, &p->next_opts->target_lut);
    target->lut = p->next_opts->target_lut.lut;
    target->lut_type = p->next_opts->target_lut.type;

    // Colorspace overrides
    const struct gl_video_opts *opts = p->opts_cache->opts;
    // If swapchain returned a value use this, override is used in hint
    if (p->output_levels)
        target->repr.levels = p->output_levels;
    if (opts->target_prim && (!target->color.primaries || !hint))
        target->color.primaries = opts->target_prim;
    if (opts->target_trc && (!target->color.transfer || !hint))
        target->color.transfer = opts->target_trc;
    if (opts->target_peak && (!target->color.hdr.max_luma || !hint))
        target->color.hdr.max_luma = opts->target_peak;
    if (opts->hdr_reference_white && (!target->color.hdr.max_luma || !hint) &&
        !pl_color_transfer_is_hdr(target->color.transfer)) {
        target->color.hdr.max_luma = opts->hdr_reference_white;
    }
    if ((!target->color.hdr.min_luma || !hint))
        apply_target_contrast(p, &target->color, min_luma);
    if (opts->target_gamut)
        mp_parse_raw_primaries(mp_null_log, opts->target_gamut, &target->color.hdr.prim);
    int dither_depth = opts->dither_depth;
    if (dither_depth == 0) {
        struct ra_swapchain *sw = p->ra_ctx->swapchain;
        dither_depth = sw->fns->color_depth ? sw->fns->color_depth(sw) : 0;
    }
#if PL_API_VER >= 362
    // Don't dither scRGB, assume downstream will handle quantization properly.
    if (target->color.transfer == PL_COLOR_TRC_SCRGB)
        dither_depth = -1;
#endif
    if (dither_depth > 0) {
        struct pl_bit_encoding *tbits = &target->repr.bits;
        tbits->color_depth += dither_depth - tbits->sample_depth;
        tbits->sample_depth = dither_depth;
    }

    if (opts->icc_opts->icc_use_luma) {
        p->icc_params.max_luma = 0.0f;
    } else {
        pl_color_space_nominal_luma_ex(pl_nominal_luma_params(
            .color    = &target->color,
            .metadata = PL_HDR_METADATA_HDR10, // use only static HDR nits
            .scaling  = PL_HDR_NITS,
            .out_max  = &p->icc_params.max_luma,
        ));
    }

    pl_icc_update(p->pllog, &p->icc_profile, NULL, &p->icc_params);
    target->icc = p->icc_profile;
}

static void apply_crop(struct pl_frame *frame, struct mp_rect crop,
                       int width, int height)
{
    frame->crop = (struct pl_rect2df) {
        .x0 = crop.x0,
        .y0 = crop.y0,
        .x1 = crop.x1,
        .y1 = crop.y1,
    };

    // mpv gives us rotated/flipped rects, libplacebo expects unrotated
    pl_rect2df_rotate(&frame->crop, -frame->rotation);
    if (frame->crop.x1 < frame->crop.x0) {
        frame->crop.x0 = width - frame->crop.x0;
        frame->crop.x1 = width - frame->crop.x1;
    }

    if (frame->crop.y1 < frame->crop.y0) {
        frame->crop.y0 = height - frame->crop.y0;
        frame->crop.y1 = height - frame->crop.y1;
    }
}

static bool set_colorspace_hint(struct priv *p, struct pl_color_space *hint)
{
    struct ra_swapchain *sw = p->ra_ctx->swapchain;

    struct mp_image_params params = {
        .color = hint ? *hint : pl_color_space_srgb,
        .repr = {
            .sys = PL_COLOR_SYSTEM_RGB,
            .levels = p->output_levels ? p->output_levels : PL_COLOR_LEVELS_FULL,
            .alpha = p->ra_ctx->opts.want_alpha ? PL_ALPHA_INDEPENDENT : PL_ALPHA_NONE,
        },
    };

    if (sw->fns->set_color && sw->fns->set_color(sw, hint ? &params : NULL)) {
        if (hint) {
            *hint = params.color;
            return true;
        }
    }
    pl_swapchain_colorspace_hint(p->sw, hint);
    return false;
}

static void update_tm_viz(struct pl_color_map_params *params,
                          const struct pl_frame *target)
{
    if (!params->visualize_lut)
        return;

    // Use right half of screen for TM visualization, constrain to 1:1 AR
    const float out_w = fabsf(pl_rect_w(target->crop));
    const float out_h = fabsf(pl_rect_h(target->crop));
    const float size = MPMIN(out_w / 2.0f, out_h);
    params->visualize_rect = (pl_rect2df) {
        .x0 = 1.0f - size / out_w,
        .x1 = 1.0f,
        .y0 = 0.0f,
        .y1 = size / out_h,
    };

    // Visualize red-blue plane
    params->visualize_hue = M_PI / 4.0;
}

static void update_hook_opts_dynamic(struct priv *p, const struct pl_hook *hook,
                                     const struct mp_image *mpi);

static bool draw_frame(struct vo *vo, struct vo_frame *frame)
{
    struct priv *p = vo->priv;
    pl_options pars = p->pars;
    pl_gpu gpu = p->gpu;
    update_options(vo);

    struct pl_render_params params = pars->params;
    const struct gl_video_opts *opts = p->opts_cache->opts;
    bool will_redraw = frame->display_synced && frame->num_vsyncs > 1;
    bool cache_frame = will_redraw || frame->still || p->paused;
    bool can_interpolate = opts->interpolation && frame->display_synced &&
                           !frame->still && frame->num_frames > 1 && !p->paused;
    double pts_offset = can_interpolate ? frame->ideal_frame_vsync : 0;
    params.info_callback = info_callback;
    params.info_priv = vo;
    params.skip_caching_single_frame = !cache_frame;
    params.preserve_mixing_cache = p->next_opts->inter_preserve && !frame->still;
    if (frame->still || p->paused)
        params.frame_mixer = NULL;

    if (frame->current && frame->current->params.vflip) {
        pl_matrix2x2 m = { .m = {{1, 0}, {0, -1}}, };
        pars->distort_params.transform.mat = m;
        params.distort_params = &pars->distort_params;
    } else {
        params.distort_params = NULL;
    }

    // pl_queue advances its internal virtual PTS and culls available frames
    // based on this value and the VPS/FPS ratio. Requesting a non-monotonic PTS
    // is an invalid use of pl_queue. Reset it if this happens in an attempt to
    // recover as much as possible. Ideally, this should never occur, and if it
    // does, it should be corrected. The ideal_frame_vsync may be negative if
    // the last draw did not align perfectly with the vsync. In this case, we
    // should have the previous frame available in pl_queue, or a reset is
    // already requested. Clamp the check to 0, as we don't have the previous
    // frame in vo_frame anyway.
    struct pl_source_frame vpts;
    if (frame->current && !p->want_reset) {
        if (pl_queue_peek(p->queue, 0, &vpts) &&
            frame->current->pts + MPMAX(0, pts_offset) < vpts.pts)
        {
            MP_VERBOSE(vo, "Forcing queue refill, PTS(%f + %f | %f) < VPTS(%f)\n",
                       frame->current->pts, pts_offset,
                       frame->ideal_frame_vsync_duration, vpts.pts);
            p->want_reset = true;
        }
    }

    // Push all incoming frames into the frame queue
    for (int n = 0; n < frame->num_frames; n++) {
        int id = frame->frame_id + n;

        if (p->want_reset) {
            pl_queue_reset(p->queue);
            p->last_pts = 0.0;
            p->last_id = 0;
            p->want_reset = false;
            p->flush_cache = true;
        }

        if (p->flush_cache) {
            pl_renderer_flush_cache(p->rr);
            p->flush_cache = false;
        }

        if (id <= p->last_id)
            continue; // ignore already seen frames

        struct mp_image *mpi = mp_image_new_ref(frame->frames[n]);
        struct frame_priv *fp = talloc_zero(mpi, struct frame_priv);
        mpi->priv = fp;
        fp->vo = vo;

        pl_queue_push(p->queue, &(struct pl_source_frame) {
            .pts = mpi->pts,
            .duration = can_interpolate ? frame->approx_duration : 0,
            .frame_data = mpi,
            .map = map_frame,
            .unmap = unmap_frame,
            .discard = discard_frame,
        });

        p->last_id = id;
    }

    struct ra_swapchain *sw = p->ra_ctx->swapchain;

    struct pl_color_space target_csp = {0};
    // TODO: Implement this for all backends
    if (sw->fns->target_csp)
        target_csp = sw->fns->target_csp(sw);
    if (target_csp.primaries == PL_COLOR_PRIM_UNKNOWN)
        target_csp.primaries = mp_get_best_prim_container(&target_csp.hdr.prim);
    if (!pl_color_transfer_is_hdr(target_csp.transfer)) {
        // limit min_luma to 1000:1 contrast ratio in SDR mode
        if (target_csp.hdr.min_luma > PL_COLOR_SDR_WHITE / PL_COLOR_SDR_CONTRAST)
            target_csp.hdr.min_luma = 0;
    }
    // maxFALL in display metadata is in fact MaxFullFrameLuminance. Wayland
    // reports it as maxFALL directly, but this doesn't mean the same thing.
    target_csp.hdr.max_fall = 0;

    struct pl_color_space hint = {0};
    bool target_hint = p->next_opts->target_hint == 1 ||
                       (p->next_opts->target_hint == -1 &&
                        target_csp.transfer != PL_COLOR_TRC_UNKNOWN);
    // Assume HDR is supported, if target_csp() is not available
    // TODO: Remove this fallback when all backends support target_csp()
    bool target_unknown = target_csp.transfer == PL_COLOR_TRC_UNKNOWN;
    if (target_unknown) {
        target_csp = (struct pl_color_space){
            .transfer = opts->target_trc ? opts->target_trc : pl_color_space_hdr10.transfer };
    }
    bool external_params = false;
    if (target_hint && frame->current) {
        const struct pl_color_space *source = &frame->current->params.color;
        const struct pl_color_space *target = &target_csp;
        hint = *source;
        // Apply target contrast to the hint, this is important for SDR, because
        // libplacebo defaults to 1000:1 contrast ratio otherwise.
        if (!hint.hdr.min_luma)
            hint.hdr.min_luma = target->hdr.min_luma;
        if (p->next_opts->target_hint_mode == 0) {
            hint = *target;
            if (pl_color_transfer_is_hdr(hint.transfer) && !pl_primaries_valid(&hint.hdr.prim))
                pl_color_space_merge(&hint, source);
            if (target_unknown && !opts->target_trc && !pl_color_transfer_is_hdr(source->transfer))
                hint = *source;
            // Restore target luminance if it was present, note that we check
            // max_luma only, this make sure that max_cll/max_fall is not take
            // from source.
            if (target->hdr.max_luma) {
                hint.hdr.max_luma = target->hdr.max_luma;
                hint.hdr.min_luma = target->hdr.min_luma;
                hint.hdr.max_cll  = target->hdr.max_cll;
                hint.hdr.max_fall = target->hdr.max_fall;
            }
        }
        if (p->next_opts->target_hint_mode == 2) { // source-dynamic
            pl_color_space_nominal_luma_ex(pl_nominal_luma_params(
                .color      = &hint,
                .metadata   = PL_HDR_METADATA_ANY,
                .scaling    = PL_HDR_NITS,
                .out_min    = !hint.hdr.min_luma ? &hint.hdr.min_luma : NULL,
                .out_max    = &hint.hdr.max_luma,
            ));
            // Set maxCLL to dynamic max luminance. Note that libplacebo uses
            // max luminace as maxCLL in practice.
            hint.hdr.max_cll = hint.hdr.max_luma;
            // Keep maxFALL from static metadata, unless its value is too high.
            // Could be set to 0, but let's keep it for now.
            if (hint.hdr.max_fall > hint.hdr.max_cll)
                hint.hdr.max_fall = 0;
        }
        // Infer missing bits now. This is important so that we don't lose
        // information after user option overrides. For example, if the user
        // sets target_trc to PQ, but the hint(source) is SDR, we want to fill
        // in SDR luminance values instead of the default PQ range.
        struct pl_color_space source_csp = *source;
        pl_color_space_infer_map(&source_csp, &hint);
        // Always prefer target luminance and transfer for inverse tone mapping
        if (pl_color_transfer_is_hdr(target->transfer) && opts->tone_map.inverse) {
            hint.transfer     = target->transfer;
            hint.hdr.max_luma = target->hdr.max_luma;
            hint.hdr.min_luma = target->hdr.min_luma;
            hint.hdr.max_cll  = target->hdr.max_cll;
            hint.hdr.max_fall = target->hdr.max_fall;
        }
        if (opts->target_prim)
            hint.primaries = opts->target_prim;
        if (opts->target_gamut)
            mp_parse_raw_primaries(mp_null_log, opts->target_gamut, &hint.hdr.prim);
        if (opts->target_trc)
            hint.transfer = opts->target_trc;
        if (opts->target_peak)
            hint.hdr.max_luma = opts->target_peak;
        if (opts->hdr_reference_white && !pl_color_transfer_is_hdr(hint.transfer))
            hint.hdr.max_luma = opts->hdr_reference_white;
        // Always set maxCLL, display uses this metadata and we shouldn't let it
        // fallback to default value.
        if (!hint.hdr.max_cll)
            hint.hdr.max_cll = hint.hdr.max_luma;
        // If tone mapping is required, adjust maxCLL and maxFALL
        if (source->hdr.max_luma > hint.hdr.max_luma || opts->tone_map.inverse) {
            // Set maxCLL to the target luminance if it's not already lower
            if (!hint.hdr.max_cll || hint.hdr.max_luma < hint.hdr.max_cll || opts->tone_map.inverse)
                hint.hdr.max_cll = hint.hdr.max_luma;
            // There's no reliable way to estimate maxFALL here
            hint.hdr.max_fall = 0;
        }
        if (hint.hdr.max_cll && hint.hdr.max_fall > hint.hdr.max_cll)
            hint.hdr.max_fall = 0;
        apply_target_contrast(p, &hint, hint.hdr.min_luma);
        if (p->icc_profile)
            hint = p->icc_profile->csp;
        if (opts->icc_opts->icc_use_luma) {
            p->icc_params.max_luma = 0.0f;
        } else {
            pl_color_space_nominal_luma_ex(pl_nominal_luma_params(
                .color    = &hint,
                .metadata = PL_HDR_METADATA_HDR10, // use only static HDR nits
                .scaling  = PL_HDR_NITS,
                .out_max  = &p->icc_params.max_luma,
            ));
        }
        pl_icc_update(p->pllog, &p->icc_profile, NULL, &p->icc_params);
        // Update again after possible max_luma change
        if (p->icc_profile)
            hint = p->icc_profile->csp;
        external_params = set_colorspace_hint(p, &hint);
    } else if (!target_hint) {
        if (!hint.hdr.min_luma)
            hint.hdr.min_luma = target_csp.hdr.min_luma;
        external_params = set_colorspace_hint(p, NULL);
    }

    struct pl_swapchain_frame swframe;
    bool should_draw = sw->fns->start_frame(sw, NULL); // for wayland logic
    if (!should_draw || !pl_swapchain_start_frame(p->sw, &swframe)) {
        if (frame->current) {
            // Advance the queue state to the current PTS to discard unused frames
            struct pl_queue_params qparams = *pl_queue_params(
                .pts = frame->current->pts + pts_offset,
                .radius = pl_frame_mix_radius(&params),
                .vsync_duration = can_interpolate ? frame->ideal_frame_vsync_duration : 0,
                .drift_compensation = 0,
            );
            pl_queue_update(p->queue, NULL, &qparams);
        }
        return VO_FALSE;
    }

    bool valid = false;
    p->is_interpolated = false;

    // Calculate target
    struct pl_frame target;
    pl_frame_from_swapchain(&target, &swframe);
    if (external_params)
        target.color = hint;
    bool strict_sw_params = target_hint && p->next_opts->target_hint_strict;
    apply_target_options(p, &target, hint.hdr.min_luma, strict_sw_params);
    bool clip_gamut = pl_primaries_valid(&target.color.hdr.prim);
#if PL_API_VER >= 362
    clip_gamut = clip_gamut && target.color.transfer != PL_COLOR_TRC_SCRGB;
#endif
    if (clip_gamut) {
        // Ensure resulting gamut still fits inside container
        target.color.hdr.prim = pl_primaries_clip(&target.color.hdr.prim,
                                    pl_raw_primaries_get(target.color.primaries));
    }
    if (target.color.transfer == PL_COLOR_TRC_SRGB && frame->current &&
        ((opts->sdr_adjust_gamma == 0 && opts->target_trc == PL_COLOR_TRC_UNKNOWN) ||
         opts->sdr_adjust_gamma == -1))
    {
        switch (frame->current->params.color.transfer) {
        case PL_COLOR_TRC_BT_1886:
        case PL_COLOR_TRC_GAMMA22:
        case PL_COLOR_TRC_SRGB:
            target.color.transfer = frame->current->params.color.transfer;
        }
    }
    if (target.color.transfer == PL_COLOR_TRC_SRGB) {
        // sRGB reference display is pure 2.2 power function, see IEC 61966-2-1-1999.
        if (opts->treat_srgb_as_power22 & 2)
            target.color.transfer = PL_COLOR_TRC_GAMMA22;

        // TODO: Vulkan on Wayland currently interprets VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
        // in ambiguous way, depending if compositor advertises sRGB support.
        // There is currently no clear path forward to resolve this ambiguity.
        // Depending how it's resolved in Wayland Protocol, Mesa, things will
        // change.
        // See: <https://gitlab.freedesktop.org/wayland/wayland-protocols/-/merge_requests/456>
#ifdef _WIN32
        // Windows uses the sRGB piecewise function. Send piecewise sRGB to
        // Windows in HDR mode so that it can be converted to PQ, the same way
        // as mpv does internally. Note that in SDR mode, even with ACM enabled,
        // Windows assumes the display is sRGB. It doesn't perform gamma
        // conversion, or any conversions would roundtrip back to sRGB.
        // In which case the EOTF depends on the display.
        // Ideally, compositors would agree on how to handle sRGB, but I’ll
        // leave that part of the story for the reader to explore.
        // Note: Older Windows versions, without ACM, were not able to convert
        // sRGB to PQ output. We are not concerned about this case, as it would
        // look wrong anyway.
        bool target_pq = !target_unknown && target_csp.transfer == PL_COLOR_TRC_PQ;
        if (opts->treat_srgb_as_power22 & 4 && target_pq)
            target.color.transfer = PL_COLOR_TRC_SRGB;
#endif
    }
    stats_time_start(p->stats, "osd-update");
    update_overlays(vo, p->osd_res,
                    (frame->current && opts->blend_subs) ? OSD_DRAW_OSD_ONLY : 0,
                    PL_OVERLAY_COORDS_DST_FRAME, &p->osd_state, &target, frame->current,
                    frame->current ? frame->current->params.stereo3d : 0);
    stats_time_end(p->stats, "osd-update");
    apply_crop(&target, p->dst, swframe.fbo->params.w, swframe.fbo->params.h);
    update_tm_viz(&pars->color_map_params, &target);

    struct pl_frame_mix mix = {0};
    if (frame->current) {
        // Update queue state
        struct pl_queue_params qparams = *pl_queue_params(
            .pts = frame->current->pts + pts_offset,
            .radius = pl_frame_mix_radius(&params),
            .vsync_duration = can_interpolate ? frame->ideal_frame_vsync_duration : 0,
            .interpolation_threshold = opts->interpolation_threshold,
            .drift_compensation = 0,
        );

        // Depending on the vsync ratio, we may be up to half of the vsync
        // duration before the current frame time. This works fine because
        // pl_queue will have this frame, unless it's after a reset event. In
        // this case, start from the first available frame.
        struct pl_source_frame first;
        if (pl_queue_peek(p->queue, 0, &first) && qparams.pts < first.pts) {
            if (first.pts != frame->current->pts)
                MP_VERBOSE(vo, "Current PTS(%f) != VPTS(%f)\n", frame->current->pts, first.pts);
            MP_VERBOSE(vo, "Clamping first frame PTS from %f to %f\n", qparams.pts, first.pts);
            qparams.pts = first.pts;
        }
        p->last_pts = qparams.pts;

        switch (pl_queue_update(p->queue, &mix, &qparams)) {
        case PL_QUEUE_ERR:
            MP_ERR(vo, "Failed updating frames!\n");
            goto done;
        case PL_QUEUE_EOF:
            abort(); // we never signal EOF
        case PL_QUEUE_MORE:
            // This is expected to happen semi-frequently near the start and
            // end of a file, so only log it at high verbosity and move on.
            if (!frame->still)
                MP_DBG(vo, "Render queue underrun.\n");
            break;
        case PL_QUEUE_OK:
            break;
        }

        // Update source crop and overlays on all existing frames. We
        // technically own the `pl_frame` struct so this is kosher. This could
        // be partially avoided by instead flushing the queue on resizes, but
        // doing it this way avoids unnecessarily re-uploading frames.
        for (int i = 0; i < mix.num_frames; i++) {
            struct pl_frame *image = (struct pl_frame *) mix.frames[i];
            struct mp_image *mpi = image->user_data;
            struct frame_priv *fp = mpi->priv;
            apply_crop(image, p->src, vo->params->w, vo->params->h);
            if (opts->blend_subs) {
                if (frame->redraw)
                    p->osd_sync++;
                if (fp->osd_sync < p->osd_sync) {
                    float w = pl_rect_w(opts->blend_subs == BLEND_SUBS_VIDEO ? image->crop : target.crop);
                    float h = pl_rect_h(opts->blend_subs == BLEND_SUBS_VIDEO ? image->crop : target.crop);
                    float rx = w / pl_rect_w(image->crop);
                    float ry = h / pl_rect_h(image->crop);
                    struct mp_osd_res res = {
                        .w = w,
                        .h = h,
                        .ml = -image->crop.x0 * rx,
                        .mr = (image->crop.x1 - vo->params->w) * rx,
                        .mt = -image->crop.y0 * ry,
                        .mb = (image->crop.y1 - vo->params->h) * ry,
                        .display_par = 1.0,
                    };
                    enum pl_overlay_coords rel = opts->blend_subs == BLEND_SUBS_VIDEO
                        ? PL_OVERLAY_COORDS_SRC_CROP : PL_OVERLAY_COORDS_DST_CROP;
                    stats_time_start(p->stats, "osd-blend-update");
                    update_overlays(vo, res, OSD_DRAW_SUB_ONLY,
                                    rel, &fp->subs, image, mpi,
                                    mpi->params.stereo3d);
                    stats_time_end(p->stats, "osd-blend-update");
                    fp->osd_sync = p->osd_sync;
                }
            } else {
                // Disable overlays when blend_subs is disabled
                image->num_overlays = 0;
                fp->osd_sync = 0;
            }

            // Update the frame signature to include the current OSD sync
            // value, in order to disambiguate between identical frames with
            // modified OSD. Shift the OSD sync value by a lot to avoid
            // collisions with low signature values.
            //
            // This is safe to do because `pl_frame_mix.signature` lives in
            // temporary memory that is only valid for this `pl_queue_update`.
            ((uint64_t *) mix.signatures)[i] ^= fp->osd_sync << 48;
        }

        // Update dynamic hook parameters
        for (int i = 0; i < pars->params.num_hooks; i++)
            update_hook_opts_dynamic(p, p->hooks[i], frame->current);
    }

    // Render frame
    stats_time_start(p->stats, "render");
    bool render_ok = pl_render_image_mix(p->rr, &mix, &target, &params);
    stats_time_end(p->stats, "render");
    if (!render_ok) {
        MP_ERR(vo, "Failed rendering frame!\n");
        goto done;
    }

    {   // DEBUG: re-render the final composited frame to a host-readable target, dump PPM
        static int fdumped = 0;
        if (!fdumped && getenv("MPV_DUMP_FRAME")) {
            pl_tex fbo = swframe.fbo;
            int fw = fbo->params.w, fh = fbo->params.h;
            pl_tex rd = pl_tex_create(gpu, pl_tex_params(.w = fw, .h = fh,
                .format = fbo->params.format, .renderable = true, .host_readable = true));
            if (rd && rd->params.host_readable) {
                struct pl_frame t2 = target;
                t2.planes[0].texture = rd;
                if (pl_render_image_mix(p->rr, &mix, &t2, &params)) {
                    int nc = fbo->params.format->num_components;
                    size_t px = (size_t) fw * fh;
                    uint8_t *buf = malloc(px * nc);
                    if (buf && pl_tex_download(gpu, pl_tex_transfer_params(.tex = rd, .ptr = buf))) {
                        FILE *f = fopen("/tmp/frame_dump.ppm", "wb");
                        if (f) { fprintf(f, "P6\n%d %d\n255\n", fw, fh);
                            for (size_t i = 0; i < px; i++) fwrite(buf + i * nc, 1, 3, f);
                            fclose(f); }
                        fdumped = 1;
                        fprintf(stderr, "GCDBG dumped frame %dx%d nc=%d -> /tmp/frame_dump.ppm\n", fw, fh, nc);
                        fflush(stderr);
                    }
                    free(buf);
                }
            } else {
                fprintf(stderr, "GCDBG frame dump: rd=%p readable=%d\n", (void*)rd,
                        rd ? rd->params.host_readable : -1);
                fflush(stderr);
            }
            pl_tex_destroy(gpu, &rd);
        }
    }

    struct pl_frame ref_frame;
    pl_frames_infer_mix(p->rr, &mix, &target, &ref_frame);

    mp_mutex_lock(&vo->params_mutex);
    p->target_params = (struct mp_image_params){
        .imgfmt_name = swframe.fbo->params.format
                        ? swframe.fbo->params.format->name : NULL,
        .w = mp_rect_w(p->dst),
        .h = mp_rect_h(p->dst),
        .color = target.color,
        .repr = target.repr,
        .rotate = target.rotation,
    };
    vo->target_params = &p->target_params;

    if (vo->params) {
        // Augment metadata with peak detection max_pq_y / avg_pq_y
        vo->has_peak_detect_values = pl_renderer_get_hdr_metadata(p->rr, &vo->params->color.hdr);
    }
    mp_mutex_unlock(&vo->params_mutex);

    p->is_interpolated = pts_offset != 0 && mix.num_frames > 1;
    valid = true;
    // fall through

done:
    if (!valid) // clear with purple to indicate error
        pl_tex_clear(gpu, swframe.fbo, (float[4]){ 0.5, 0.0, 1.0, 1.0 });

    pl_gpu_flush(gpu);
    p->frame_pending = true;
    return VO_TRUE;
}

static void flip_page(struct vo *vo)
{
    struct priv *p = vo->priv;
    struct ra_swapchain *sw = p->ra_ctx->swapchain;

    if (p->frame_pending) {
        if (!pl_swapchain_submit_frame(p->sw))
            MP_ERR(vo, "Failed presenting frame!\n");
        p->frame_pending = false;
    }

    sw->fns->swap_buffers(sw);
}

static void get_vsync(struct vo *vo, struct vo_vsync_info *info)
{
    struct priv *p = vo->priv;
    struct ra_swapchain *sw = p->ra_ctx->swapchain;
    if (sw->fns->get_vsync)
        sw->fns->get_vsync(sw, info);
}

static int query_format(struct vo *vo, int format)
{
    struct priv *p = vo->priv;
    if (ra_hwdec_get(&p->hwdec_ctx, format))
        return true;

    bool supported = format_supported(vo, format, false);
    if (!supported)
        supported = format_supported(vo, format, true);

    return supported;
}

static void resize(struct vo *vo)
{
    struct priv *p = vo->priv;
    struct mp_rect src, dst;
    struct mp_osd_res osd;
    vo_get_src_dst_rects(vo, &src, &dst, &osd);
    if (vo->dwidth && vo->dheight) {
        gpu_ctx_resize(p->context, vo->dwidth, vo->dheight);
        vo->want_redraw = true;
    }

    if (mp_rect_equals(&p->src, &src) &&
        mp_rect_equals(&p->dst, &dst) &&
        osd_res_equals(p->osd_res, osd))
        return;

    p->osd_sync++;
    p->osd_res = osd;
    p->src = src;
    p->dst = dst;
}

static int reconfig(struct vo *vo, struct mp_image_params *params)
{
    struct priv *p = vo->priv;
    if (!p->ra_ctx->fns->reconfig(p->ra_ctx))
        return -1;

    resize(vo);
    mp_mutex_lock(&vo->params_mutex);
    vo->target_params = NULL;
    mp_mutex_unlock(&vo->params_mutex);
    return 0;
}

// Takes over ownership of `icc`. Can be used to unload profile (icc.len == 0)
static bool update_icc(struct priv *p, struct bstr icc)
{
    struct pl_icc_profile profile = {
        .data = icc.start,
        .len  = icc.len,
    };

    pl_icc_profile_compute_signature(&profile);

    bool ok = pl_icc_update(p->pllog, &p->icc_profile, &profile, &p->icc_params);
    talloc_free(icc.start);
    return ok;
}

// Returns whether the ICC profile was updated (even on failure)
static bool update_auto_profile(struct priv *p, int *events)
{
    const struct gl_video_opts *opts = p->opts_cache->opts;
    if (!opts->icc_opts || !opts->icc_opts->profile_auto || p->icc_path)
        return false;

    MP_VERBOSE(p, "Querying ICC profile...\n");
    bstr icc = {0};
    int r = p->ra_ctx->fns->control(p->ra_ctx, events, VOCTRL_GET_ICC_PROFILE, &icc);

    if (r != VO_NOTAVAIL) {
        if (r == VO_FALSE) {
            MP_WARN(p, "Could not retrieve an ICC profile.\n");
        } else if (r == VO_NOTIMPL) {
            MP_ERR(p, "icc-profile-auto not implemented on this platform.\n");
        }

        update_icc(p, icc);
        return true;
    }

    return false;
}

static void video_screenshot(struct vo *vo, struct voctrl_screenshot *args)
{
    struct priv *p = vo->priv;
    pl_options pars = p->pars;
    pl_gpu gpu = p->gpu;
    pl_tex fbo = NULL;
    args->res = NULL;

    update_options(vo);
    struct pl_render_params params = pars->params;
    params.info_callback = NULL;
    params.skip_caching_single_frame = true;
    params.preserve_mixing_cache = false;
    params.frame_mixer = NULL;

    struct pl_peak_detect_params peak_params;
    if (params.peak_detect_params) {
        peak_params = *params.peak_detect_params;
        params.peak_detect_params = &peak_params;
        peak_params.allow_delayed = false;
    }

    // Retrieve the current frame from the frame queue
    struct pl_frame_mix mix;
    enum pl_queue_status status;
    struct pl_queue_params qparams = *pl_queue_params(
        .pts = p->last_pts,
        .drift_compensation = 0,
    );
    status = pl_queue_update(p->queue, &mix, &qparams);
    mp_assert(status != PL_QUEUE_EOF);
    if (status == PL_QUEUE_ERR) {
        MP_ERR(vo, "Unknown error occurred while trying to take screenshot!\n");
        return;
    }
    if (!mix.num_frames) {
        MP_ERR(vo, "No frames available to take screenshot of, is a file loaded?\n");
        return;
    }

    // Passing an interpolation radius of 0 guarantees that the first frame in
    // the resulting mix is the correct frame for this PTS
    struct pl_frame image = *(struct pl_frame *) mix.frames[0];
    struct mp_image *mpi = image.user_data;
    struct mp_rect src = p->src, dst = p->dst;
    struct mp_osd_res osd = p->osd_res;
    if (!args->scaled) {
        int w, h;
        mp_image_params_get_dsize(&mpi->params, &w, &h);
        if (w < 1 || h < 1)
            return;

        int src_w = mpi->params.w;
        int src_h = mpi->params.h;
        src = (struct mp_rect) {0, 0, src_w, src_h};
        dst = (struct mp_rect) {0, 0, w, h};

        if (mp_image_crop_valid(&mpi->params))
            src = mpi->params.crop;

        if (mpi->params.rotate % 180 == 90) {
            MPSWAP(int, w, h);
            MPSWAP(int, src_w, src_h);
        }
        mp_rect_rotate(&src, src_w, src_h, mpi->params.rotate);
        mp_rect_rotate(&dst, w, h, mpi->params.rotate);

        osd = (struct mp_osd_res) {
            .display_par = 1.0,
            .w = mp_rect_w(dst),
            .h = mp_rect_h(dst),
        };
    }

    // Create target FBO, try high bit depth first
    int mpfmt;
    for (int depth = args->high_bit_depth ? 16 : 8; depth; depth -= 8) {
        if (depth == 16) {
            mpfmt = IMGFMT_RGBA64;
        } else {
            mpfmt = p->ra_ctx->opts.want_alpha ? IMGFMT_RGBA : IMGFMT_RGB0;
        }
        pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 4, depth, depth,
                                 PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE);
        if (!fmt)
            continue;

        fbo = pl_tex_create(gpu, pl_tex_params(
            .w = osd.w,
            .h = osd.h,
            .format = fmt,
            .blit_dst = true,
            .renderable = true,
            .host_readable = true,
            .storable = fmt->caps & PL_FMT_CAP_STORABLE,
        ));
        if (fbo)
            break;
    }

    if (!fbo) {
        MP_ERR(vo, "Failed creating target FBO for screenshot!\n");
        return;
    }

    struct pl_frame target = {
        .repr = pl_color_repr_rgb,
        .num_planes = 1,
        .planes[0] = {
            .texture = fbo,
            .components = 4,
            .component_mapping = {0, 1, 2, 3},
        },
    };

    const struct gl_video_opts *opts = p->opts_cache->opts;
    if (args->scaled) {
        // Apply target LUT, ICC profile and CSP override only in window mode
        apply_target_options(p, &target, 0, false);
    } else if (args->native_csp) {
        target.color = image.color;
    } else {
        target.color = pl_color_space_srgb;
    }

    // sRGB reference display is pure 2.2 power function, see IEC 61966-2-1-1999.
    // Round-trip back to sRGB if the source is also sRGB. In other cases, we
    // use piecewise sRGB transfer function, as this is likely the be expected
    // for file encoding.
    if (opts->treat_srgb_as_power22 & 1 &&
        target.color.transfer == PL_COLOR_TRC_SRGB &&
        mpi->params.color.transfer == PL_COLOR_TRC_SRGB)
    {
        target.color.transfer = PL_COLOR_TRC_GAMMA22;
    }

    apply_crop(&image, src, mpi->params.w, mpi->params.h);
    apply_crop(&target, dst, fbo->params.w, fbo->params.h);
    update_tm_viz(&pars->color_map_params, &target);

    int osd_flags = 0;
    if (!args->subs)
        osd_flags |= OSD_DRAW_OSD_ONLY;
    if (!args->osd)
        osd_flags |= OSD_DRAW_SUB_ONLY;

    struct frame_priv *fp = mpi->priv;
    if (opts->blend_subs) {
        float w = pl_rect_w(opts->blend_subs == BLEND_SUBS_VIDEO ? image.crop : target.crop);
        float h = pl_rect_h(opts->blend_subs == BLEND_SUBS_VIDEO ? image.crop : target.crop);
        float rx = w / pl_rect_w(image.crop);
        float ry = h / pl_rect_h(image.crop);
        struct mp_osd_res res = {
            .w = w,
            .h = h,
            .ml = -image.crop.x0 * rx,
            .mr = (image.crop.x1 - vo->params->w) * rx,
            .mt = -image.crop.y0 * ry,
            .mb = (image.crop.y1 - vo->params->h) * ry,
            .display_par = 1.0,
        };
        enum pl_overlay_coords rel = opts->blend_subs == BLEND_SUBS_VIDEO
            ? PL_OVERLAY_COORDS_SRC_CROP : PL_OVERLAY_COORDS_DST_CROP;
        update_overlays(vo, res, osd_flags,
                        rel, &fp->subs, &image, mpi,
                        mpi->params.stereo3d);
    } else {
        // Disable overlays when blend_subs is disabled
        update_overlays(vo, osd, osd_flags, PL_OVERLAY_COORDS_DST_FRAME,
                        &p->osd_state, &target, mpi,
                        mpi->params.stereo3d);
        image.num_overlays = 0;
    }

    if (!pl_render_image(p->rr, &image, &target, &params)) {
        MP_ERR(vo, "Failed rendering frame!\n");
        goto done;
    }

    args->res = mp_image_alloc(mpfmt, fbo->params.w, fbo->params.h);
    if (!args->res)
        goto done;

    args->res->params.color.primaries = target.color.primaries;
    args->res->params.color.transfer = target.color.transfer;
    args->res->params.repr.levels = target.repr.levels;
    args->res->params.color.hdr = target.color.hdr;
    if (args->scaled)
        args->res->params.p_w = args->res->params.p_h = 1;

    bool ok = pl_tex_download(gpu, pl_tex_transfer_params(
        .tex = fbo,
        .ptr = args->res->planes[0],
        .row_pitch = args->res->stride[0],
    ));

    if (!ok)
        TA_FREEP(&args->res);

    // fall through
done:
    pl_tex_destroy(gpu, &fbo);
}

static inline void copy_frame_info_to_mp(struct frame_info *pl,
                                         struct mp_frame_perf *mp,
                                         struct mp_pass_perf *hwdec_perf,
                                         struct mp_pass_perf *sw_upload_perf)
{
    static_assert(MP_ARRAY_SIZE(pl->info) == MP_ARRAY_SIZE(mp->perf), "");
    mp_assert(pl->count <= VO_PASS_PERF_MAX);

    struct mp_pass_perf *perf = mp->perf;
    char (*desc)[VO_PASS_DESC_MAX_LEN] = mp->desc;
    struct mp_pass_perf *perf_end = perf + VO_PASS_PERF_MAX;

    if (hwdec_perf && hwdec_perf->count > 0) {
        *perf++ = *hwdec_perf;
        snprintf(*desc, sizeof(*desc), "map frame (hwdec)");
        desc++;
    }

    if (sw_upload_perf && sw_upload_perf->count > 0) {
        *perf++ = *sw_upload_perf;
        snprintf(*desc, sizeof(*desc), "upload frame");
        desc++;
    }

    for (int i = 0; i < pl->count && perf < perf_end; ++i) {
        const struct pl_dispatch_info *pass = &pl->info[i];

        static_assert(VO_PERF_SAMPLE_COUNT >= MP_ARRAY_SIZE(pass->samples), "");
        mp_assert(pass->num_samples <= MP_ARRAY_SIZE(pass->samples));

        perf->count = MPMIN(pass->num_samples, VO_PERF_SAMPLE_COUNT);
        memcpy(perf->samples, pass->samples, perf->count * sizeof(pass->samples[0]));
        perf->last = pass->last;
        perf->peak = pass->peak;
        perf->avg = pass->average;

        strncpy(*desc, pass->shader->description, sizeof(*desc) - 1);
        (*desc)[sizeof(*desc) - 1] = '\0';
        perf++;
        desc++;
    }

    mp->count = perf - mp->perf;
}

static void update_ra_ctx_options(struct vo *vo, struct ra_ctx_opts *ctx_opts)
{
    struct priv *p = vo->priv;
    struct gl_video_opts *gl_opts = p->opts_cache->opts;
    bool border_alpha = (p->next_opts->border_background == BACKGROUND_COLOR &&
                         gl_opts->background_color.a != 255) ||
                         p->next_opts->border_background == BACKGROUND_NONE;
    ctx_opts->want_alpha = (gl_opts->background == BACKGROUND_COLOR &&
                            gl_opts->background_color.a != 255) ||
                            gl_opts->background == BACKGROUND_NONE ||
                            border_alpha;
}

static int control(struct vo *vo, uint32_t request, void *data)
{
    struct priv *p = vo->priv;

    switch (request) {
    case VOCTRL_SET_PANSCAN:
        resize(vo);
        return VO_TRUE;
    case VOCTRL_PAUSE:
        if (p->is_interpolated)
            vo->want_redraw = true;
        p->paused = true;
        return VO_TRUE;
    case VOCTRL_RESUME:
        p->paused = false;
        return VO_TRUE;

    case VOCTRL_UPDATE_RENDER_OPTS: {
        update_ra_ctx_options(vo, &p->ra_ctx->opts);
        if (p->ra_ctx->fns->update_render_opts)
            p->ra_ctx->fns->update_render_opts(p->ra_ctx);
        vo->want_redraw = true;

        // Special case for --image-lut which requires a full reset.
        int old_type = p->next_opts->image_lut.type;
        update_options(vo);
        struct user_lut image_lut = p->next_opts->image_lut;
        p->want_reset |= image_lut.opt && ((!image_lut.path && image_lut.opt) ||
                         (image_lut.path && strcmp(image_lut.path, image_lut.opt)) ||
                         (old_type != image_lut.type));

        // Also re-query the auto profile, in case `update_render_options`
        // unloaded a manually specified icc profile in favor of
        // icc-profile-auto
        int events = 0;
        update_auto_profile(p, &events);
        vo_event(vo, events);
        return VO_TRUE;
    }

    case VOCTRL_RESET:
        // Defer until the first new frame (unique ID) actually arrives
        p->want_reset = true;
        return VO_TRUE;

    case VOCTRL_PERFORMANCE_DATA: {
        struct voctrl_performance_data *perf = data;
        copy_frame_info_to_mp(&p->perf_fresh, &perf->fresh, &p->hwdec_perf, &p->sw_upload_perf);
        copy_frame_info_to_mp(&p->perf_redraw, &perf->redraw, NULL, NULL);
        return true;
    }

    case VOCTRL_SCREENSHOT:
        video_screenshot(vo, data);
        return true;

    case VOCTRL_EXTERNAL_RESIZE:
        reconfig(vo, NULL);
        return true;

    case VOCTRL_LOAD_HWDEC_API:
        ra_hwdec_ctx_load_fmt(&p->hwdec_ctx, vo->hwdec_devs, data);
        return true;
    }

    int events = 0;
    int r = p->ra_ctx->fns->control(p->ra_ctx, &events, request, data);
    if (events & VO_EVENT_ICC_PROFILE_CHANGED) {
        if (update_auto_profile(p, &events))
            vo->want_redraw = true;
    }
    if (events & VO_EVENT_RESIZE)
        resize(vo);
    if (events & VO_EVENT_EXPOSE)
        vo->want_redraw = true;
    vo_event(vo, events);

    return r;
}

static void wakeup(struct vo *vo)
{
    struct priv *p = vo->priv;
    if (p->ra_ctx && p->ra_ctx->fns->wakeup)
        p->ra_ctx->fns->wakeup(p->ra_ctx);
}

static void wait_events(struct vo *vo, int64_t until_time_ns)
{
    struct priv *p = vo->priv;
    if (p->ra_ctx && p->ra_ctx->fns->wait_events) {
        p->ra_ctx->fns->wait_events(p->ra_ctx, until_time_ns);
    } else {
        vo_wait_default(vo, until_time_ns);
    }
}

static char *cache_filepath(void *ta_ctx, char *dir, const char *prefix, uint64_t key)
{
    bstr filename = {0};
    bstr_xappend_asprintf(ta_ctx, &filename, "%s_%016" PRIx64, prefix, key);
    return mp_path_join_bstr(ta_ctx, bstr0(dir), filename);
}

static pl_cache_obj cache_load_obj(void *p, uint64_t key)
{
    struct cache *c = p;
    void *ta_ctx = talloc_new(NULL);
    pl_cache_obj obj = {0};

    if (!c->dir)
        goto done;

    char *filepath = cache_filepath(ta_ctx, c->dir, c->name, key);
    if (!filepath)
        goto done;

    if (stat(filepath, &(struct stat){0}))
        goto done;

    int64_t load_start = mp_time_ns();
    struct bstr data = stream_read_file(filepath, ta_ctx, c->global, STREAM_MAX_READ_SIZE);
    int64_t load_end = mp_time_ns();
    MP_DBG(c, "%s: key(%" PRIx64 "), size(%zu), load time(%.3f ms)\n",
           __func__, key, data.len,
           MP_TIME_NS_TO_MS(load_end - load_start));

    obj = (pl_cache_obj){
        .key = key,
        .data = talloc_steal(NULL, data.start),
        .size = data.len,
        .free = talloc_free,
    };

done:
    talloc_free(ta_ctx);
    return obj;
}

static void cache_save_obj(void *p, pl_cache_obj obj)
{
    const struct cache *c = p;
    void *ta_ctx = talloc_new(NULL);

    if (!c->dir)
        goto done;

    char *filepath = cache_filepath(ta_ctx, c->dir, c->name, obj.key);
    if (!filepath)
        goto done;

    if (!obj.data || !obj.size) {
        unlink(filepath);
        goto done;
    }

    // Don't save if already exists
    struct stat st;
    if (!stat(filepath, &st) && st.st_size == obj.size) {
        MP_DBG(c, "%s: key(%"PRIx64"), size(%zu)\n", __func__, obj.key, obj.size);
        goto done;
    }

    int64_t save_start = mp_time_ns();
    mp_save_to_file(filepath, obj.data, obj.size);
    int64_t save_end = mp_time_ns();
    MP_DBG(c, "%s: key(%" PRIx64 "), size(%zu), save time(%.3f ms)\n",
           __func__, obj.key, obj.size,
           MP_TIME_NS_TO_MS(save_end - save_start));

done:
    talloc_free(ta_ctx);
}

static void cache_init(struct vo *vo, struct cache *cache, size_t max_size,
                       const char *dir_opt)
{
    struct priv *p = vo->priv;
    const char *name = cache == &p->shader_cache ? "shader" : "icc";
    const size_t limit = cache == &p->shader_cache ? 128 << 20 : 1536 << 20;

    char *dir;
    if (dir_opt && dir_opt[0]) {
        dir = mp_get_user_path(vo, p->global, dir_opt);
    } else {
        dir = mp_find_user_file(vo, p->global, "cache", "");
    }
    if (!dir || !dir[0])
        return;

    mp_mkdirp(dir);
    *cache = (struct cache){
        .log        = p->log,
        .global     = p->global,
        .dir        = dir,
        .name       = name,
        .size_limit = limit,
        .cache = pl_cache_create(pl_cache_params(
            .log = p->pllog,
            .get = cache_load_obj,
            .set = cache_save_obj,
            .priv = cache
        )),
    };
}

struct file_entry {
    char *filepath;
    size_t size;
    time_t atime;
};

static int compare_atime(const void *a, const void *b)
{
    return (((struct file_entry *)b)->atime - ((struct file_entry *)a)->atime);
}

static void cache_uninit(struct priv *p, struct cache *cache)
{
    if (!cache->cache)
        return;

    void *ta_ctx = talloc_new(NULL);
    struct file_entry *files = NULL;
    size_t num_files = 0;
    mp_assert(cache->dir);
    mp_assert(cache->name);

    DIR *d = opendir(cache->dir);
    if (!d)
        goto done;

    struct dirent *dir;
    while ((dir = readdir(d)) != NULL) {
        char *filepath = mp_path_join(ta_ctx, cache->dir, dir->d_name);
        if (!filepath)
            continue;
        struct stat filestat;
        if (stat(filepath, &filestat))
            continue;
        if (!S_ISREG(filestat.st_mode))
            continue;
        bstr fname = bstr0(dir->d_name);
        if (!bstr_eatstart0(&fname, cache->name))
            continue;
        if (!bstr_eatstart0(&fname, "_"))
            continue;
        if (fname.len != 16) // %016x
            continue;
        MP_TARRAY_APPEND(ta_ctx, files, num_files,
                         (struct file_entry){
                             .filepath = filepath,
                             .size     = filestat.st_size,
                             .atime    = filestat.st_atime,
                         });
    }
    closedir(d);

    if (!num_files)
        goto done;

    qsort(files, num_files, sizeof(struct file_entry), compare_atime);

    time_t t = time(NULL);
    size_t cache_size = 0;
    size_t cache_limit = cache->size_limit ? cache->size_limit : SIZE_MAX;
    for (int i = 0; i < num_files; i++) {
        // Remove files that exceed the size limit but are older than one day.
        // This allows for temporary maintaining a larger cache size while
        // adjusting the configuration. The cache will be cleared the next day
        // for unused entries. We don't need to be overly aggressive with cache
        // cleaning; in most cases, it will not grow much, and in others, it may
        // actually be useful to cache more.
        cache_size += files[i].size;
        double rel_use = difftime(t, files[i].atime);
        if (cache_size > cache_limit && rel_use > 60 * 60 * 24) {
            MP_VERBOSE(p, "Removing %s | size: %9zu bytes | last used: %9d seconds ago\n",
                       files[i].filepath, files[i].size, (int)rel_use);
            unlink(files[i].filepath);
        }
    }

done:
    talloc_free(ta_ctx);
    pl_cache_destroy(&cache->cache);
}

static void uninit(struct vo *vo)
{
    struct priv *p = vo->priv;

    // Drain any in-flight uploads.
    if (p->gpu)
        pl_gpu_finish(p->gpu);

    pl_queue_destroy(&p->queue); // destroy this first
    for (int i = 0; i < MP_ARRAY_SIZE(p->osd_state.entries); i++) {
        pl_tex_destroy(p->gpu, &p->osd_state.entries[i].tex);
        pl_tex_destroy(p->gpu, &p->osd_state.entries[i].blur_tex);
        pl_tex_destroy(p->gpu, &p->osd_state.entries[i].tmp_tex);
        pl_tex_destroy(p->gpu, &p->osd_state.entries[i].result_tex);
    }
    for (int i = 0; i < p->num_sub_tex; i++)
        pl_tex_destroy(p->gpu, &p->sub_tex[i]);
    for (int i = 0; i < p->num_sub_scratch; i++)
        pl_tex_destroy(p->gpu, &p->sub_scratch[i]);
    pl_tex_destroy(p->gpu, &p->run_acc);
    pl_tex_destroy(p->gpu, &p->run_tmp);
    pl_tex_destroy(p->gpu, &p->run_cov_f);
    pl_tex_destroy(p->gpu, &p->run_cov_b);
    pl_tex_destroy(p->gpu, &p->glyph_atlas);
    pl_tex_destroy(p->gpu, &p->edge_tex);
    pl_tex_destroy(p->gpu, &p->hdr_tex);
    pl_tex_destroy(p->gpu, &p->work_tex);
    pl_tex_destroy(p->gpu, &p->result_acc);
    pl_tex_destroy(p->gpu, &p->result_tmp);
    pl_tex_destroy(p->gpu, &p->blurwork_tex);
    pl_tex_destroy(p->gpu, &p->combwork_tex);
    for (int i = 0; i < 3; i++)
        pl_buf_destroy(p->gpu, &p->glyph_stage[i]);
    for (int i = 0; i < NUM_OVERLAY_BUFS; i++)
        pl_buf_destroy(p->gpu, &p->overlay_bufs[i]);
    for (int i = 0; i < p->num_user_hooks; i++)
        pl_mpv_user_shader_destroy(&p->user_hooks[i].hook);

    timer_pool_destroy(p->sw_upload_timer);

    if (vo->hwdec_devs) {
        ra_hwdec_mapper_free(&p->hwdec_mapper);
        timer_pool_destroy(p->hwdec_timer);
        ra_hwdec_ctx_uninit(&p->hwdec_ctx);
        hwdec_devices_set_loader(vo->hwdec_devs, NULL, NULL);
        hwdec_devices_destroy(vo->hwdec_devs);
    }

    mp_assert(p->num_dr_buffers == 0);
    mp_mutex_destroy(&p->dr_lock);

    cache_uninit(p, &p->shader_cache);
    cache_uninit(p, &p->icc_cache);

    pl_lut_free(&p->next_opts->image_lut.lut);
    pl_lut_free(&p->next_opts->lut.lut);
    pl_lut_free(&p->next_opts->target_lut.lut);

    pl_icc_close(&p->icc_profile);
    pl_renderer_destroy(&p->rr);
    pl_dispatch_destroy(&p->osd_dp);

    for (int i = 0; i < VO_PASS_PERF_MAX; ++i) {
        pl_shader_info_deref(&p->perf_fresh.info[i].shader);
        pl_shader_info_deref(&p->perf_redraw.info[i].shader);
    }

    pl_options_free(&p->pars);

    p->ra_ctx = NULL;
    p->pllog = NULL;
    p->gpu = NULL;
    p->sw = NULL;
    gpu_ctx_destroy(&p->context);
}

static void load_hwdec_api(void *ctx, struct hwdec_imgfmt_request *params)
{
    vo_control(ctx, VOCTRL_LOAD_HWDEC_API, params);
}

static int preinit(struct vo *vo)
{
    struct priv *p = vo->priv;
    p->opts_cache = m_config_cache_alloc(p, vo->global, &gl_video_conf);
    p->next_opts_cache = m_config_cache_alloc(p, vo->global, &gl_next_conf);
    p->next_opts = p->next_opts_cache->opts;
    p->video_eq = mp_csp_equalizer_create(p, vo->global);
    p->global = vo->global;
    p->log = vo->log;
    p->stats = stats_ctx_create(p, vo->global, "vo/gpu-next");

    struct gl_video_opts *gl_opts = p->opts_cache->opts;
    struct ra_ctx_opts *ctx_opts = mp_get_config_group(vo, vo->global, &ra_ctx_conf);
    update_ra_ctx_options(vo, ctx_opts);
    p->context = gpu_ctx_create(vo, ctx_opts);
    talloc_free(ctx_opts);
    if (!p->context)
        goto err_out;
    // For the time being
    p->ra_ctx = p->context->ra_ctx;
    p->pllog = p->context->pllog;
    p->gpu = p->context->gpu;
    p->sw = p->context->swapchain;
    p->hwdec_ctx = (struct ra_hwdec_ctx) {
        .log = p->log,
        .global = p->global,
        .ra_ctx = p->ra_ctx,
    };

    vo->hwdec_devs = hwdec_devices_create();
    hwdec_devices_set_loader(vo->hwdec_devs, load_hwdec_api, vo);
    ra_hwdec_ctx_init(&p->hwdec_ctx, vo->hwdec_devs, gl_opts->hwdec_interop, false);
    mp_mutex_init(&p->dr_lock);

    if (gl_opts->shader_cache)
        cache_init(vo, &p->shader_cache, 10 << 20, gl_opts->shader_cache_dir);
    if (gl_opts->icc_opts->cache)
        cache_init(vo, &p->icc_cache, 20 << 20, gl_opts->icc_opts->cache_dir);

    pl_gpu_set_cache(p->gpu, p->shader_cache.cache);
    p->rr = pl_renderer_create(p->pllog, p->gpu);
    p->osd_dp = pl_dispatch_create(p->pllog, p->gpu);
    p->queue = pl_queue_create(p->gpu);
    p->osd_fmt[SUBBITMAP_LIBASS] = pl_find_named_fmt(p->gpu, "r8");
    p->osd_fmt[SUBBITMAP_BGRA] = pl_find_named_fmt(p->gpu, "bgra8");
    p->osd_fmt[SUBBITMAP_LIBASS_GLYPHS] = p->osd_fmt[SUBBITMAP_LIBASS];
    p->osd_fmt[SUBBITMAP_LIBASS_OUTLINES] = p->osd_fmt[SUBBITMAP_LIBASS];
    p->osd_acc_fmt = pl_find_named_fmt(p->gpu, "r32f");
    p->osd_sync = 1;

    p->pars = pl_options_alloc(p->pllog);
    update_render_options(vo);
    return 0;

err_out:
    uninit(vo);
    return -1;
}

static const struct pl_filter_config *map_scaler(struct priv *p,
                                                 enum scaler_unit unit)
{
    const struct pl_filter_preset fixed_scalers[] = {
        { "bilinear",       &pl_filter_bilinear },
        { "bicubic_fast",   &pl_filter_bicubic },
        { "nearest",        &pl_filter_nearest },
        { "oversample",     &pl_filter_oversample },
        {0},
    };

    const struct pl_filter_preset fixed_frame_mixers[] = {
        { "linear",         &pl_filter_bilinear },
        { "oversample",     &pl_filter_oversample },
        {0},
    };

    const struct pl_filter_preset *fixed_presets =
        unit == SCALER_TSCALE ? fixed_frame_mixers : fixed_scalers;

    const struct gl_video_opts *opts = p->opts_cache->opts;
    const struct scaler_config *cfg = &opts->scaler[unit];
    struct scaler_config tmp;
    if (cfg->kernel.function == SCALER_INHERIT) {
        tmp = *cfg;
        scaler_conf_merge(&tmp, &opts->scaler[SCALER_SCALE], unit);
        cfg = &tmp;
    }

    const char *kernel_name = m_opt_choice_str(cfg->kernel.functions,
                                               cfg->kernel.function);

    for (int i = 0; fixed_presets[i].name; i++) {
        if (strcmp(kernel_name, fixed_presets[i].name) == 0)
            return fixed_presets[i].filter;
    }

    // Attempt loading filter preset first, fall back to raw filter function
    struct scaler_params *par = &p->scalers[unit];
    const struct pl_filter_preset *preset;
    const struct pl_filter_function_preset *fpreset;
    if ((preset = pl_find_filter_preset(kernel_name))) {
        par->config = *preset->filter;
    } else if ((fpreset = pl_find_filter_function_preset(kernel_name))) {
        par->config = (struct pl_filter_config) {
            .kernel = fpreset->function,
            .params[0] = fpreset->function->params[0],
            .params[1] = fpreset->function->params[1],
        };
    } else {
        MP_ERR(p, "Failed mapping filter function '%s', no libplacebo analog?\n",
               kernel_name);
        return &pl_filter_bilinear;
    }

    const struct pl_filter_function_preset *wpreset;
    if ((wpreset = pl_find_filter_function_preset(
             m_opt_choice_str(cfg->window.functions, cfg->window.function)))) {
        par->config.window = wpreset->function;
        par->config.wparams[0] = wpreset->function->params[0];
        par->config.wparams[1] = wpreset->function->params[1];
    }

    for (int i = 0; i < 2; i++) {
        if (!isnan(cfg->kernel.params[i]))
            par->config.params[i] = cfg->kernel.params[i];
        if (!isnan(cfg->window.params[i]))
            par->config.wparams[i] = cfg->window.params[i];
    }

    par->config.clamp = cfg->clamp;
    if (cfg->antiring > 0.0)
        par->config.antiring = cfg->antiring;
    if (cfg->kernel.blur > 0.0)
        par->config.blur = cfg->kernel.blur;
    if (cfg->kernel.taper > 0.0)
        par->config.taper = cfg->kernel.taper;
    if (cfg->radius > 0.0) {
        if (par->config.kernel->resizable) {
            par->config.radius = cfg->radius;
        } else {
            MP_WARN(p, "Filter radius specified but filter '%s' is not "
                    "resizable, ignoring\n", kernel_name);
        }
    }

    return &par->config;
}

static const struct pl_hook *load_hook(struct priv *p, const char *path)
{
    if (!path || !path[0])
        return NULL;

    for (int i = 0; i < p->num_user_hooks; i++) {
        if (strcmp(p->user_hooks[i].path, path) == 0)
            return p->user_hooks[i].hook;
    }

    char *fname = mp_get_user_path(NULL, p->global, path);
    bstr shader = stream_read_file(fname, p, p->global, 1000000000); // 1GB
    talloc_free(fname);

    const struct pl_hook *hook = NULL;
    if (shader.len)
        hook = pl_mpv_user_shader_parse(p->gpu, shader.start, shader.len);

    MP_TARRAY_APPEND(p, p->user_hooks, p->num_user_hooks, (struct user_hook) {
        .path = talloc_strdup(p, path),
        .hook = hook,
    });

    return hook;
}

static void update_icc_opts(struct priv *p, const struct mp_icc_opts *opts)
{
    if (!opts)
        return;

    if (!opts->profile_auto && !p->icc_path) {
        // Un-set any auto-loaded profiles if icc-profile-auto was disabled
        update_icc(p, (bstr) {0});
    }

    int s_r = 0, s_g = 0, s_b = 0;
    gl_parse_3dlut_size(opts->size_str, &s_r, &s_g, &s_b);
    p->icc_params = pl_icc_default_params;
    p->icc_params.intent = opts->intent;
    p->icc_params.size_r = s_r;
    p->icc_params.size_g = s_g;
    p->icc_params.size_b = s_b;
    p->icc_params.cache = p->icc_cache.cache;

    if (!opts->profile || !opts->profile[0]) {
        // No profile enabled, un-load any existing profiles
        update_icc(p, (bstr) {0});
        TA_FREEP(&p->icc_path);
        return;
    }

    if (p->icc_path && strcmp(opts->profile, p->icc_path) == 0)
        return; // ICC profile hasn't changed

    char *fname = mp_get_user_path(NULL, p->global, opts->profile);
    MP_VERBOSE(p, "Opening ICC profile '%s'\n", fname);
    struct bstr icc = stream_read_file(fname, p, p->global, 100000000); // 100 MB
    talloc_free(fname);
    update_icc(p, icc);

    // Update cached path
    talloc_replace(p, p->icc_path, opts->profile);
}

static void update_lut(struct priv *p, struct user_lut *lut)
{
    if (!lut->opt || !lut->opt[0]) {
        pl_lut_free(&lut->lut);
        TA_FREEP(&lut->path);
        return;
    }

    if (lut->path && strcmp(lut->path, lut->opt) == 0)
        return; // no change

    // Update cached path
    pl_lut_free(&lut->lut);
    talloc_replace(p, lut->path, lut->opt);

    // Load LUT file
    char *fname = mp_get_user_path(NULL, p->global, lut->path);
    MP_VERBOSE(p, "Loading custom LUT '%s'\n", fname);
    const int lut_max_size = 1536 << 20; // 1.5 GiB, matches lut cache limit
    struct bstr lutdata = stream_read_file(fname, NULL, p->global, lut_max_size);
    if (!lutdata.len) {
        MP_ERR(p, "Failed to read LUT data from %s, make sure it's a valid file "
                  "and smaller or equal to %d bytes\n", fname, lut_max_size);
    } else {
        lut->lut = pl_lut_parse_cube(p->pllog, lutdata.start, lutdata.len);
    }
    talloc_free(fname);
    talloc_free(lutdata.start);
}

static void update_hook_opts_dynamic(struct priv *p, const struct pl_hook *hook,
                                     const struct mp_image *mpi)
{
    for (int i = 0; i < hook->num_parameters; i++) {
        double val;
        const struct pl_hook_par *hp = &hook->parameters[i];
        if (!gpu_get_auto_param(mpi, bstr0(hp->name), &val))
            continue;

        switch (hp->type) {
        case PL_VAR_FLOAT: hp->data->f = val; break;
        case PL_VAR_SINT:  hp->data->i = lrint(val); break;
        case PL_VAR_UINT:  hp->data->u = lrint(val); break;
        }
    }
}

static void update_hook_opts(struct priv *p, char **opts, const char *shaderpath,
                             const struct pl_hook *hook)
{
    for (int i = 0; i < hook->num_parameters; i++) {
        const struct pl_hook_par *hp = &hook->parameters[i];
        memcpy(hp->data, &hp->initial, sizeof(*hp->data));
    }

    if (!opts)
        return;

    const char *basename = mp_basename(shaderpath);
    struct bstr shadername;
    if (!mp_splitext(basename, &shadername))
        shadername = bstr0(basename);

    for (int n = 0; opts[n * 2]; n++) {
        struct bstr k = bstr0(opts[n * 2 + 0]);
        struct bstr v = bstr0(opts[n * 2 + 1]);
        int pos;
        if ((pos = bstrchr(k, '/')) >= 0) {
            if (!bstr_equals(bstr_splice(k, 0, pos), shadername))
                continue;
            k = bstr_cut(k, pos + 1);
        }

        for (int i = 0; i < hook->num_parameters; i++) {
            const struct pl_hook_par *hp = &hook->parameters[i];
            if (!bstr_equals0(k, hp->name) != 0)
                continue;

            m_option_t opt = {
                .name = hp->name,
            };

            if (hp->names) {
                for (int j = hp->minimum.i; j <= hp->maximum.i; j++) {
                    if (bstr_equals0(v, hp->names[j])) {
                        hp->data->i = j;
                        goto next_hook;
                    }
                }
            }

            switch (hp->type) {
            case PL_VAR_FLOAT:
                opt.type = &m_option_type_float;
                opt.min = hp->minimum.f;
                opt.max = hp->maximum.f;
                break;
            case PL_VAR_SINT:
                opt.type = &m_option_type_int;
                opt.min = hp->minimum.i;
                opt.max = hp->maximum.i;
                break;
            case PL_VAR_UINT:
                opt.type = &m_option_type_int;
                opt.min = MPMIN(hp->minimum.u, INT_MAX);
                opt.max = MPMIN(hp->maximum.u, INT_MAX);
                break;
            }

            if (!opt.type)
                goto next_hook;

            opt.type->parse(p->log, &opt, k, v, hp->data);
            goto next_hook;
        }

    next_hook:;
    }
}

static void update_render_options(struct vo *vo)
{
    struct priv *p = vo->priv;
    pl_options pars = p->pars;
    const struct gl_video_opts *opts = p->opts_cache->opts;
    pars->params.background_color[0] = opts->background_color.r / 255.0;
    pars->params.background_color[1] = opts->background_color.g / 255.0;
    pars->params.background_color[2] = opts->background_color.b / 255.0;
    pars->params.background_transparency = 1 - opts->background_color.a / 255.0;
    pars->params.skip_anti_aliasing = !opts->correct_downscaling;
    pars->params.disable_linear_scaling = !opts->linear_downscaling && !opts->linear_upscaling;
    pars->params.disable_fbos = opts->dumb_mode == 1;

    static const int map_background_types[] = {
        [BACKGROUND_NONE]  = PL_CLEAR_SKIP,
        [BACKGROUND_COLOR] = PL_CLEAR_COLOR,
        [BACKGROUND_TILES] = PL_CLEAR_TILES,
        [BACKGROUND_BLUR]  = PL_CLEAR_BLUR,
    };
    pars->params.background = map_background_types[opts->background];
    pars->params.border = map_background_types[p->next_opts->border_background];
    pars->params.blur_radius = p->next_opts->background_blur_radius;
    pars->params.tile_size = opts->background_tile_size * 2;
    for (int i = 0; i < 2; ++i) {
        pars->params.tile_colors[i][0] = opts->background_tile_color[i].r / 255.0f;
        pars->params.tile_colors[i][1] = opts->background_tile_color[i].g / 255.0f;
        pars->params.tile_colors[i][2] = opts->background_tile_color[i].b / 255.0f;
    }

    pars->params.corner_rounding = p->next_opts->corner_rounding;
    pars->params.correct_subpixel_offsets = !opts->scaler_resizes_only;

    // Map scaler options as best we can
    pars->params.upscaler = map_scaler(p, SCALER_SCALE);
    pars->params.downscaler = map_scaler(p, SCALER_DSCALE);
    pars->params.plane_upscaler = map_scaler(p, SCALER_CSCALE);
    pars->params.frame_mixer = opts->interpolation ? map_scaler(p, SCALER_TSCALE) : NULL;

    // Request as many frames as required from the decoder, depending on the
    // speed VPS/FPS ratio libplacebo may need more frames. Request frames up to
    // ratio of 1/2, but only if anti aliasing is enabled.
    int req_frames = 2;
    if (pars->params.frame_mixer) {
        req_frames += ceilf(pars->params.frame_mixer->kernel->radius) *
                      (pars->params.skip_anti_aliasing ? 1 : 2);
    }
    vo_set_queue_params(vo, 0, MPMIN(VO_MAX_REQ_FRAMES, req_frames));

    pars->params.deband_params = opts->deband ? &pars->deband_params : NULL;
    pars->deband_params.iterations = opts->deband_opts->iterations;
    pars->deband_params.radius = opts->deband_opts->range;
    pars->deband_params.threshold = opts->deband_opts->threshold / 16.384;
    pars->deband_params.grain = opts->deband_opts->grain / 8.192;

    pars->params.sigmoid_params = opts->sigmoid_upscaling ? &pars->sigmoid_params : NULL;
    pars->sigmoid_params.center = opts->sigmoid_center;
    pars->sigmoid_params.slope = opts->sigmoid_slope;

    pars->params.peak_detect_params = opts->tone_map.compute_peak >= 0 ? &pars->peak_detect_params : NULL;
    pars->peak_detect_params.smoothing_period = opts->tone_map.decay_rate;
    pars->peak_detect_params.scene_threshold_low = opts->tone_map.scene_threshold_low;
    pars->peak_detect_params.scene_threshold_high = opts->tone_map.scene_threshold_high;
    pars->peak_detect_params.percentile = opts->tone_map.peak_percentile;
    pars->peak_detect_params.allow_delayed = p->next_opts->delayed_peak;

    const struct pl_tone_map_function * const tone_map_funs[] = {
        [TONE_MAPPING_AUTO]     = &pl_tone_map_auto,
        [TONE_MAPPING_CLIP]     = &pl_tone_map_clip,
        [TONE_MAPPING_MOBIUS]   = &pl_tone_map_mobius,
        [TONE_MAPPING_REINHARD] = &pl_tone_map_reinhard,
        [TONE_MAPPING_HABLE]    = &pl_tone_map_hable,
        [TONE_MAPPING_GAMMA]    = &pl_tone_map_gamma,
        [TONE_MAPPING_LINEAR]   = &pl_tone_map_linear,
        [TONE_MAPPING_SPLINE]   = &pl_tone_map_spline,
        [TONE_MAPPING_BT_2390]  = &pl_tone_map_bt2390,
        [TONE_MAPPING_BT_2446A] = &pl_tone_map_bt2446a,
        [TONE_MAPPING_ST2094_40] = &pl_tone_map_st2094_40,
        [TONE_MAPPING_ST2094_10] = &pl_tone_map_st2094_10,
    };

    const struct pl_gamut_map_function * const gamut_modes[] = {
        [GAMUT_AUTO]            = pl_color_map_default_params.gamut_mapping,
        [GAMUT_CLIP]            = &pl_gamut_map_clip,
        [GAMUT_PERCEPTUAL]      = &pl_gamut_map_perceptual,
        [GAMUT_RELATIVE]        = &pl_gamut_map_relative,
        [GAMUT_SATURATION]      = &pl_gamut_map_saturation,
        [GAMUT_ABSOLUTE]        = &pl_gamut_map_absolute,
        [GAMUT_DESATURATE]      = &pl_gamut_map_desaturate,
        [GAMUT_DARKEN]          = &pl_gamut_map_darken,
        [GAMUT_WARN]            = &pl_gamut_map_highlight,
        [GAMUT_LINEAR]          = &pl_gamut_map_linear,
    };

    pars->color_map_params.tone_mapping_function = tone_map_funs[opts->tone_map.curve];
AV_NOWARN_DEPRECATED(
    pars->color_map_params.tone_mapping_param = opts->tone_map.curve_param;
    if (isnan(pars->color_map_params.tone_mapping_param)) // vo_gpu compatibility
        pars->color_map_params.tone_mapping_param = 0.0;
)
    pars->color_map_params.inverse_tone_mapping = opts->tone_map.inverse;
    pars->color_map_params.contrast_recovery = opts->tone_map.contrast_recovery;
    pars->color_map_params.visualize_lut = opts->tone_map.visualize;
    pars->color_map_params.contrast_smoothness = opts->tone_map.contrast_smoothness;
    pars->color_map_params.gamut_mapping = gamut_modes[opts->tone_map.gamut_mode];

    pars->params.dither_params = NULL;
    pars->params.error_diffusion = NULL;

    switch (opts->dither_algo) {
    case DITHER_ERROR_DIFFUSION:
        pars->params.error_diffusion = pl_find_error_diffusion_kernel(opts->error_diffusion);
        if (!pars->params.error_diffusion) {
            MP_WARN(p, "Could not find error diffusion kernel '%s', falling "
                    "back to fruit.\n", opts->error_diffusion);
        }
        MP_FALLTHROUGH;
    case DITHER_ORDERED:
    case DITHER_FRUIT:
        pars->params.dither_params = &pars->dither_params;
        pars->dither_params.method = opts->dither_algo == DITHER_ORDERED
                                ? PL_DITHER_ORDERED_FIXED
                                : PL_DITHER_BLUE_NOISE;
        pars->dither_params.lut_size = opts->dither_size;
        pars->dither_params.temporal = opts->temporal_dither;
        break;
    }

    if (opts->dither_depth < 0) {
        pars->params.dither_params = NULL;
        pars->params.error_diffusion = NULL;
    }

    update_icc_opts(p, opts->icc_opts);

    pars->params.num_hooks = 0;
    const struct pl_hook *hook;
    for (int i = 0; opts->user_shaders && opts->user_shaders[i]; i++) {
        if ((hook = load_hook(p, opts->user_shaders[i]))) {
            MP_TARRAY_APPEND(p, p->hooks, pars->params.num_hooks, hook);
            update_hook_opts(p, opts->user_shader_opts, opts->user_shaders[i], hook);
        }
    }

    pars->params.hooks = p->hooks;

    MP_DBG(p, "Render options updated, flushing renderer cache.\n");
    p->flush_cache = p->paused || !p->next_opts->inter_preserve;
}

const struct vo_driver video_out_gpu_next = {
    .description = "Video output based on libplacebo",
    .name = "gpu-next",
    .caps = VO_CAP_ROTATE90 |
            VO_CAP_FILM_GRAIN |
            VO_CAP_VFLIP |
            0x0,
    .preinit = preinit,
    .query_format = query_format,
    .reconfig = reconfig,
    .control = control,
    .get_image_ts = get_image,
    .draw_frame = draw_frame,
    .flip_page = flip_page,
    .get_vsync = get_vsync,
    .wait_events = wait_events,
    .wakeup = wakeup,
    .uninit = uninit,
    .priv_size = sizeof(struct priv),
};
