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
    pl_tex inter_tex; // capped-resolution composite target (see update_overlays)
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

// WP-E: glyph atlas is split into GC_SEGMENTS horizontal eviction segments. When
// the active packing region can't fit a glyph, the packer advances (ring-style)
// into the next segment and evicts ONLY that segment's stale entries (epoch bump)
// instead of ever flushing the whole atlas. 8 per the WP-E spec; a glyph taller
// than one segment simply spans several and is evicted when the FIRST of them is
// recycled (it is tagged to its top segment, which -- in ring sweep order -- is
// always the earliest of its segments to be recycled again).
#define GC_SEGMENTS 8

// WP-E: one-time worst-frame preallocation sizes for the async upload rings, so
// staging-grow / overlay-buf-grow stay 0 after the first frame (the grow paths
// remain as counted safety fallbacks). Derived from a dense-frame upper bound:
//   glyph staging holds one frame's r8 cache-miss coverage (repacked tight); a
//   very dense sign frame uploads at most a few MiB, 16 MiB is comfortable.
//   the overlay ring holds one plain-LIBASS packed sub atlas (r8/bgra); 8 MiB
//   covers an 8Kx1K r8 dialogue atlas or a 2Kx1K bgra overlay.
#define GLYPH_STAGE_MAX_BYTES (16u << 20)
#define OVERLAY_BUF_MAX_BYTES  (8u << 20)

// WP-H5a: display-derived worst-case sizing for the raster/compose pools that
// gc_prealloc_pools preallocates so they never grow mid-playback (the round-2
// 533ms "other" stall class). All are functions of the output area, so they
// scale with resolution; the counted grow fallback (raster-pool-grow) catches
// any content that still exceeds them (then bump these). Calibrated against the
// dense 8K corpus (dense_ep09 / dense_signs / mixed_8k) with generous headroom.
//   RUN_MARGIN    -- per-run scratch halo (both sides) for the blur/\be expand;
//                    512 covers a full-screen sign run with a wide blur.
//   COVER_K       -- coverage the raster writes in one frame as a multiple of
//                    the screen area (fill+border+shadow layers over the same
//                    pixels): bounds the raster-tile / segment count.
//   SEGS_PER_TILE -- avg outline segments touching one 16x16 tile (edge_tex).
//   RESULT_H_MULT -- per-sub result-atlas height as a multiple of the screen
//                    height (runs shelf-packed at 4096 width). Measured dense
//                    real content (kobayashi ~0.3x, animated ep09 text ~0.75x)
//                    packs to well under 1x; a sudden 200-sign+churn density
//                    wall packs to ~3.25x. 4x covers that with headroom (capped
//                    at the GPU limit, ~3.8x of an 8K screen). Even denser
//                    stress subs can still exceed it -- then the counted grow
//                    fallback fires (raise this).
#define RASTER_RUN_MARGIN    512
#define RASTER_COVER_K       4
#define RASTER_SEGS_PER_TILE 8
#define RASTER_RESULT_H_MULT 4

struct osd_state {
    struct osd_entry entries[MAX_OSD_PARTS];
    struct pl_overlay overlays[MAX_OSD_PARTS];
};

// WP-E3 never-block presentation guard: the overlay state is double-buffered
// (ping-pong). Every frame builds into the buffer NOT holding the last
// complete build, and commits it only when the build ran to completion. If
// the per-frame deadline (--sub-present-guard-ms) expires mid-build, the
// partial buffer is abandoned and the previous complete build -- which no
// build step has touched -- is presented instead (subs at most one frame
// stale). This is the "defer mutation-commit to the end" design: it makes
// every checkpoint boundary safe without restructuring compose_glyph_runs,
// because all state a presented overlay references (entry->tex/blur_tex/
// inter_tex/result_tex, parts arrays, the pl_overlay array) is per-buffer.
// Shared caches the build does touch (glyph atlas, run_* scratch, upload
// rings) are never referenced by a committed overlay: the atlas is only a
// compose-time *source* (resolved coverage is copied into the per-buffer
// result_tex), and the scratch/ring contents are consumed within the build.
struct osd_guard {
    struct osd_state states[2];
    int good;                 // index of the last complete build; -1 = none
    int good_num;             // its number of overlays
    double good_pts;          // video pts it was built for (validity window)
    uint64_t good_epoch;      // osd_sub_track_epoch() at build time
    struct mp_osd_res good_res; // geometry it was built for
    int good_flags;           // osd_render flags it was built with
    enum pl_overlay_coords good_coords;
    // Forward-progress guarantee: a guard fire sets this, and the NEXT build
    // for this consumer ignores the deadline and runs to completion. Without
    // it, content that is systematically over-deadline would bail every
    // frame and freeze the subs at the last committed build; with it the
    // worst case degrades to committing every other frame, keeping the
    // "at most one frame stale" invariant. (The guard targets rare transient
    // stalls, where this flag simply never matters.)
    bool must_complete;
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
    pl_renderer osd_rr; // dedicated renderer for the capped-res overlay composite
    pl_dispatch osd_dp; // for the deferred-blur compute passes
    bool osd_blur_unsupported; // logged once if the format can't be blurred
    pl_fmt osd_inter_fmt; // RGBA target for the capped-res overlay composite (NULL = disabled)
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
    pl_fmt osd_acc_fmt;        // r32f (storable) for the combine accumulator
    pl_tex run_acc;            // r32f combine accumulator (sized to max run)
    pl_tex run_tmp;            // r32f blur H->V intermediate (14-bit-equiv precision)
    pl_tex run_cov_f, run_cov_b; // r8 per-run fill/border coverage (pre-copy)

    // Single-entry cache for the per-sigma blur tap weights (see blur_weights):
    // sigma is constant across an event's runs, so recompute only on change.
    float blur_cache_sigma;    // -1 = empty
    int blur_cache_use;        // 1 = use the cascade FIR weights below, 0 = analytic
    float blur_cache_w[9];     // cascade half-kernel wt[0..8] (0 padded)

    // Persistent GPU glyph cache (Stage B): each unique glyph (keyed by libass
    // cache_id) is uploaded once into glyph_atlas, so the per-frame upload is
    // just the cache misses instead of the whole packed atlas.
    pl_tex glyph_atlas;
    // Open-addressed id->slot table (cap = pow2). A slot is live iff its `gen`
    // still matches its segment's current generation (gseg_gen[seg]); a segment
    // recycle bumps that generation, invalidating all its slots in O(1) without
    // touching the table. Stale slots stay OCCUPIED (never emptied) so probe
    // chains never break, and are reused in place on the next insert.
    struct gcache_slot { uint64_t id; int ax, ay, w, h; uint32_t gen; int seg; } *gcache;
    int gcache_cap;                          // open-addressed table (cap = pow2)
    int gatlas_w, gatlas_h, gsx, gsy, growh; // skyline allocator cursor (ring over atlas)
    // WP-E epoch-segmented eviction state (replaces the watermark full-flush).
    int gseg_h, gnsegs;                      // segment height (px) and count (<=GC_SEGMENTS)
    uint32_t gseg_gen[GC_SEGMENTS];          // per-segment generation (bumped on recycle)
    uint32_t gseg_pass[GC_SEGMENTS];         // gc_pass that last claimed each segment
    uint32_t gseg_pin[GC_SEGMENTS];          // WP-H1d: gc_pass that last cache-HIT
                                             // into each segment; such a segment is
                                             // never recycled later in that pass (the
                                             // hit's coverage is still to be composed)
                                             // -- the placement falls back to the
                                             // transient store instead
    uint32_t gseg_claim_wrap[GC_SEGMENTS];   // WP-H1d: gc_pass_wraps value when the
                                             // segment was claimed by the current
                                             // pass (re-entering it after another
                                             // wrap would overwrite own placements)
    uint32_t gc_pass_wraps;                  // ring wraps performed by this pass
    bool gseg_spanned[GC_SEGMENTS];          // WP-H1d: segment holds the LOWER rows
                                             // of glyph(s) tagged to an earlier
                                             // segment; gseg_count[s]==0 does NOT
                                             // mean its rows are free then (the
                                             // pre-fill's no-recycle mode must not
                                             // claim it as empty)
    int gseg_count[GC_SEGMENTS];             // live entries per segment (for evict-n)
    uint32_t gc_pass;                        // per-compose-item pass id (overcommit scope)
    int gc_pass_claims;                      // segments claimed by the current pass
    bool gc_warmed;                          // WP-E: config-time warm-up + prealloc done
    pl_buf glyph_stage[3];                   // async upload staging ring (no VO stall)
    unsigned glyph_stage_idx;
    uint8_t *gstage_cpu; size_t gstage_cpu_sz; // tight-repack scratch for misses

    // WP-H1d: per-item transient glyph store. Glyphs that are not cached --
    // above the size cap (gc_cacheable) or refused by the allocator when a
    // single item's working set exceeds the whole atlas -- are rasterized/
    // uploaded here EVERY frame instead of being silently skipped (the old
    // overcommit skip rendered runs from cpos {0,0}: wrong-glyph garbage or
    // missing text, non-deterministic with eviction history). Shelf-packed,
    // cursor reset per compose item; contents live only from an item's
    // resolve+flush to its region composes, so committed overlays never
    // reference it (guard-safe like the run_* scratch).
    pl_tex trans_atlas;
    int tr_x, tr_y, tr_rowh;                 // per-item shelf cursor

    // GPU glyph rasterizer (SUBBITMAP_LIBASS_OUTLINES): cache misses are
    // rasterized into glyph_atlas from libass's tile-export blobs instead of
    // being uploaded (only used when built with HAVE_ASS_OUTLINE_DEFERRED).
    pl_tex edge_tex;                         // rgba32f segment pool (2 texels/seg)
    float *ebuf; int ebuf_cap;               // CPU segment scratch (in float units)
    pl_tex work_tex;                         // batched raster: one 16x16 tile per workgroup
    float *wbuf; int wbuf_cap;               // CPU work-list scratch (in float units)

    struct mp_rect src, dst;
    struct mp_osd_res osd_res;
    struct osd_guard osd_guard;

    // WP-E3 present-guard runtime state (see struct osd_guard). guard_t0/
    // guard_deadline_ns are computed once per draw_frame; guard_abs is the
    // absolute bail time for the update_overlays call currently running
    // (0 = guard inactive for that call). guard_fired latches any bail this
    // frame so stale-present counts once per frame, not once per overlay set.
    // guard_presented_empty latches when a bail this frame presented NO
    // overlays (a vanish) so it is counted as guard-empty rather than
    // stale-present; sticky across the frame's overlay builds (any vanish wins).
    int64_t guard_t0;
    int64_t guard_deadline_ns;
    int64_t guard_abs;
    bool guard_fired;
    bool guard_presented_empty;

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

    // Permanent perf instrumentation (WP-A3); zero-cost when --dump-stats/-v off.
    // cnt_* are live counters (plain int64 increments); stat_* hold the last
    // value emitted as an MP_STATS "value" line (emit-on-change). stat_* are
    // seeded to -1 in preinit so the first frame always emits a baseline sample
    // (so e.g. gcache-flush==0 is assertable). first_frame_drawn gates the
    // vo-alloc-after-first-frame aggregate.
    int64_t cnt_gcache_flush, cnt_atlas_overflow, cnt_staging_grow;
    int64_t cnt_overlay_buf_grow, cnt_tex_realloc, cnt_vo_alloc_after_first;
    int64_t cnt_raster_dispatches, cnt_raster_tiles;
    // WP-H5a: any raster/compose pool (run_acc/run_tmp/run_cov_*, edge_tex,
    // work_tex, trans_atlas, per-sub result_tex) recreated mid-playback. These
    // are all preallocated to a display-derived worst case at gc_warmup, so a
    // grow here after the first frame is a VO-thread alloc stall (the round-2
    // 533ms "other" class) -- gated ==0 and fed into vo-alloc-after-first-frame.
    int64_t cnt_raster_pool_grow;
    int64_t stat_gcache_flush, stat_atlas_overflow, stat_staging_grow;
    int64_t stat_overlay_buf_grow, stat_tex_realloc, stat_vo_alloc_after_first;
    int64_t stat_raster_dispatches, stat_raster_tiles;
    int64_t stat_raster_pool_grow;
    // WP-E: epoch-eviction + warm-up counters. gcache-epoch-advance = segment
    // recycles (bounded, replaces the flush); gcache-evict-n = entries evicted
    // (cumulative); gcache-overcommit = glyphs skipped because a single item's
    // working set exceeded the whole atlas (never a stall); shader-warmups =
    // compute variants dispatched once at config time (no playback-time compile).
    int64_t cnt_gcache_epoch_advance, cnt_gcache_evict_n, cnt_gcache_overcommit;
    int64_t cnt_shader_warmups;
    int64_t stat_gcache_epoch_advance, stat_gcache_evict_n, stat_gcache_overcommit;
    int64_t stat_shader_warmups;
    // WP-E3: presentation-guard engagements. Split by what the bail served:
    //   stale-present -- the previous complete overlay state was still valid
    //                    for this frame and was presented (subs at most one
    //                    frame stale). This is the guard working as designed.
    //   guard-empty   -- the bail presented NO overlays (a visible sub vanish)
    //                    because no valid previous snapshot existed: cold start
    //                    (good == -1), post-seek/track-change (reset), a pts
    //                    discontinuity, or a >0.5 s content gap where the stale
    //                    snapshot would be wrong content. Reachable only in
    //                    those genuinely-invalid cases -- in steady-state
    //                    continuous playback the must_complete alternation keeps
    //                    the last good build <=2 frames back (well inside the
    //                    0.5 s backstop) so an overrun always serves stale.
    // Acceptance gates BOTH at 0 (guard-empty is a visible error; stale-present
    // is 1-frame-stale but still a stall we don't want on a clean run). Distinct
    // from ra-stale (dec_sub-level render-ahead staleness).
    int64_t cnt_stale_present, stat_stale_present;
    int64_t cnt_guard_empty, stat_guard_empty;
    // WP-H1b: glyphs uploaded/rasterized into the atlas by the idle pre-fill
    // (i.e. atlas-fill work moved OFF the frame that first shows them).
    int64_t cnt_prefill_glyphs, stat_prefill_glyphs;
    // WP-H1d: glyphs handled per-frame via the transient store because they
    // are above the cache size cap (giant sign glyphs; caching them evicts
    // hundreds of small glyphs for zero reuse value -- the measured 8K atlas
    // thrash). Informational; expected nonzero on heavy typesetting.
    int64_t cnt_glyph_uncached, stat_glyph_uncached;
    // WP-H1b pre-fill mode for the glyph-cache allocator: never recycle a
    // segment holding another pass's live glyphs (the current frame's working
    // set); gc_refused reports that the allocator refused for that reason.
    bool gc_no_recycle, gc_refused;
    // WP-H1b: "this frame's sub/OSD state was cheap" (osd_render change_id
    // unchanged from the previous presented frame -- empty or static subs).
    // Only such frames run the idle pre-fill.
    bool sub_state_cheap;
    int64_t last_sub_change_id;
    bool first_frame_drawn;
    // Phase wall-times (ns) for the MSGL_V [slowframe] line; only computed when
    // MSGL_V is active. The phase start/end dump-stats events themselves come
    // from stats_time_*() and are independent of these timers.
    int64_t dbg_subrender_ns, dbg_capcomp_ns, dbg_blur_ns;

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
    int sub_glyph_atlas_size;   // WP-E: persistent glyph atlas edge, created once
    int sub_glyph_atlas_height; // WP-A3 debug knob; 0 = default (see option below)
    int sub_present_guard_ms;   // WP-E3: overlay-build deadline; -1 auto, 0 off
    int sub_debug_stall_ms;     // WP-E3 debug: injected overlay-section sleep
    int sub_prefill_budget_ms;  // WP-H1b: idle glyph pre-fill budget; 0 = off
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
        // WP-E: edge length (px) of the square persistent GPU glyph atlas. The
        // atlas is created ONCE at this size at config time and never resized or
        // rebuilt during playback (epoch-segmented eviction reclaims space), so
        // no mid-playback atlas realloc can stall the VO thread. Clamped to the
        // GPU's max 2D texture size. Bigger = more glyphs cached at once (fewer
        // epoch advances) at a memory cost (NxN r8 bytes). WP-H1d: 0 = auto
        // (8192, or 16384 on 4K+ displays where the scaled working set
        // overflows 8192^2). The debug --sub-glyph-atlas-height override
        // below still wins for tests.
        {"sub-glyph-atlas-size", OPT_INT(sub_glyph_atlas_size), M_RANGE(0, 16384)},
        // DEBUG/dev only: cap the GPU glyph atlas size (pixels) at creation.
        // 0 = default (GPU max, up to 16384). A small value (e.g. 256) shrinks
        // the atlas to N x N and forces glyph-atlas overflow / cache-flush storms
        // so the perf counters (atlas-overflow, gcache-flush) can be exercised in
        // tests. Not for normal use -- a low cap makes dense subtitle frames flash.
        {"sub-glyph-atlas-height", OPT_INT(sub_glyph_atlas_height), M_RANGE(0, 16384)},
        // WP-E3: never-block presentation guard. Deadline (ms) for the
        // per-frame subtitle/OSD overlay build; on expiry the frame presents
        // the previous complete overlay state instead of waiting (counted as
        // stale-present). -1 = auto (the frame's duration), 0 = off.
        {"sub-present-guard-ms", OPT_INT(sub_present_guard_ms), M_RANGE(-1, 10000)},
        // DEBUG/dev only: sleep this many ms inside the overlay-build section
        // (after the first guard checkpoint), so the guard can be exercised
        // deterministically in tests. Not for normal use.
        {"sub-debug-stall-ms", OPT_INT(sub_debug_stall_ms), M_RANGE(0, 10000)},
        // WP-H1b: per-frame wall-clock budget (ms) for the idle GPU glyph
        // pre-fill: on frames whose sub state didn't change, upcoming
        // render-ahead frames' cache-miss glyphs are uploaded/rasterized
        // into the atlas ahead of time, so the first frame of a dense event
        // doesn't pay the whole atlas fill. 0 = off.
        {"sub-prefill-budget-ms", OPT_INT(sub_prefill_budget_ms), M_RANGE(0, 20)},
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
        .sub_glyph_atlas_size = 0,      // WP-H1d: auto (8192; 16384 on 4K+)
        .sub_present_guard_ms = -1,     // WP-E3: auto (frame duration)
        .sub_prefill_budget_ms = 3,     // WP-H1b: idle glyph pre-fill budget
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

// GPU deferred blur/composite machinery. All of it consumes the forked-libass
// deferred primitives (unblurred coverage + per-glyph runs); compiled out
// against a libass lacking those APIs, where sd_ass never advertises the modes
// and the CPU path is used. osd_blur_part is shared by both the blur-only path
// and the composite path, so it lives under either feature.
#if HAVE_ASS_BLUR_DEFERRED || HAVE_ASS_COMPOSITE_DEFERRED
// --- libass cascade tap weights ---------------------------------------------
// The analytic gaussian exp(-t^2/2s^2), sampled at integer taps and renormalised
// over its truncated support, matches libass's cascade blur to well under a LSB
// for sigma >= ~1 -- but NOT for very small sigma (e.g. \blur0.5), where the
// narrow kernel's shape diverges from libass's coefficient-table FIR by ~3.6/255
// on a hard edge (sweep: \blur0.5 bordered text = maxdiff up to 3858). For those
// small-sigma cases libass runs its cascade at level 0 -- i.e. a plain symmetric
// FIR, no down/upscaling -- so its exact effective kernel is just that FIR:
// center d = 1 - 2*sum(coeff), tap +-i = coeff[i-1]. We reproduce libass's
// find_best_method()/calc_coeff() here (double precision, at pass-build time,
// cached per sigma) and feed those weights to the blur shader in place of the
// analytic gaussian, sampled at the SAME tap positions (perf-neutral: identical
// tap count/striding/passes, only the weight values change). For sigma large
// enough to need down/upscaling (level > 0) the analytic gaussian already meets
// the bar, so we keep it (avoids porting the resampling cascade).
static void ass_calc_gauss(double *res, int n, double r2)
{
    double alpha = 0.5 / r2, mul = exp(-alpha), mul2 = mul * mul;
    double cur = sqrt(alpha / M_PI);
    res[0] = cur; cur *= mul; res[1] = cur;
    for (int i = 2; i < n; i++) { mul *= mul2; cur *= mul; res[i] = cur; }
}
static void ass_coeff_filter(double *c, int n, const double k[4])
{
    double p1 = c[1], p2 = c[2], p3 = c[3];
    for (int i = 0; i < n; i++) {
        double r = c[i]*k[0] + (p1+c[i+1])*k[1] + (p2+c[i+2])*k[2] + (p3+c[i+3])*k[3];
        p3 = p2; p2 = p1; p1 = c[i]; c[i] = r;
    }
}
static void ass_calc_matrix(double mat[8][8], const double *mf, int n)
{
    for (int i = 0; i < n; i++) {
        mat[i][i] = mf[2*i+2] + 3*mf[0] - 4*mf[i+1];
        for (int j = i+1; j < n; j++)
            mat[i][j] = mat[j][i] = mf[i+j+2] + mf[j-i] + 2*(mf[0]-mf[i+1]-mf[j+1]);
    }
    for (int k = 0; k < n; k++) {
        double z = 1/mat[k][k]; mat[k][k] = 1;
        for (int i = 0; i < n; i++) {
            if (i == k) continue;
            double m = mat[i][k]*z; mat[i][k] = 0;
            for (int j = 0; j < n; j++) mat[i][j] -= mat[k][j]*m;
        }
        for (int j = 0; j < n; j++) mat[k][j] *= z;
    }
}
static void ass_calc_coeff(double mu[8], int n, double r2, double mul)
{
    const double w = 12096;
    double kernel[4] = {
        (((+3280/w)*mul + 1092/w)*mul + 2520/w)*mul + 5204/w,
        (((-2460/w)*mul -  273/w)*mul -  210/w)*mul + 2943/w,
        (((+ 984/w)*mul -  546/w)*mul -  924/w)*mul +  486/w,
        (((- 164/w)*mul +  273/w)*mul -  126/w)*mul +   17/w,
    };
    double mat_freq[17] = { kernel[0], kernel[1], kernel[2], kernel[3] };
    ass_coeff_filter(mat_freq, 7, kernel);
    double vec_freq[12];
    ass_calc_gauss(vec_freq, n + 4, r2 * mul);
    ass_coeff_filter(vec_freq, n + 1, kernel);
    double mat[8][8] = {{0}};
    ass_calc_matrix(mat, mat_freq, n);
    double vec[8];
    for (int i = 0; i < n; i++)
        vec[i] = mat_freq[0] - mat_freq[i+1] - vec_freq[0] + vec_freq[i+1];
    for (int i = 0; i < n; i++) {
        double res = 0;
        for (int j = 0; j < n; j++) res += mat[i][j]*vec[j];
        mu[i] = res > 0 ? res : 0;
    }
}
// Fill p->blur_cache_{use,w} for `sigma` (cached; single entry). Returns whether
// the cascade FIR weights apply (level 0); otherwise the caller uses analytic.
static bool blur_weights(struct priv *p, float sigma)
{
    if (sigma == p->blur_cache_sigma)
        return p->blur_cache_use;
    p->blur_cache_sigma = sigma;
    p->blur_cache_use = 0;
    for (int i = 0; i < 9; i++) p->blur_cache_w[i] = 0;
    double r2 = (double) sigma * sigma;
    if (r2 <= 0.001)
        return false;
    int level, radius;
    double mu[8] = {0};
    if (r2 < 0.5) {
        level = 0; radius = 4;
        mu[1] = 0.085 * r2 * r2 * r2;
        mu[0] = 0.5 * r2 - 4 * mu[1];
    } else {
        double frac = frexp(sqrt(0.11569*r2 + 0.20591047), &level);
        if (level != 0)
            return false;   // needs the resampling cascade; analytic is fine there
        double mul = pow(0.25, level);
        radius = 8 - (int)((10.1525 + 0.8335*mul)*(1 - frac));
        if (radius < 4) radius = 4;
        ass_calc_coeff(mu, radius, r2, mul);
    }
    // Quantise to libass's Q16 coeff table, then normalise to floating weights;
    // half[0] = center, half[i] = tap at +-i. radius <= 8 so this fits wt[0..8].
    int coeff[8], sum = 0;
    for (int i = 0; i < radius; i++) { coeff[i] = (int)(0x10000*mu[i] + 0.5); sum += coeff[i]; }
    p->blur_cache_w[0] = (65536 - 2*sum) / 65536.0f;
    for (int i = 0; i < radius; i++) p->blur_cache_w[i+1] = coeff[i] / 65536.0f;
    p->blur_cache_use = 1;
    return true;
}

// Separable gaussian over one atlas sub-region [ox,oy,rw,rh], clamped to that
// region (so a blurred part can't read/write its atlas neighbours). sigma == 0
// degenerates to a copy. For small sigma the per-tap weight comes from libass's
// cascade FIR (uc!=0, weights wt[]); otherwise the analytic gaussian is used.
// Validated against a CPU reference via lavapipe.
static const char *const osd_blur_body_h =
    "ivec2 g = ivec2(gl_GlobalInvocationID.xy);\n"
    "if (g.x < rw && g.y < rh) {\n"
    "    int px = g.x + ox, py = g.y + oy;\n"
    "    float wt[9];\n"
    "    wt[0]=w0.x; wt[1]=w0.y; wt[2]=w0.z; wt[3]=w0.w;\n"
    "    wt[4]=w1.x; wt[5]=w1.y; wt[6]=w1.z; wt[7]=w1.w;\n"
    "    wt[8]=w2.x;\n"
    "    float acc = 0.0, wsum = 0.0;\n"
    "    for (int t = -radius; t <= radius; t++) {\n"
    "        int q = px + t;\n"
    "        if (q >= ox && q < ox+rw) {\n"
    "            float w = uc != 0 ? wt[abs(t)]\n"
    "                    : (sigma > 0.0 ? exp(-0.5*float(t*t)/(sigma*sigma)) : float(t==0));\n"
    "            acc += w * texelFetch(src, ivec2(q, py), 0).r; wsum += w;\n"
    "        }\n"
    "    }\n"
    "    imageStore(dst, ivec2(px, py), vec4(acc / wsum, 0.0, 0.0, 0.0));\n"
    "}\n";
// The V (final) pass writes 8-bit coverage. libass does NOT round-to-nearest at
// its 14->8 bit pack: ass_stripe_pack applies a 2x2 ordered dither tied to the
// bitmap's own pixel grid -- out8 = (v14 - (v14>>8) + dither) >> 6, with
// dither in {8,40,56,24} chosen by (x&1,y&1). Round-to-nearest instead leaves a
// systematic, position-correlated ~1-LSB coverage error on every blurred edge;
// across many overlapping blurred layers over a high-contrast background that
// stacks into visible outliers (kobayashi). Because the GPU raster is bit-exact
// with libass, the run-local (atlas) grid IS libass's combined-bitmap grid, so
// gl_GlobalInvocationID parity matches libass's bitmap-local dither phase.
// sigma==0 is the unblurred (bordered) fill: libass never packs it, so keep it a
// bit-exact copy (no dither).
static const char *const osd_blur_body_v =
    "ivec2 g = ivec2(gl_GlobalInvocationID.xy);\n"
    "if (g.x < rw && g.y < rh) {\n"
    "    int px = g.x + ox, py = g.y + oy;\n"
    "    float wt[9];\n"
    "    wt[0]=w0.x; wt[1]=w0.y; wt[2]=w0.z; wt[3]=w0.w;\n"
    "    wt[4]=w1.x; wt[5]=w1.y; wt[6]=w1.z; wt[7]=w1.w;\n"
    "    wt[8]=w2.x;\n"
    "    float acc = 0.0, wsum = 0.0;\n"
    "    for (int t = -radius; t <= radius; t++) {\n"
    "        int q = py + t;\n"
    "        if (q >= oy && q < oy+rh) {\n"
    "            float w = uc != 0 ? wt[abs(t)]\n"
    "                    : (sigma > 0.0 ? exp(-0.5*float(t*t)/(sigma*sigma)) : float(t==0));\n"
    "            acc += w * texelFetch(src, ivec2(px, q), 0).r; wsum += w;\n"
    "        }\n"
    "    }\n"
    "    float cf = acc / wsum, outv;\n"
    "    if (sigma > 0.0) {\n"
    "        int v14 = clamp(int(cf * 16384.0 + 0.5), 0, 16384);\n"
    "        int dth = (g.y & 1) == 0 ? ((g.x & 1) == 0 ? 8 : 40)\n"
    "                                 : ((g.x & 1) == 0 ? 56 : 24);\n"
    "        outv = float((v14 - (v14 >> 8) + dth) >> 6) / 255.0;\n"
    "    } else {\n"
    "        outv = cf;\n"
    "    }\n"
    "    imageStore(dst, ivec2(px, py), vec4(outv, 0.0, 0.0, 0.0));\n"
    "}\n";

static void osd_blur_part(struct priv *p, pl_tex src, pl_tex dst,
                          int ox, int oy, int rw, int rh, float sigma,
                          const char *body)
{
    int radius = (int)(3.0f * sigma + 0.999f); // ~ceil(3*sigma); 0 when sigma==0
    // Cascade FIR tap weights for small sigma (else analytic). radius <= 8
    // whenever these apply (level 0), so wt[0..8] covers every tap.
    int uc = blur_weights(p, sigma) ? 1 : 0;
    float w0[4] = { p->blur_cache_w[0], p->blur_cache_w[1], p->blur_cache_w[2], p->blur_cache_w[3] };
    float w1[4] = { p->blur_cache_w[4], p->blur_cache_w[5], p->blur_cache_w[6], p->blur_cache_w[7] };
    float w2[4] = { p->blur_cache_w[8], 0, 0, 0 };
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
        { .var = pl_var_int("radius"), .data=&radius },
        { .var = pl_var_float("sigma"),.data=&sigma },
        { .var = pl_var_int("uc"),     .data=&uc },
        { .var = pl_var_vec4("w0"),    .data=w0 },
        { .var = pl_var_vec4("w1"),    .data=w1 },
        { .var = pl_var_vec4("w2"),    .data=w2 },
    };
    struct pl_custom_shader cs = {
        .input = PL_SHADER_SIG_NONE, .output = PL_SHADER_SIG_NONE,
        .compute = true, .compute_group_size = {16, 16},
        .descriptors = descs, .num_descriptors = 2,
        .variables = vars, .num_variables = 10,
        .body = body,
    };
    if (pl_shader_custom(sh, &cs)) {
        pl_dispatch_compute(p->osd_dp, pl_dispatch_compute_params(
            .shader = &sh, .dispatch_size = { (rw+15)/16, (rh+15)/16, 1 }));
    } else {
        pl_dispatch_abort(p->osd_dp, &sh);
    }
}
#endif // HAVE_ASS_BLUR_DEFERRED || HAVE_ASS_COMPOSITE_DEFERRED

#if HAVE_ASS_COMPOSITE_DEFERRED
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
static const char *const osd_fixoutline_body =
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

static void osd_copy(struct priv *p, pl_tex src, pl_tex dst, int dx, int dy,
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

struct gpos { int ax, ay; int t; };   // a glyph's position; t=1: in the per-item
                                      // transient store instead of the atlas

// WP-E: an entry is live iff its generation still matches its segment's current
// generation. A segment recycle bumps that generation, so all its entries become
// stale in O(1) -- no table scan, no memset.
static inline bool gc_slot_live(struct priv *p, const struct gcache_slot *s)
{
    return s->id && s->gen == p->gseg_gen[s->seg];
}

// WP-E: full reset is INIT/UNINIT ONLY now (epoch eviction handles the steady
// state); it must never run mid-playback (gcache-flush stays 0 forever after
// init). Clears the table, the ring cursor, and all segment epoch state.
static void gcache_reset(struct priv *p)
{
    if (p->gcache)
        memset(p->gcache, 0, p->gcache_cap * sizeof(p->gcache[0]));
    p->gsx = p->gsy = p->growh = 0;
    p->gc_pass = 0;
    p->gc_pass_claims = 0;
    for (int s = 0; s < GC_SEGMENTS; s++) {
        p->gseg_gen[s] = 0;
        p->gseg_pass[s] = 0;
        p->gseg_pin[s] = 0;
        p->gseg_spanned[s] = false;
        p->gseg_claim_wrap[s] = 0;
        p->gseg_count[s] = 0;
    }
    p->gc_pass_wraps = 0;
}

// WP-E: allocate a ring slot for a glyph (padded nw x nh), recycling only the
// segment(s) the glyph lands in when they still hold a previous pass's entries.
// Never flushes. WP-H1d: segments that must not be recycled right now -- ones
// this pass cache-HIT into (their coverage is still to be composed), and, in
// the pre-fill's no-recycle mode, any segment with live coverage -- are
// SKIPPED: the cursor jumps past them and tries the next segment, so a hot
// pinned segment (e.g. long-lived dialogue glyphs re-hit every frame) doesn't
// wedge the ring and force every later placement into the transient store.
// Returns false (overcommit -> the caller falls back to the transient store)
// only when no segment can take the glyph this pass -- also covering the
// historical case of a pass wrapping around onto its OWN placements.
// On success returns the position and the glyph's home (top) segment.
static bool gc_place(struct priv *p, int nw, int nh, int *gx, int *gy, int *seg)
{
    if (nh > p->gatlas_h)
        return false;                         // taller than the whole atlas
    for (int attempt = 0; attempt < 2 * p->gnsegs + 2; attempt++) {
        if (p->gsx + nw > p->gatlas_w) { p->gsy += p->growh; p->gsx = 0; p->growh = 0; }
        if (p->gsy + nh > p->gatlas_h) {
            p->gsy = 0; p->gsx = 0; p->growh = 0; // ring wrap back to the top
            p->gc_pass_wraps++;
        }
        // Clamp both ends: gseg_h*gnsegs may be < gatlas_h (integer remainder),
        // so a glyph in the last few rows would otherwise index a segment past
        // gnsegs-1.
        int s0 = MPMIN(p->gsy / p->gseg_h, p->gnsegs - 1);
        int s1 = MPMIN((p->gsy + nh - 1) / p->gseg_h, p->gnsegs - 1);
        // First check every segment the glyph's rows touch; a protected one
        // makes the cursor jump past it and retry (bounded by `attempt`).
        int skip = -1;
        for (int s = s0; s <= s1; s++) {
            if (p->gseg_pass[s] == p->gc_pass) {
                // Appending within a segment this pass claimed since the last
                // ring wrap is fine (the cursor moves monotonically within
                // it); re-entering a claim from BEFORE a wrap would overwrite
                // this pass's OWN earlier placements (the historical
                // overcommit case) -- skip it instead.
                if (p->gseg_claim_wrap[s] != p->gc_pass_wraps) {
                    skip = s;
                    break;
                }
                continue;
            }
            // WP-H1b/H1d: the pre-fill's no-recycle mode may only claim
            // segments whose rows hold NO live coverage: neither their own
            // entries (gseg_count) nor the lower rows of glyphs tagged to an
            // earlier segment (gseg_spanned -- those don't show in this
            // segment's count, and rastering over them corrupted still-live
            // cached glyphs).
            if (p->gc_no_recycle && (p->gseg_count[s] > 0 || p->gseg_spanned[s])) {
                skip = s;
                break;
            }
            // WP-H1d: never recycle a segment this pass already cache-HIT
            // into -- the hit's coverage is resolved into cpos[] but not yet
            // composed, so evicting it would compose freshly rastered other-
            // glyph data (one of the pressure corruption paths behind the
            // missing-text artifact).
            if (p->gseg_pin[s] == p->gc_pass && p->gseg_count[s] > 0) {
                skip = s;
                break;
            }
        }
        if (skip >= 0) {
            p->gsy = (skip + 1) * p->gseg_h;  // jump past the protected segment
            p->gsx = 0;
            p->growh = 0;
            continue;
        }
        // Claim/recycle the spanned segments, then place at the cursor.
        for (int s = s0; s <= s1; s++) {
            if (p->gseg_pass[s] == p->gc_pass)
                continue;                     // appending within a claimed segment
            // recycle: evict this segment's (previous-pass) live entries in O(1)
            p->cnt_gcache_evict_n += p->gseg_count[s];
            p->gseg_count[s] = 0;
            p->gseg_spanned[s] = false;       // its rows are free again (WP-H1d)
            p->gseg_gen[s]++;
            p->gseg_pass[s] = p->gc_pass;
            p->gseg_claim_wrap[s] = p->gc_pass_wraps;
            p->gc_pass_claims++;
            p->cnt_gcache_epoch_advance++;
        }
        *gx = p->gsx; *gy = p->gsy; *seg = s0;
        p->gsx += nw;
        if (nh > p->growh) p->growh = nh;
        p->gseg_count[s0]++;
        for (int s = s0 + 1; s <= s1; s++)
            p->gseg_spanned[s] = true;        // lower rows in use; see gseg_spanned
        return true;
    }
    return false;                             // nowhere to place it this pass
}
#endif // HAVE_ASS_COMPOSITE_DEFERRED (resumes at gcache_reserve)

// WP-A3: bump a growth/realloc counter and, when it happens after the first
// successfully drawn frame, also the aggregate vo-alloc-after-first-frame
// counter. These are the VO-thread stalls the 8K effort chased (buffer/texture
// reallocs, cache flushes, atlas rebuilds -- see the post-mortem tail spikes).
static void vo_alloc_bump(struct priv *p, int64_t *counter)
{
    (*counter)++;
    if (p->first_frame_drawn)
        p->cnt_vo_alloc_after_first++;
}

// WP-A3: emit "value <n> <name>" to --dump-stats, but only when the counter
// changed since the last emission (MP_STATS no-ops unless --dump-stats is
// active; the caller seeds *last = -1 so one baseline sample is always emitted).
static void emit_counter(struct vo *vo, const char *name, int64_t cur, int64_t *last)
{
    if (cur != *last) {
        *last = cur;
        MP_STATS(vo, "value %lld %s", (long long) cur, name);
    }
}

// WP-E3 guard checkpoint: true when the overlay build blew its deadline and
// should bail at this (safe) boundary. Zero overhead when the guard is
// inactive (guard_abs == 0: option off, must_complete build, or a
// non-presentation caller like video_screenshot) -- no clock read.
static inline bool sub_guard_expired(struct priv *p)
{
    return p->guard_abs && mp_time_ns() > p->guard_abs;
}

// WP-H1c: (w, h) floor for the legacy-overlay textures (the packed-atlas
// fallback path: OSD, image subs, non-deferred libass), derived from the
// display size instead of the historical fixed 2048 -- at 8K the first OSD
// overlay's packed_w/h exceeds 2048 and used to recreate the texture
// mid-playback (the measured tex-realloc=1..2 per scene). Width is the next
// power of two >= the display width because the bitmap packer's internal
// atlas width grows in powers of two and a row can fill up to it, so
// packed_w may exceed the display width; height uses the display height
// (256-aligned) -- packed_h is the packer's USED height, bounded by the
// (screen-clipped) part areas. Both keep the old 2048 minimum so sub-4K
// displays behave exactly as before, and are capped by the GPU limit. A
// truly oversized overlay still grows once via the counted fallback.
static void overlay_tex_floor(struct priv *p, int *fw, int *fh)
{
    int maxd = p->gpu ? p->gpu->limits.max_tex_2d_dim : 2048;
    int w = 2048;
    while (w < p->osd_res.w && w * 2 <= maxd)
        w <<= 1;
    int h = MPMAX(2048, (p->osd_res.h + 255) & ~255);
    *fw = MPMIN(w, maxd);
    *fh = MPMIN(h, maxd);
}

#if HAVE_ASS_COMPOSITE_DEFERRED
// WP-H5a: gc_ensure for the raster/compose pools that historically grew on
// demand mid-playback (run_acc/run_tmp/run_cov_*, edge_tex, work_tex,
// trans_atlas, per-sub result_tex). They are all preallocated to a display-
// derived worst case at gc_warmup (gc_prealloc_pools), so the hot-path call
// no-ops. If content still outgrows the prealloc, this counted fallback grows
// it AND bumps raster-pool-grow (a mid-playback pl_tex_recreate = VO-thread
// stall of the round-2 533ms class). A real (re)allocation is detected by a
// dimension change, so the first (warm-up) allocation via plain gc_ensure does
// not double-count here.
static bool gc_ensure_pool(struct priv *p, pl_tex *t, pl_fmt fmt, int w, int h,
                           bool storable, bool sampleable, bool blit_src,
                           bool blit_dst, bool host_writable)
{
    int pw = *t ? (*t)->params.w : -1;
    int ph = *t ? (*t)->params.h : -1;
    bool ok = gc_ensure(p->gpu, t, fmt, w, h, storable, sampleable,
                        blit_src, blit_dst, host_writable);
    if (ok && *t && ((*t)->params.w != pw || (*t)->params.h != ph))
        vo_alloc_bump(p, &p->cnt_raster_pool_grow);
    return ok;
}

// Reserve a glyph's slot in the persistent atlas. Cache hit (live entry, same
// size): *upload stays false. Miss: a ring slot is allocated + the id inserted,
// *upload set true (the caller uploads/rasterizes the pixels). Returns false
// only when no slot could be claimed this pass (bigger than the atlas, or
// overcommit -- the current item's working set exceeded the whole atlas);
// WP-H1d: the caller then falls back to the per-item transient store
// (gc_trans_place) so the glyph is still drawn -- NEVER skipped (the old skip
// was silent content loss). The overcommit counter thus now means "took the
// lossless per-frame transient path for a big/overflow glyph" (proven pixel-
// identical to CPU under atlas pressure), NOT content loss. Post-H1d it is an
// ungated tuning signal in acceptance (INFO, not gated ==0): dense 8K signs
// legitimately trip it; its only cost is per-frame re-raster, already gated by
// the video-draw budget. Nonzero => raise --sub-glyph-atlas-size if any frames
// are over budget.
//
// One open-addressed probe locates the id (if present) and the first reusable
// (empty or stale) slot. Stale slots act as reusable tombstones that stay
// occupied, so probe chains never break and the table needs no deletion pass.
static bool gcache_reserve(struct priv *p, uint64_t id, int w, int h,
                           struct gpos *out, bool *upload)
{
    *upload = false;
    if (!p->gcache_cap)
        return false;
    uint64_t mask = p->gcache_cap - 1;
    uint64_t h0 = id & mask;
    int found = -1, reuse = -1;
    for (int probe = 0; probe < p->gcache_cap; probe++) {
        uint64_t hh = (h0 + probe) & mask;
        struct gcache_slot *s = &p->gcache[hh];
        if (!s->id) { if (reuse < 0) reuse = (int) hh; break; }   // empty: id absent
        if (s->id == id) { found = (int) hh; break; }
        if (reuse < 0 && s->gen != p->gseg_gen[s->seg])
            reuse = (int) hh;                                     // stale: reusable
    }
    if (found >= 0) {
        struct gcache_slot *s = &p->gcache[found];
        if (gc_slot_live(p, s) && s->w == w && s->h == h) {
            // WP-H1d: pin every segment the hit's rows span for this pass, so
            // a later placement can't recycle the coverage out from under the
            // pending compose (gc_place refuses and overflows transiently).
            int ps0 = MPMIN(s->ay / p->gseg_h, p->gnsegs - 1);
            int ps1 = MPMIN((s->ay + h) / p->gseg_h, p->gnsegs - 1);
            for (int ps = ps0; ps <= ps1; ps++)
                p->gseg_pin[ps] = p->gc_pass;
            out->ax = s->ax; out->ay = s->ay;
            return true;                                          // live cache hit
        }
        reuse = found;   // stale (or wrong-size) same-id slot: re-place in place
    }
    int nw = w + 1, nh = h + 1;      // 1px pad against bilinear bleed
    if (nw > p->gatlas_w || nh > p->gatlas_h || reuse < 0) {
        if (p->gc_no_recycle) {      // WP-H1b: a refused pre-fill isn't overcommit
            p->gc_refused = true;
            return false;
        }
        p->cnt_gcache_overcommit++;  // bigger than atlas, or table genuinely full
        return false;
    }
    int gx, gy, seg;
    if (!gc_place(p, nw, nh, &gx, &gy, &seg)) {
        if (p->gc_no_recycle) {      // WP-H1b: atlas tight; leave it to the frame
            p->gc_refused = true;
            return false;
        }
        p->cnt_gcache_overcommit++;  // ring exhausted this pass: skip (never stall)
        return false;
    }
    p->gcache[reuse] = (struct gcache_slot){ id, gx, gy, w, h, p->gseg_gen[seg], seg };
    out->ax = gx; out->ay = gy;
    *upload = true;
    return true;
}

// WP-H1d cache admission policy: giant glyphs are never cached. A single
// 8K-scaled sign glyph can cover more area than several eviction segments;
// admitting it evicts hundreds of small (dialogue) glyphs for near-zero reuse
// value -- the measured 8K acceptance runs thrashed the atlas with thousands
// of epoch advances per scene. Anything above 1/64 of the atlas area goes to
// the per-item transient store instead (re-rasterized each frame; the GPU
// raster is cheap and the historical cache-less raster config already met
// budget).
static bool gc_cacheable(struct priv *p, int w, int h)
{
    return (int64_t) (w + 1) * (h + 1) <=
           (int64_t) p->gatlas_w * p->gatlas_h / 64;
}

// WP-H1d: allocate a slot in the per-item transient store (uncached and
// overflow glyphs; see priv.trans_atlas). Shelf packing; the cursor is reset
// at the start of every compose item. Grows the texture on demand (counted
// via raster-pool-grow -- it is pre-created at warm-up so this should never
// fire in normal playback; growth happens before the item's raster/upload flush,
// so no already-flushed coverage is lost). Only fails when a single item
// needs more transient area than the GPU's max texture allows (> 8x the 8K
// screen area of uncached coverage) -- unreachable for real content.
static bool gc_trans_place(struct priv *p, int w, int h, struct gpos *out)
{
    pl_gpu gpu = p->gpu;
    pl_fmt r8 = p->osd_fmt[SUBBITMAP_LIBASS];
    int maxd = gpu->limits.max_tex_2d_dim;
    int nw = w + 1, nh = h + 1;          // 1px pad against bilinear bleed
    if (!r8 || nw > maxd || nh > maxd)
        return false;
    int tw = p->trans_atlas ? p->trans_atlas->params.w : 0;
    int th = p->trans_atlas ? p->trans_atlas->params.h : 0;
    if (p->tr_x + nw > tw) {             // shelf advance
        p->tr_y += p->tr_rowh;
        p->tr_x = 0;
        p->tr_rowh = 0;
    }
    if (nw > tw || p->tr_y + nh > th) {
        // Grow: width at least the atlas width (and this glyph; 4096 minimum
        // so a debug-shrunk atlas doesn't force a needle-thin store), height
        // to fit the current shelf; both doubled up to the GPU limit so
        // growth is O(log) over a scene.
        int target_w = MPMIN(MPMAX(MPMAX(p->gatlas_w, nw), 4096), maxd);
        int want_w = tw, want_h = th;
        while (want_w < target_w)
            want_w = want_w ? MPMIN(want_w * 2, maxd) : target_w;
        while (want_h < p->tr_y + nh) {
            if (want_h >= maxd)
                return false;            // theoretical bound; see above
            want_h = want_h ? MPMIN(want_h * 2, maxd) : 2048;
        }
#if HAVE_ASS_OUTLINE_DEFERRED
        const bool storable = true;      // the raster batch writes coverage here
#else
        const bool storable = false;
#endif
        if (!gc_ensure_pool(p, &p->trans_atlas, r8, want_w, want_h,
                            storable, true, false, false, true))
            return false;
    }
    out->ax = p->tr_x;
    out->ay = p->tr_y;
    out->t = 1;
    p->tr_x += nw;
    p->tr_rowh = MPMAX(p->tr_rowh, nh);
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
    // Outline-mode (SUBBITMAP_LIBASS_OUTLINES) run features; all-zero otherwise.
    uint32_t clip_id;            // vector \clip mask to multiply by (0 = none)
    int rcx0, rcy0, rcx1, rcy1;  // rectangular \clip; visible render-space rect
    uint32_t fill_color2;        // \kf wipe: fill colour right of wipe_x
    int wipe_x;                  // \kf wipe boundary (render x); used iff KF_WIPE
    int be_f, be_b;              // \be [1,2,1]/4 box iterations per layer
    int shift_x, shift_y;        // shadow-run sub-pixel offset, 1/64 px (0..63)
};

// Combine a region's glyph list (saturating add) into run_acc at run-local
// coords and resolve to cov. (Blur happens later -- after fix_outline -- to
// match libass's combine -> expand -> fix_outline -> blur order.)
static void gc_build_cov(struct priv *p, const struct sub_bitmaps *item,
                         struct gc_region *r, int *parts, int n,
                         pl_tex cov, int bw, int bh, struct gpos *cpos, double gs)
{
    osd_clear(p, p->run_acc, bw, bh);
    for (int k = 0; k < n; k++) {
        const struct sub_bitmap *b = &item->parts[parts[k]];
        // Glyph coverage (b->w x b->h) is at the capped render resolution; place
        // it in the run accumulator in that same capped space (gs scales the
        // display-space part position down to it).
        if (cpos[parts[k]].t < 0)
            continue;   // unplaceable (theoretical; see gc_trans_place)
        int dx = lrint(b->x * gs) - r->x0 + r->margin;
        int dy = lrint(b->y * gs) - r->y0 + r->margin;
        // WP-H1d: uncached/overflow glyphs live in the per-item transient
        // store instead of the persistent atlas.
        pl_tex src = cpos[parts[k]].t > 0 ? p->trans_atlas : p->glyph_atlas;
        osd_combine_part(p, src, p->run_acc, cpos[parts[k]].ax,
                         cpos[parts[k]].ay, dx, dy, b->w, b->h);
    }
    osd_unop(p, p->run_acc, cov, bw, bh, "acc", false, "dst", osd_resolve_body);
}

static void gc_blur(struct priv *p, pl_tex cov, int bw, int bh, float sigma)
{
    stats_time_start(p->stats, "sub-blur");   // WP-A3: GPU blur pass (composite path)
    osd_blur_part(p, cov, p->run_tmp, 0, 0, bw, bh, sigma, osd_blur_body_h);
    osd_blur_part(p, p->run_tmp, cov, 0, 0, bw, bh, sigma, osd_blur_body_v);
    stats_time_end(p->stats, "sub-blur");
}

// Run/part flags of the forked libass ABI (ASS_Image.run_flags); see the
// FilterDesc flags in the fork's ass_render.c. Bit 0 (fix_outline) is tested
// directly as `run_flags & 1` by the composite path above.
#define RUN_FLAG_CLIP_MASK    0x2  // libass ABI: this part is a vector-clip mask
#define RUN_FLAG_CLIP_INVERSE 0x4  // the clip mask is inverse (\iclip)
#define RUN_FLAG_KF_WIPE      0x8  // \kf fill: fill_color left of wipe_x, color2 right
#define RUN_FLAG_RECT_INVERSE 0x10 // \iclip rect: the clip rect is EXCLUDED, not visible
#define RUN_FLAG_SHADOW       0x20 // drop shadow: draw behind border+fill

#if HAVE_ASS_OUTLINE_DEFERRED
// --- GPU glyph rasterizer (SUBBITMAP_LIBASS_OUTLINES) -----------------------
// Exact-area analytic coverage from a glyph's line segments, written into the
// glyph atlas slot -- a faithful GLSL port of libass's tile fillers, driven by
// libass's own tile-split export (ass_outline_to_tiles). All arithmetic is
// int32, replicating the CPU filler's truncating integer ops (>>2, >>3, >>7,
// >>16), so the coverage matches libass's CPU raster bit for bit (including
// stroke self-intersections).
#define EDGE_TEX_W 8192   // edge_tex/work_tex width; element i is at (i%W, i/W).
#define WORK_TEX_W 8192   // Wide so the height (count/W) stays under max_tex_2d_dim at 8K.
#define TILE_EXPORT_W 11  // libass tile-export: tx,ty,ng,group0[4],group1[4] (matches ass_rasterizer.h)
#define SEG_EXPORT_W  8   // libass tile-export: a,b,c,flags,xmin,ymin,ymax,_ (2 rgba32f texels)
// Faithful integer port of libass update_border_line (rasterizer_template.h):
// the per-pixel coverage of a partial sub-pixel row span [up,dn] (1/64 units),
// FULL_VALUE=1024, TILE_ORDER=4. Used by the res/winding filler below so the
// GPU matches libass's analytic AA (incl. filling stroke self-overlaps).
static const char *const osd_raster_prelude =
    "int ubl(int px, int abs_a, int a, int b, int abs_b, int c, int up, int dn){\n"
    "  int size = dn - up;\n"
    "  int w = min(1024 + (size << 4) - abs_a, 1024) << 3;\n"
    "  int dc_b = (abs_b * size) >> 6;\n"
    "  int dc = (min(abs_a, dc_b) + 2) >> 2;\n"
    "  int base = (b * (up + dn)) >> 7;\n"
    "  int offs1 = size - (((base + dc) * w) >> 16);\n"
    "  int offs2 = size - (((base - dc) * w) >> 16);\n"
    "  int size2 = size * 2;\n"
    "  int cw = ((c - a * px) * w) >> 16;\n"
    "  int c1 = clamp(cw + offs1, 0, size2);\n"
    "  int c2 = clamp(cw + offs2, 0, size2);\n"
    "  return c1 + c2;\n"
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
    "int cov = 0;\n"
    "for (int gi = 0; gi < 2; gi++) {\n"
    "  if (gi >= ng) break;\n"
    "  vec4 G = GG[gi];\n"
    "  int type=int(G.x); int wind=int(G.y); int soff=int(G.z); int scnt=int(G.w);\n"
    "  int v = 0;\n"
    "  if (type == 0) { v = wind != 0 ? 255 : 0; }\n"                     // solid
    "  else if (type == 1) {\n"                                           // halfplane
    "    int e=soff*2; vec4 s0=texelFetch(edges, ivec2(e % ew, e / ew), 0);\n"
    "    int aa=int(s0.x), bb=int(s0.y);\n"
    "    int cc = int(s0.z) + 512 - ((aa+bb)>>1) - bb*ly;\n"
    "    int dl = (min(abs(aa),abs(bb))+2)>>2;\n"
    "    int c1=clamp(cc-aa*lx+dl,0,1024), c2=clamp(cc-aa*lx-dl,0,1024);\n"
    "    v = min((c1+c2)>>3, 255);\n"
    "  } else {\n"                                                        // generic
    "    int res=0, cur=256*wind;\n"
    "    for (int i=0;i<scnt;i++){\n"
    "      int e=(soff+i)*2;\n"
    "      vec4 s0=texelFetch(edges, ivec2(e % ew, e / ew), 0);\n"
    "      vec4 s1=texelFetch(edges, ivec2((e+1) % ew, (e+1) / ew), 0);\n"
    "      int a=int(s0.x), b=int(s0.y), c0=int(s0.z), flags=int(s0.w);\n"
    "      int xmin=int(s1.x), ymn=int(s1.y), ymx=int(s1.z);\n"
    "      int upd = (flags&1)!=0 ? 4 : 0, dnd=upd;\n"
    "      if (xmin==0 && (flags&4)!=0) dnd=4-dnd;\n"
    "      if ((flags&2)!=0){ int t=upd; upd=dnd; dnd=t; }\n"
    "      int up=ymn>>6, dn=ymx>>6, upp=ymn&63, dnp=ymx&63;\n"
    "      if (up   <= ly) cur-=(upd<<6)-upd*upp;\n"
    "      if (up+1 <= ly) cur-=upd*upp;\n"
    "      if (dn   <= ly) cur+=(dnd<<6)-dnd*dnp;\n"
    "      if (dn+1 <= ly) cur+=dnd*dnp;\n"
    "      if (ymn==ymx) continue;\n"
    "      int abs_a=abs(a), abs_b=abs(b);\n"
    "      int dc=(min(abs_a,abs_b)+2)>>2;\n"
    "      int base=512-(b>>1);\n"
    "      int c=c0 - (a>>1) - b*up, rup=up;\n"
    "      if (upp!=0){\n"
    "        if (dn==up){ if(ly==up) res+=ubl(lx,abs_a,a,b,abs_b,c,upp,dnp); continue; }\n"
    "        if (ly==up) res+=ubl(lx,abs_a,a,b,abs_b,c,upp,64); rup=up+1; c-=b;\n"
    "      }\n"
    "      if (ly>=rup && ly<dn){\n"
    "        int cy=c - b*(ly-rup);\n"
    "        int c1=clamp(cy-a*lx+base+dc,0,1024), c2=clamp(cy-a*lx+base-dc,0,1024);\n"
    "        res+=(c1+c2)>>3;\n"
    "      }\n"
    "      if (dnp!=0 && ly==dn){ int cy=c - b*(dn-rup); res+=ubl(lx,abs_a,a,b,abs_b,cy,0,dnp); }\n"
    "    }\n"
    "    int val=res+cur; v = min(max(val,-val), 255);\n"
    "  }\n"
    "  cov = gi==0 ? v : max(cov, v);\n"
    "}\n"
    "imageStore(dst, ivec2(ax+tx+lx, ay+ty+ly), vec4(float(cov)/255.0, 0.0, 0.0, 0.0));\n";

// Rasterize ALL collected glyphs in one dispatch: ntiles 16x16 work-list tiles.
// dst is the persistent glyph atlas for cache misses, or the per-item
// transient store for uncached/overflow glyphs (WP-H1d) -- same pipeline
// (compilation is keyed by the GLSL, not the bound object).
static void gc_raster_batch(struct priv *p, int ntiles, pl_tex dst)
{
    int ew = EDGE_TEX_W, ww = WORK_TEX_W;
    int gridx = MPMIN(ntiles, 32768), gridy = (ntiles + gridx - 1) / gridx;
    pl_shader sh = pl_dispatch_begin(p->osd_dp);
    struct pl_shader_desc descs[] = {
        { .desc = { .name = "dst", .type = PL_DESC_STORAGE_IMG, .binding = 0,
                    .access = PL_DESC_ACCESS_WRITEONLY }, .binding = { .object = dst } },
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

// \be edge-blur: one 3x3 [1,2,1]x[1,2,1]/16 pass over a coverage slot with a
// SINGLE truncating >>4, zero outside the slot -- integer-exact vs libass's
// ass_be_blur_c (vsfilter's kernel; NOT two separately-rounded axis passes).
static const char *const osd_be3_body =
    "ivec2 g = ivec2(gl_GlobalInvocationID.xy);\n"
    "if (g.x >= bw || g.y >= bh) return;\n"
    "int s = 0;\n"
    "for (int dy = -1; dy <= 1; dy++)\n"
    "for (int dx = -1; dx <= 1; dx++) {\n"
    "    int xx = g.x + dx, yy = g.y + dy;\n"
    "    if (xx < 0 || xx >= bw || yy < 0 || yy >= bh) continue;\n"
    "    int k = int(texelFetch(src, ivec2(ax+xx, ay+yy), 0).r * 255.0 + 0.5);\n"
    "    s += k * (2 - abs(dx)) * (2 - abs(dy));\n"
    "}\n"
    "imageStore(dst, ivec2(ax+g.x, ay+g.y), vec4(float(s >> 4) / 255.0, 0.0, 0.0, 0.0));\n";
static void gc_be_pass3(struct priv *p, pl_tex src, pl_tex dst,
                        int ax, int ay, int bw, int bh)
{
    pl_shader sh = pl_dispatch_begin(p->osd_dp);
    struct pl_shader_desc descs[] = {
        { .desc = { .name = "src", .type = PL_DESC_SAMPLED_TEX, .binding = 0 },
          .binding = { .object = src } },
        { .desc = { .name = "dst", .type = PL_DESC_STORAGE_IMG, .binding = 1,
                    .access = PL_DESC_ACCESS_WRITEONLY }, .binding = { .object = dst } },
    };
    struct pl_shader_var vars[] = {
        { .var = pl_var_int("ax"), .data = &ax }, { .var = pl_var_int("ay"), .data = &ay },
        { .var = pl_var_int("bw"), .data = &bw }, { .var = pl_var_int("bh"), .data = &bh },
    };
    struct pl_custom_shader cs = {
        .compute = true, .compute_group_size = {16, 16},
        .descriptors = descs, .num_descriptors = 2,
        .variables = vars, .num_variables = 4, .body = osd_be3_body,
    };
    if (pl_shader_custom(sh, &cs))
        pl_dispatch_compute(p->osd_dp, pl_dispatch_compute_params(
            .shader = &sh, .dispatch_size = { (bw+15)/16, (bh+15)/16, 1 }));
    else
        pl_dispatch_abort(p->osd_dp, &sh);
}
// Pointwise pre/post value scaling around multi-pass \be (libass be_blur_pre /
// be_blur_post): pre ((v>>1)+1)>>1 makes headroom for repeated integer passes,
// post (v<<2)-(v>32) maps back. In-place (no cross-texel reads).
static const char *const osd_be_scale_body =
    "ivec2 g = ivec2(gl_GlobalInvocationID.xy);\n"
    "if (g.x >= bw || g.y >= bh) return;\n"
    "ivec2 pt = ivec2(ax+g.x, ay+g.y);\n"
    "int k = int(imageLoad(img, pt).r * 255.0 + 0.5);\n"
    "k = post != 0 ? min((k << 2) - (k > 32 ? 1 : 0), 255) : (((k >> 1) + 1) >> 1);\n"
    "imageStore(img, pt, vec4(float(k) / 255.0, 0.0, 0.0, 0.0));\n";
static void gc_be_scale(struct priv *p, pl_tex img,
                        int ax, int ay, int bw, int bh, int post)
{
    pl_shader sh = pl_dispatch_begin(p->osd_dp);
    struct pl_shader_desc descs[] = {
        { .desc = { .name = "img", .type = PL_DESC_STORAGE_IMG, .binding = 0,
                    .access = PL_DESC_ACCESS_READWRITE }, .binding = { .object = img } },
    };
    struct pl_shader_var vars[] = {
        { .var = pl_var_int("ax"), .data = &ax }, { .var = pl_var_int("ay"), .data = &ay },
        { .var = pl_var_int("bw"), .data = &bw }, { .var = pl_var_int("bh"), .data = &bh },
        { .var = pl_var_int("post"), .data = &post },
    };
    struct pl_custom_shader cs = {
        .compute = true, .compute_group_size = {16, 16},
        .descriptors = descs, .num_descriptors = 1,
        .variables = vars, .num_variables = 5, .body = osd_be_scale_body,
    };
    if (pl_shader_custom(sh, &cs))
        pl_dispatch_compute(p->osd_dp, pl_dispatch_compute_params(
            .shader = &sh, .dispatch_size = { (bw+15)/16, (bh+15)/16, 1 }));
    else
        pl_dispatch_abort(p->osd_dp, &sh);
}
// `be` iterations on a coverage slot, replicating ass_synth_blur's sequence:
// pre-scale, be-1 passes, post-scale, one final pass (pre/post only for be>1).
static void gc_be_blur(struct priv *p, pl_tex res, pl_tex tmp,
                       int ax, int ay, int bw, int bh, int be)
{
    if (be > 1)
        gc_be_scale(p, res, ax, ay, bw, bh, 0);      // pre: (v+1)>>2
    pl_tex cur = res, oth = tmp;
    for (int i = 0; i < be; i++) {
        if (i == be - 1 && be > 1)
            gc_be_scale(p, cur, ax, ay, bw, bh, 1);  // post: (v<<2)-(v>32)
        gc_be_pass3(p, cur, oth, ax, ay, bw, bh);
        pl_tex t = cur; cur = oth; oth = t;
    }
    // All call sites use slot origin (0,0), which osd_copy's src side assumes.
    if (cur != res)
        osd_copy(p, cur, res, ax, ay, bw, bh);
}

// A vector-\clip mask, rasterized like a glyph at (ax,ay) -- into the glyph
// atlas, or the per-item transient store when uncached/overflow (t=1; screen-
// sized masks usually are, WP-H1d) -- with screen origin (sx,sy); multiply a
// run's coverage by it.
struct clipmask { uint32_t id; int ax, ay, sx, sy, w, h, inv, t; };

// Integer-exact vs libass's CPU clip multiply (ass_mul_bitmaps_c /
// ass_imul_bitmaps_c): out = (cov * mask + 255) >> 8, inverse uses 255-mask.
static const char *const osd_clipmult_body =
    "ivec2 g = ivec2(gl_GlobalInvocationID.xy);\n"
    "if (g.x >= bw || g.y >= bh) return;\n"
    "int mlx = ox + g.x - csx, mly = oy + g.y - csy;\n"   // clip-mask-local coord
    "int m = 0;\n"
    "if (mlx >= 0 && mlx < cw && mly >= 0 && mly < ch)\n"
    "    m = int(texelFetch(mask, ivec2(cax + mlx, cay + mly), 0).r * 255.0 + 0.5);\n"
    "if (inv != 0) m = 255 - m;\n"
    "ivec2 sp = ivec2(sox + g.x, soy + g.y);\n"
    "int c = int(imageLoad(cov, sp).r * 255.0 + 0.5);\n"
    "imageStore(cov, sp, vec4(float((c * m + 255) >> 8) / 255.0, 0.0, 0.0, 0.0));\n";

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
          .binding = { .object = cm->t ? p->trans_atlas : p->glyph_atlas } },
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

// Vector-\clip mask multiply on one run-local coverage slot (origin 0,0,
// screen origin (r->x0 - margin, r->y0 - margin)), after the blur.
static void gc_apply_clip(struct priv *p, pl_tex cov, int bw, int bh,
                          struct gc_region *r,
                          const struct clipmask *clips, int nclips)
{
    if (!r->clip_id)
        return;
    for (int c = 0; c < nclips; c++) {
        if (clips[c].id == r->clip_id) {
            gc_clip_mult(p, cov, bw, bh, 0, 0,
                         r->x0 - r->margin, r->y0 - r->margin, &clips[c]);
            return;
        }
    }
}

// Sub-pixel shadow shift: exact integer port of libass's ass_shift_bitmap
// (ass_bitmap.c). The CPU smears the (already blurred) shadow coverage right/
// down by fx/fy 64ths of a pixel with truncating fixed-point bilinear passes;
// its in-place high-to-low loop resolves to the closed form
//   t(x,y)   = C(x,y)   - (C(x,y)*fx   >> 6) + (C(x-1,y)*fx >> 6)
//   out(x,y) = t(x,y)   - (t(x,y)*fy   >> 6) + (t(x,y-1)*fy >> 6)
// with C() = 0 outside the slot. (The CPU skips the "- term" on the bitmap's
// last column/row, but coverage never reaches it: the rasterization bbox
// keeps >= 1px right/bottom slack, which blur expansion preserves -- see
// ASS_Image.shift_x64.) All values stay in 0..255, no saturation needed.
// Cost: 4 texel fetches/px + one copy-back pass, on shadow runs only.
static const char *const osd_subshift_body =
    "ivec2 g = ivec2(gl_GlobalInvocationID.xy);\n"
    "if (g.x >= bw || g.y >= bh) return;\n"
    "int c00 = int(texelFetch(src, g, 0).r * 255.0 + 0.5);\n"
    "int c10 = g.x > 0 ? int(texelFetch(src, ivec2(g.x-1, g.y), 0).r * 255.0 + 0.5) : 0;\n"
    "int c01 = g.y > 0 ? int(texelFetch(src, ivec2(g.x, g.y-1), 0).r * 255.0 + 0.5) : 0;\n"
    "int c11 = (g.x > 0 && g.y > 0) ? int(texelFetch(src, ivec2(g.x-1, g.y-1), 0).r * 255.0 + 0.5) : 0;\n"
    "int t0 = c00 - ((c00 * fx) >> 6) + ((c10 * fx) >> 6);\n"
    "int t1 = c01 - ((c01 * fx) >> 6) + ((c11 * fx) >> 6);\n"
    "int o  = t0 - ((t0 * fy) >> 6) + ((t1 * fy) >> 6);\n"
    "imageStore(dst, g, vec4(float(o) / 255.0, 0.0, 0.0, 0.0));\n";

static void gc_subshift(struct priv *p, pl_tex src, pl_tex dst,
                        int bw, int bh, int fx, int fy)
{
    pl_shader sh = pl_dispatch_begin(p->osd_dp);
    struct pl_shader_desc descs[] = {
        { .desc = { .name = "src", .type = PL_DESC_SAMPLED_TEX, .binding = 0 },
          .binding = { .object = src } },
        { .desc = { .name = "dst", .type = PL_DESC_STORAGE_IMG, .binding = 1,
                    .access = PL_DESC_ACCESS_WRITEONLY },
          .binding = { .object = dst } },
    };
    struct pl_shader_var vars[] = {
        { .var = pl_var_int("bw"), .data = &bw }, { .var = pl_var_int("bh"), .data = &bh },
        { .var = pl_var_int("fx"), .data = &fx }, { .var = pl_var_int("fy"), .data = &fy },
    };
    struct pl_custom_shader cs = {
        .compute = true, .compute_group_size = { 16, 16 },
        .descriptors = descs, .num_descriptors = 2,
        .variables = vars, .num_variables = 4, .body = osd_subshift_body,
    };
    if (pl_shader_custom(sh, &cs))
        pl_dispatch_compute(p->osd_dp, pl_dispatch_compute_params(
            .shader = &sh, .dispatch_size = { (bw + 15) / 16, (bh + 15) / 16, 1 }));
    else
        pl_dispatch_abort(p->osd_dp, &sh);
}
#endif // HAVE_ASS_OUTLINE_DEFERRED

// WP-E: create the persistent glyph atlas ONCE at its full (option) size and set
// up the id table + epoch-segment state. Called at config-time warm-up and, as a
// fallback, lazily on first compose. Never resizes/rebuilds the atlas afterwards
// -- epoch-segmented eviction reclaims space -- so no mid-playback atlas realloc
// can stall the VO thread. Returns false if the atlas can't be created.
static bool gc_ensure_atlas(struct priv *p)
{
    if (p->glyph_atlas)
        return true;
    pl_gpu gpu = p->gpu;
    pl_fmt r8 = p->osd_fmt[SUBBITMAP_LIBASS];
    if (!r8)
        return false;
    int want = p->next_opts->sub_glyph_atlas_size;   // WP-E option (0 = auto)
    if (want <= 0) {
        // WP-H1d auto default: 8192^2 r8 (64 MiB), but 16384^2 (256 MiB) on
        // 4K+ displays -- glyph dimensions scale with the output size, and
        // the 8K-scaled small/medium working set (dialogue + typical signs;
        // giant glyphs are not cached, see gc_cacheable) overflows an
        // 8192^2 atlas, which thrashed it (thousands of epoch advances per
        // scene) and overflowed into per-frame fallbacks.
        bool big = p->osd_res.w >= 3840 || p->osd_res.h >= 2160;
        want = big ? 16384 : 8192;
    }
    want = MPMIN(want, gpu->limits.max_tex_2d_dim);
    p->gatlas_w = p->gatlas_h = want;
    // WP-A3 debug override (--sub-glyph-atlas-height): shrink the atlas so tests
    // can force epoch-eviction thrash. Clamp BOTH dims (square atlas; the height
    // is the reachable binding constraint). 0 = no cap.
    if (p->next_opts->sub_glyph_atlas_height > 0) {
        int cap = p->next_opts->sub_glyph_atlas_height;
        p->gatlas_h = MPMIN(p->gatlas_h, cap);
        p->gatlas_w = MPMIN(p->gatlas_w, cap);
    }
#if HAVE_ASS_OUTLINE_DEFERRED
    const bool atlas_storable = true;   // outline rasterizer writes coverage here
#else
    const bool atlas_storable = false;
#endif
    if (!gc_ensure(gpu, &p->glyph_atlas, r8, p->gatlas_w, p->gatlas_h,
                   atlas_storable, true, false, false, true))
        return false;
    // WP-E: GC_SEGMENTS horizontal eviction segments; a glyph may span several.
    p->gnsegs = GC_SEGMENTS;
    p->gseg_h = MPMAX(p->gatlas_h / p->gnsegs, 1);
    // Size the id table generously above the max glyphs the atlas can hold at
    // once (~area / a small glyph area), so live entries never fill it (stale
    // reuse + a probe bound guard the pathological case).
    int64_t cap = (int64_t) p->gatlas_w * p->gatlas_h / 512;
    int bits = 15;                       // >= 1<<15
    while ((1 << bits) < cap && bits < 20) bits++;
    p->gcache_cap = 1 << bits;
    p->gcache = talloc_zero_array(p, struct gcache_slot, p->gcache_cap);
    gcache_reset(p);                     // init the ring cursor + segment epochs
    return true;
}

// WP-H1c: per-buffer floor for the legacy-overlay async upload ring, derived
// from the display size (packed atlas bytes scale with resolution: the ring
// stages one frame's packed r8/bgra atlas, whose dims scale with the output).
// A quarter of the floor-texture area covers a full-floor-width r8 band over
// a quarter of the display height -- comfortably above the heaviest realistic
// OSD/legacy atlas at that resolution (~2x margin over an --osd-level=3 +
// stats-page frame) without ballooning the NUM_OVERLAY_BUFS-deep ring; the
// historical 8 MiB stays as the minimum (sub-4K displays are unchanged) and
// the counted grow path remains for pathological frames.
static size_t overlay_buf_floor(struct priv *p)
{
    int fw, fh;
    overlay_tex_floor(p, &fw, &fh);
    size_t want = MPMAX((size_t) OVERLAY_BUF_MAX_BYTES, (size_t) fw * fh / 4);
    return (want + (4u << 20) - 1) & ~(size_t)((4u << 20) - 1);
}

// WP-H1c: (re)create the legacy overlay texture pool and the overlay upload
// ring at the current display-derived floor. Called from gc_warmup and again
// on every reconfig -- i.e. BEFORE playback of the (new) stream, never
// mid-playback -- so a display-size change re-floors everything while the hot
// path's pl_tex_recreate / size checks still no-op on the popped objects.
// The pool holds floor-sized textures with the hot path's exact params; WP-E3
// double-buffers the overlay state, so two concurrent legacy entries need 4.
static void ensure_overlay_pool(struct priv *p)
{
    pl_gpu gpu = p->gpu;
    if (!gpu)
        return;
    size_t bfloor = overlay_buf_floor(p);
    for (int i = 0; i < NUM_OVERLAY_BUFS; i++) {
        if (!p->overlay_bufs[i] || p->overlay_bufs[i]->params.size < bfloor)
            pl_buf_recreate(gpu, &p->overlay_bufs[i],
                            pl_buf_params(.size = bfloor, .host_writable = true));
    }
    pl_fmt r8 = p->osd_fmt[SUBBITMAP_LIBASS];
    if (!r8)
        return;
    int fw, fh;
    overlay_tex_floor(p, &fw, &fh);
    // Drop pooled (unpopped) textures below the floor; textures already
    // popped into entries keep working and only grow through the counted
    // path if content actually outgrows them.
    for (int i = p->num_sub_tex - 1; i >= 0; i--) {
        if (p->sub_tex[i]->params.w < fw || p->sub_tex[i]->params.h < fh) {
            pl_tex_destroy(gpu, &p->sub_tex[i]);
            MP_TARRAY_REMOVE_AT(p->sub_tex, p->num_sub_tex, i);
        }
    }
    while (p->num_sub_tex < 4) {
        pl_tex t = pl_tex_create(gpu, &(struct pl_tex_params) {
            .format = r8,
            .w = fw,
            .h = fh,
            .host_writable = true,
            .sampleable = true,
        });
        if (!t)
            break;
        MP_TARRAY_APPEND(p, p->sub_tex, p->num_sub_tex, t);
    }
}

// WP-H5a: preallocate the raster/compose pools that would otherwise grow on
// demand mid-playback (the round-2 533ms "other" stall was one such grow -- an
// uncounted pl_tex_recreate in the compose/raster path). Sizes derive from the
// display (osd_res), so they scale with the output resolution, and are capped
// at the GPU 2D-texture limit; the counted grow fallback (gc_ensure_pool ->
// raster-pool-grow) still catches any content that exceeds them. Called from
// gc_warmup and re-floored on every reconfig (display-size change), always OFF
// the hot path -- so the plain gc_ensure calls here never bump the counter.
static void gc_prealloc_pools(struct priv *p)
{
    pl_gpu gpu = p->gpu;
    if (!gpu)
        return;
    int maxd = gpu->limits.max_tex_2d_dim;
    pl_fmt r8  = p->osd_fmt[SUBBITMAP_LIBASS];
    pl_fmt r32 = p->osd_acc_fmt;
    bool have_r8  = r8  && (r8->caps  & PL_FMT_CAP_STORABLE);
    bool have_r32 = r32 && (r32->caps & PL_FMT_CAP_STORABLE);
    int ow = MPMAX(p->osd_res.w, 512), oh = MPMAX(p->osd_res.h, 512);

    // Per-run scratch (run_acc/run_tmp r32f; run_cov_f/run_cov_b r8): a single
    // deferred run's bbox is bounded by the screen, plus a blur/\be halo margin.
    int rw = MPMIN(ow + RASTER_RUN_MARGIN, maxd);
    int rh = MPMIN(oh + RASTER_RUN_MARGIN, maxd);
    if (have_r32) {
        gc_ensure(gpu, &p->run_acc, r32, rw, rh, true, false, false, false, false);
        gc_ensure(gpu, &p->run_tmp, r32, rw, rh, true, true,  false, false, false);
    }
    if (have_r8) {
        gc_ensure(gpu, &p->run_cov_f, r8, rw, rh, true, true, false, false, false);
        gc_ensure(gpu, &p->run_cov_b, r8, rw, rh, true, true, false, false, false);
    }

    // Per-sub result atlas (compose_glyph_runs shelf-packs an item's runs at
    // AW=4096 width): width up to the widest run; height RESULT_H_MULT x the
    // screen height (real dense content packs to well under 1x). Only the
    // subtitle entries (render_index OSDTYPE_SUB==0 / OSDTYPE_SUB2==1, the only
    // OSD objects that emit deferred glyph/outline runs) ever reach the compose
    // path, and the overlay state is double-buffered, so preallocate both
    // subtitle result_tex slots in each of states[0]/states[1].
    if (have_r8) {
        int rtw = MPMIN(MPMAX(4096, ow + RASTER_RUN_MARGIN), maxd);
        int rth = MPMIN(RASTER_RESULT_H_MULT * oh, maxd);
        for (int s = 0; s < 2; s++)
            for (int e = 0; e <= 1; e++)   // OSDTYPE_SUB, OSDTYPE_SUB2
                gc_ensure(gpu, &p->osd_guard.states[s].entries[e].result_tex,
                          r8, rtw, rth, true, true, false, false, false);
    }

#if HAVE_ASS_OUTLINE_DEFERRED
    // Outline raster pools: edge_tex holds 2 rgba32f texels/segment, work_tex 4/
    // tile (one 16x16 tile per workgroup). Worst-case tiles ~ screen area x
    // COVER_K / 256; segments ~ tiles x SEGS_PER_TILE. Widths are fixed
    // (WORK_TEX_W/EDGE_TEX_W); size the heights and cap at the GPU limit.
    pl_fmt ef = pl_find_named_fmt(gpu, "rgba32f");
    if (ef) {
        int64_t tiles = (int64_t) ow * oh * RASTER_COVER_K / 256;
        int wh = (int) MPMIN((4 * tiles + WORK_TEX_W - 1) / WORK_TEX_W, (int64_t) maxd);
        int64_t segs = tiles * RASTER_SEGS_PER_TILE;
        int eh = (int) MPMIN((2 * segs + EDGE_TEX_W - 1) / EDGE_TEX_W, (int64_t) maxd);
        gc_ensure(gpu, &p->work_tex, ef, WORK_TEX_W, MPMAX(wh, 1),
                  false, true, false, false, true);
        gc_ensure(gpu, &p->edge_tex, ef, EDGE_TEX_W, MPMAX(eh, 1),
                  false, true, false, false, true);
    }
#endif
}

// WP-E (E1.2/E1.3/E4a): one-time, config-time cold-start warm-up, run OFF the hot
// path before playback. (a) create the glyph atlas at full size, (b) preallocate
// the upload rings at their worst-frame max so staging-grow/overlay-buf-grow stay
// 0 during playback, (c) dispatch every OSD compute variant once against tiny
// dummy textures so no first-use pipeline compile happens during playback, and
// (d) touch the atlas/scratch/ring textures so the first real subtitle frame does
// no first-time allocation. libplacebo compiles a pass on first dispatch (keyed by
// the generated GLSL, not texture size), so a warm-up dispatch of each variant IS
// the compile guarantee -- there is no separate compile hook to count.
static void gc_warmup(struct priv *p)
{
    if (p->gc_warmed)
        return;
    p->gc_warmed = true;                 // once, even if parts below bail
    pl_gpu gpu = p->gpu;
    if (!gpu || !p->osd_dp)
        return;

    // (a) atlas + id table + segment epochs (created once, never rebuilt).
    gc_ensure_atlas(p);

    // WP-H1d: pre-create the per-item transient store for uncached/overflow
    // glyphs at the atlas dimensions (one frame's worth of giant-sign
    // coverage; at 8K a handful of screen-scale glyphs is ~100Mpx, within a
    // 16384x16384 store), so no giant-glyph frame pays the allocation
    // mid-playback (gc_trans_place still grows it, counted, if a frame
    // needs even more).
    if (p->glyph_atlas) {
        pl_fmt tr8 = p->osd_fmt[SUBBITMAP_LIBASS];
#if HAVE_ASS_OUTLINE_DEFERRED
        const bool tr_storable = true;
#else
        const bool tr_storable = false;
#endif
        if (tr8)
            gc_ensure(gpu, &p->trans_atlas, tr8, p->gatlas_w, p->gatlas_h,
                      tr_storable, true, false, false, true);
    }

    // (b) preallocate the async upload rings at their worst-frame maxima. The hot
    // path only (re)creates a ring when it's too small, so with these in place
    // staging-grow / overlay-buf-grow can no longer fire in normal playback (the
    // grow paths stay as counted safety fallbacks for pathological frames).
    // WP-H1c: the overlay ring + legacy pool textures are display-size-derived
    // and re-floored on every reconfig (ensure_overlay_pool below).
    for (int i = 0; i < 3; i++)
        pl_buf_recreate(gpu, &p->glyph_stage[i],
                        pl_buf_params(.size = GLYPH_STAGE_MAX_BYTES, .host_writable = true));

    // (c)+(d) WP-H5a: preallocate the raster/compose pools (run_acc/run_tmp/
    // run_cov_*, edge_tex/work_tex, per-sub result_tex) to the display-derived
    // worst case so the first -- and every subsequent -- real frame does no
    // first-time allocation and never grows them mid-playback (raster-pool-grow
    // stays 0). run_tmp must be r32f like run_acc (see the r_tmp rationale in
    // compose_glyph_runs), which gc_prealloc_pools honours.
    gc_prealloc_pools(p);
    pl_fmt r8  = p->osd_fmt[SUBBITMAP_LIBASS];
    pl_fmt r32 = p->osd_acc_fmt;
    const int D = 16;  // dummy dispatch dims (1 group)
    bool have_r8  = r8  && (r8->caps  & PL_FMT_CAP_STORABLE);
    bool have_r32 = r32 && (r32->caps & PL_FMT_CAP_STORABLE);
    int64_t *n = &p->cnt_shader_warmups;
    // Compilation is keyed by the generated GLSL (body + descriptor layout +
    // group size), independent of texture size, so a 16x16 dummy dispatch of each
    // variant compiles the exact pass the hot path later reuses.
    if (have_r32 && p->run_acc) {
        osd_clear(p, p->run_acc, D, D);                                  (*n)++;
        if (p->glyph_atlas) { osd_combine_part(p, p->glyph_atlas, p->run_acc, 0, 0, 0, 0, D, D); (*n)++; }
    }
    if (have_r8 && have_r32 && p->run_acc && p->run_cov_f)
        { osd_unop(p, p->run_acc, p->run_cov_f, D, D, "acc", false, "dst", osd_resolve_body); (*n)++; }
    if (have_r8 && p->run_cov_f && p->run_cov_b)
        { osd_unop(p, p->run_cov_f, p->run_cov_b, D, D, "fill", true, "bord", osd_fixoutline_body); (*n)++; }
    if (have_r8 && p->run_cov_f && p->run_tmp) {
        osd_blur_part(p, p->run_cov_f, p->run_tmp, 0, 0, D, D, 1.0f, osd_blur_body_h); (*n)++;
        osd_blur_part(p, p->run_tmp, p->run_cov_f, 0, 0, D, D, 1.0f, osd_blur_body_v); (*n)++;
        osd_copy(p, p->run_cov_f, p->run_tmp, 0, 0, D, D);              (*n)++;
    }
#if HAVE_ASS_OUTLINE_DEFERRED
    if (have_r8 && p->run_cov_f && p->run_tmp) {
        gc_be_pass3(p, p->run_cov_f, p->run_tmp, 0, 0, D, D);           (*n)++;
        gc_be_scale(p, p->run_cov_f, 0, 0, D, D, 0);                    (*n)++;
    }
    if (have_r8 && p->run_cov_f && p->run_cov_b)
        { gc_subshift(p, p->run_cov_f, p->run_cov_b, D, D, 0, 0);       (*n)++; }
    if (have_r8 && p->run_cov_f && p->glyph_atlas) {
        struct clipmask cm = { .w = D, .h = D };
        gc_clip_mult(p, p->run_cov_f, D, D, 0, 0, 0, 0, &cm);          (*n)++;
    }
    // Raster batch: bind zeroed edge/work pools so the single dummy tile has
    // gw==gh==0 and every invocation early-returns (no imageStore, no OOB).
    pl_fmt ef = pl_find_named_fmt(gpu, "rgba32f");
    if (ef && p->glyph_atlas &&
        gc_ensure(gpu, &p->edge_tex, ef, EDGE_TEX_W, 1, false, true, false, false, true) &&
        gc_ensure(gpu, &p->work_tex, ef, WORK_TEX_W, 1, false, true, false, false, true)) {
        static const float zeros[WORK_TEX_W * 4] = {0};
        pl_tex_upload(gpu, pl_tex_transfer_params(.tex = p->work_tex,
            .rc = { .x1 = WORK_TEX_W, .y1 = 1 },
            .row_pitch = (size_t) WORK_TEX_W * 4 * sizeof(float), .ptr = (void *) zeros));
        gc_raster_batch(p, 1, p->glyph_atlas);                          (*n)++;
    }
#endif
    // (d) touch the glyph atlas with a 1px host upload so it's fully resident.
    if (p->glyph_atlas) {
        static const uint8_t px = 0;
        pl_tex_upload(gpu, pl_tex_transfer_params(.tex = p->glyph_atlas,
            .rc = { .x1 = 1, .y1 = 1 }, .row_pitch = 1, .ptr = (void *) &px));
    }
    // (e) legacy overlay texture pool + overlay upload ring, floored to the
    // display size (WP-H1c; see ensure_overlay_pool).
    ensure_overlay_pool(p);
    // Force all compiles + uploads to finish now, before playback starts.
    pl_gpu_finish(gpu);
}

// A pending glyph-atlas upload (non-outline cache miss): the glyph's slot and
// its source pixels in the item's packed image.
struct gmiss { int ax, ay, w, h; const uint8_t *src; };

#if HAVE_ASS_OUTLINE_DEFERRED
// A pending outline-mode raster job: the glyph's atlas slot plus its
// tile-export blob's tiles (segments already appended to p->ebuf at sbase).
struct rjob { int ax, ay, w, h, sbase, nt; const float *tiles; };

// Upload the shared CPU segment pool (p->ebuf, filled by the resolve loops)
// to edge_tex. Done once per flush round, before the per-target raster
// dispatches (persistent atlas and, WP-H1d, the transient store share it).
static bool gc_flush_edges(struct priv *p, int ne)
{
    pl_gpu gpu = p->gpu;
    int seg_texels = ne * 2;                                // 2 rgba32f texels per segment
    int eh = (seg_texels + EDGE_TEX_W - 1) / EDGE_TEX_W;
    size_t eneed = (size_t) EDGE_TEX_W * eh * 4;
    if ((size_t) p->ebuf_cap < eneed) {
        p->ebuf_cap = eneed;
        p->ebuf = talloc_realloc(p, p->ebuf, float, p->ebuf_cap);
    }
    pl_fmt ef = pl_find_named_fmt(gpu, "rgba32f");
    if (!ef || !gc_ensure_pool(p, &p->edge_tex, ef, EDGE_TEX_W, eh, false, true,
                               false, false, true))
        return false;
    pl_tex_upload(gpu, pl_tex_transfer_params(.tex = p->edge_tex,
        .rc = { .x1 = EDGE_TEX_W, .y1 = eh },
        .row_pitch = (size_t) EDGE_TEX_W * 4 * sizeof(float), .ptr = p->ebuf));
    return true;
}

// Rasterize queued outline-mode raster jobs into dst: build the per-tile
// work-list (one 16x16 tile = one workgroup), then ONE batched dispatch.
// Shared by compose_glyph_runs (current frame, both atlases) and the WP-H1b
// idle pre-fill (persistent atlas only). gc_flush_edges must have run.
static void gc_flush_raster(struct priv *p, struct rjob *rjobs, int nrjobs,
                            pl_tex dst)
{
    pl_gpu gpu = p->gpu;
    stats_time_start(p->stats, "sub-raster");   // work-list build + dispatch
    // 4 texels per tile:
    // [ax,ay,tx,ty] [w,h,ng,_] group0[type,wind,segoff,cnt] group1[...]
    int ntiles = 0;
    for (int j = 0; j < nrjobs; j++)
        ntiles += rjobs[j].nt;
    int wh = (4 * ntiles + WORK_TEX_W - 1) / WORK_TEX_W;
    size_t wneed = (size_t) WORK_TEX_W * wh * 4;
    if ((size_t) p->wbuf_cap < wneed) {
        p->wbuf_cap = wneed;
        p->wbuf = talloc_realloc(p, p->wbuf, float, p->wbuf_cap);
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
    if (gc_ensure_pool(p, &p->work_tex, pl_find_named_fmt(gpu, "rgba32f"),
                       WORK_TEX_W, wh, false, true, false, false, true)) {
        pl_tex_upload(gpu, pl_tex_transfer_params(.tex = p->work_tex,
            .rc = { .x1 = WORK_TEX_W, .y1 = wh },
            .row_pitch = (size_t) WORK_TEX_W * 4 * sizeof(float), .ptr = p->wbuf));
        gc_raster_batch(p, ntiles, dst);
        p->cnt_raster_dispatches++;
        p->cnt_raster_tiles += ntiles;
    }
    stats_time_end(p->stats, "sub-raster");
}
#endif // HAVE_ASS_OUTLINE_DEFERRED

// Flush queued glyph upload misses into dst (the persistent atlas, or WP-H1d
// the per-item transient store): tight-repack into a CPU scratch, one async
// buffer write, then per-glyph async texture uploads (buffer-backed = no
// VO-thread stall). Shared by compose_glyph_runs and the WP-H1b idle pre-fill.
static void gc_flush_misses(struct priv *p, struct gmiss *miss, int nmiss,
                            size_t miss_bytes, ptrdiff_t pstride, pl_tex dst)
{
    pl_gpu gpu = p->gpu;
    stats_time_start(p->stats, "sub-upload");   // WP-A3: glyph atlas upload
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
        vo_alloc_bump(p, &p->cnt_staging_grow);   // WP-A3: glyph staging grow
    }
    size_t off = 0;
    size_t *offs = talloc_array(NULL, size_t, nmiss);
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
            pl_tex_upload(gpu, pl_tex_transfer_params(.tex = dst,
                .rc = { .x0 = g->ax, .y0 = g->ay, .x1 = g->ax + g->w, .y1 = g->ay + g->h },
                .row_pitch = g->w, .buf = *ring, .buf_offset = offs[m]));
        }
    } else {                                    // fallback: synchronous
        for (int m = 0; m < nmiss; m++) {
            struct gmiss *g = &miss[m];
            pl_tex_upload(gpu, pl_tex_transfer_params(.tex = dst,
                .rc = { .x0 = g->ax, .y0 = g->ay, .x1 = g->ax + g->w, .y1 = g->ay + g->h },
                .row_pitch = g->w, .ptr = p->gstage_cpu + offs[m]));
        }
    }
    talloc_free(offs);
    stats_time_end(p->stats, "sub-upload");
}

// Composite a SUBBITMAP_LIBASS_GLYPHS item: reproduce libass's per-run combine
// + blur + fix_outline on the GPU into entry->tex (a result atlas), then emit a
// single monochrome overlay whose parts reference each run's coverage region.
// SUBBITMAP_LIBASS_OUTLINES items take the same path, except cache-miss glyph
// coverage is rasterized on the GPU (gc_raster_batch) instead of uploaded.
//
// WP-E3: returns false iff the present guard expired at one of the internal
// checkpoints and the item was abandoned mid-build (the caller then presents
// the previous complete overlay state). Checkpoint placement is constrained
// by the persistent glyph cache: gcache_reserve() claims an atlas slot for a
// miss when the glyph is RESOLVED, but its pixels are only written by the
// batched raster/upload flush further down. Bailing between those two points
// would leave slots that cache-hit garbage forever after, so there is
// deliberately NO checkpoint inside the resolve->flush span; the safe
// boundaries are before the resolve loop, after the flush, and between
// per-region compose iterations (which only touch this build buffer's entry
// plus scratch never referenced by a committed overlay).
static bool compose_glyph_runs(struct priv *p, const struct sub_bitmaps *item,
                               struct osd_entry *entry, struct pl_frame *frame,
                               struct osd_state *state, enum pl_overlay_coords coords,
                               struct mp_image *src, double gs)
{
    // gs = render_w/display (<=1 when --sub-render-res-limit caps the render):
    // the glyphs are rendered at the capped resolution, so compose the runs in
    // that capped space and upscale the result_tex region to display in the
    // final overlay. gs == 1 (uncapped) leaves coords unchanged.
    pl_fmt r8 = p->osd_fmt[SUBBITMAP_LIBASS];
    // The gaussian's H->V intermediate (run_tmp) must keep more than 8 bits:
    // libass unpacks its 8-bit coverage to 14 bits (0x4000) and holds that
    // precision across BOTH separable passes, packing back to 8 bits only once,
    // after the V pass. An r8 intermediate would round the H-pass result to 8
    // bits before the V pass -- a second 8-bit rounding libass never does --
    // which leaves a ~1-LSB error on every blurred edge that compounds across
    // overlapping layers (kobayashi maxdiff 2000). r32f is the storable
    // high-precision format already used for run_acc; carrying the H result at
    // that precision makes the pair of passes match libass's single final pack.
    pl_fmt r_tmp = p->osd_acc_fmt;
    if (!r8 || !r_tmp || !(r_tmp->caps & PL_FMT_CAP_STORABLE) ||
        !(r8->caps & PL_FMT_CAP_STORABLE))
        return true;

    // Persistent glyph cache: ensure the atlas + id table exist (created once,
    // never rebuilt), then upload/rasterize only the cache-miss glyphs. WP-E:
    // no watermark flush -- epoch-segmented eviction (gc_place) reclaims space.
    if (!gc_ensure_atlas(p))
        return true;
    // WP-E3 checkpoint: last safe point before the glyph resolve claims atlas
    // slots (see the function comment); nothing is mutated yet.
    if (sub_guard_expired(p))
        return false;
    // WP-E: each compose item is one eviction "pass". A segment claimed earlier
    // in this pass is never recycled mid-pass, so a glyph resolved into cpos[]
    // here stays valid through this item's raster+composite below (WP-H1d:
    // extra glyphs overflow into the per-item transient store instead of
    // overwriting already-placed ones -- or being skipped).
    p->gc_pass++;
    p->gc_pass_claims = 0;
    p->gc_pass_wraps = 0;
    // WP-H1d: the transient store is per-item; its content from the previous
    // item was consumed by that item's composes.
    p->tr_x = p->tr_y = p->tr_rowh = 0;

    void *tmp = talloc_new(NULL);

    // Resolve every deferred glyph to its atlas position, collecting cache
    // misses to upload in one async batch (a synchronous per-glyph .ptr upload
    // stalls the VO thread on d3d11; see the staging-buffer note below).
    // Outline mode (SUBBITMAP_LIBASS_OUTLINES) has no packed atlas: a cache
    // miss is instead RASTERIZED into its slot from the part's tile-export
    // blob, batched into one dispatch below (rasterize-on-miss-only: hits use
    // the cached coverage untouched).
    bool is_outline = item->format == SUBBITMAP_LIBASS_OUTLINES;
    struct gpos *cpos = talloc_array(tmp, struct gpos, item->num_parts);
    // Pending uploads/raster jobs, split by destination: the persistent atlas
    // (cache misses) vs the per-item transient store (WP-H1d: uncached giants
    // + overflow -- never skipped).
    struct gmiss *miss = NULL, *tmiss = NULL;
    int nmiss = 0, ntmiss = 0;
    size_t miss_bytes = 0, tmiss_bytes = 0;
#if HAVE_ASS_OUTLINE_DEFERRED
    // outline raster jobs: per missed glyph, its slot + the blob's tiles
    struct rjob *rjobs = NULL, *trjobs = NULL;
    int nrjobs = 0, ntrjobs = 0, ne = 0;     // ne = pooled segments (2 texels each)
    // Vector-\clip masks found in this item: rasterized like glyphs, then used
    // to multiply each clipped run's coverage (gc_outline_features).
    struct clipmask *clips = NULL;
    int nclips = 0;
#endif
    ptrdiff_t pstride = is_outline ? 0 : item->packed->stride[0];
    // WP-E: resolve each glyph via the epoch-evicting cache (gc_place). A miss
    // recycles at most one ring segment (bounded); nothing full-flushes.
    // WP-H1d: a glyph the cache does not admit (above the size cap) or cannot
    // place (this item overcommitting the whole atlas; counted in
    // gcache_reserve) goes to the per-item transient store and is re-
    // rasterized/uploaded THIS frame -- never dropped. The old skip left
    // cpos at {0,0}, so affected runs composed whatever lived at the atlas
    // origin: wrong-glyph garbage or missing text, varying with eviction
    // history (the "signs without their text" 8K artifact).
    for (int i = 0; i < item->num_parts; i++) {
        const struct sub_bitmap *b = &item->parts[i];
        cpos[i] = (struct gpos){0, 0, 0};
        if (b->libass.glyph_id == 0)
            continue;
        bool up = false;
        bool placed = gc_cacheable(p, b->w, b->h) &&
                      gcache_reserve(p, b->libass.glyph_id, b->w, b->h,
                                     &cpos[i], &up);
        if (!placed) {
            if (!gc_trans_place(p, b->w, b->h, &cpos[i])) {
                // Theoretical bound (> max-texture transient demand): make
                // the glyph invisible rather than compose garbage, and let
                // the acceptance gate catch it.
                cpos[i].t = -1;
                p->cnt_gcache_overcommit++;
                continue;
            }
            if (!gc_cacheable(p, b->w, b->h))
                p->cnt_glyph_uncached++;  // policy (giant glyph); allocator
                                          // overflow was counted as overcommit
            up = true;                    // transient content is per-item
        }
#if HAVE_ASS_OUTLINE_DEFERRED
        if (b->libass.run_flags & RUN_FLAG_CLIP_MASK) {
            // Record the mask (rasterized below like a glyph on a miss). Its
            // screen origin lives in the same (capped) render space as the
            // region coords, so scale the display-space part position by gs.
            MP_TARRAY_APPEND(tmp, clips, nclips, ((struct clipmask){
                b->libass.run_id, cpos[i].ax, cpos[i].ay,
                (int) lrint(b->x * gs), (int) lrint(b->y * gs), b->w, b->h,
                !!(b->libass.run_flags & RUN_FLAG_CLIP_INVERSE), cpos[i].t }));
        }
        if (up && is_outline) {
            // libass tile-export blob: [n_tiles, n_segs, tiles(11f), segs(8f)].
            // Append this glyph's segments to the shared seg pool (2 rgba32f
            // texels each) and record the glyph's tiles for the work-list below.
            const int32_t *blob = b->libass.outline;
            if (!blob || b->libass.n_outline < 2)
                continue;
            int nt = blob[0], ns = blob[1];
            const float *gtiles = (const float *)(blob + 2);
            const float *gsegs  = (const float *)(blob + 2 + (size_t) nt * TILE_EXPORT_W);
            if (p->ebuf_cap < (ne + ns) * SEG_EXPORT_W) {
                p->ebuf_cap = MPMAX((ne + ns) * SEG_EXPORT_W * 2, 8192);
                p->ebuf = talloc_realloc(p, p->ebuf, float, p->ebuf_cap);
            }
            memcpy(p->ebuf + (size_t) ne * SEG_EXPORT_W, gsegs,
                   (size_t) ns * SEG_EXPORT_W * sizeof(float));
            struct rjob rj = { cpos[i].ax, cpos[i].ay, b->w, b->h, ne, nt, gtiles };
            if (cpos[i].t) {
                MP_TARRAY_APPEND(tmp, trjobs, ntrjobs, rj);
            } else {
                MP_TARRAY_APPEND(tmp, rjobs, nrjobs, rj);
            }
            ne += ns;
            continue;
        }
#endif // HAVE_ASS_OUTLINE_DEFERRED
        if (up) {
            const uint8_t *gsrc = (const uint8_t *) item->packed->planes[0]
                                + (ptrdiff_t) b->src_y * pstride + b->src_x;
            struct gmiss gm = { cpos[i].ax, cpos[i].ay, b->w, b->h, gsrc };
            if (cpos[i].t) {
                MP_TARRAY_APPEND(tmp, tmiss, ntmiss, gm);
                tmiss_bytes += (size_t) b->w * b->h;
            } else {
                MP_TARRAY_APPEND(tmp, miss, nmiss, gm);
                miss_bytes += (size_t) b->w * b->h;
            }
        }
    }

#if HAVE_ASS_OUTLINE_DEFERRED
    // Rasterize the outline-mode misses: the shared segment pool once, then
    // one batched dispatch per destination.
    if (ne && gc_flush_edges(p, ne)) {
        if (nrjobs)
            gc_flush_raster(p, rjobs, nrjobs, p->glyph_atlas);
        if (ntrjobs)
            gc_flush_raster(p, trjobs, ntrjobs, p->trans_atlas);
    }
#endif // HAVE_ASS_OUTLINE_DEFERRED

    // Flush the upload misses (async, buffer-backed).
    if (nmiss)
        gc_flush_misses(p, miss, nmiss, miss_bytes, pstride, p->glyph_atlas);
    if (ntmiss)
        gc_flush_misses(p, tmiss, ntmiss, tmiss_bytes, pstride, p->trans_atlas);

    // WP-E3 checkpoint: the miss flush is recorded (every claimed atlas slot
    // has its raster/upload in flight), so the glyph cache is consistent again
    // and bailing is safe from here on.
    if (sub_guard_expired(p)) {
        talloc_free(tmp);
        return false;
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
            regs[ri] = (struct gc_region){ .x0 = lrint(b->x * gs), .y0 = lrint(b->y * gs),
                .x1 = lrint((b->x + b->dw) * gs), .y1 = lrint((b->y + b->dh) * gs),
                .run_flags = b->libass.run_flags, .single_layer = 0xff,
                .clip_id = b->libass.clip_id,
                .rcx0 = b->libass.clip_rx0, .rcy0 = b->libass.clip_ry0,
                .rcx1 = b->libass.clip_rx1, .rcy1 = b->libass.clip_ry1,
                .shift_x = b->libass.shift_x64, .shift_y = b->libass.shift_y64 };
        }
        r = &regs[ri];
        r->x0 = MPMIN(r->x0, (int) lrint(b->x * gs));
        r->y0 = MPMIN(r->y0, (int) lrint(b->y * gs));
        r->x1 = MPMAX(r->x1, (int) lrint((b->x + b->dw) * gs));
        r->y1 = MPMAX(r->y1, (int) lrint((b->y + b->dh) * gs));
        if (b->libass.layer == 1) {
            r->bord_color = b->libass.color;
            r->blur_b = b->libass.blur_x;
            r->be_b = b->libass.be;
            MP_TARRAY_APPEND(tmp, r->bord, r->nbord, i);
        } else {
            r->fill_color = b->libass.color;
            r->blur_f = b->libass.blur_x;
            r->be_f = b->libass.be;
            r->fill_color2 = b->libass.color2;
            r->wipe_x = b->libass.wipe_x;
            // KF_WIPE lives on the fill part; the region's run_flags came from
            // whichever part appeared first (the border), so OR it in here.
            r->run_flags |= b->libass.run_flags & RUN_FLAG_KF_WIPE;
            MP_TARRAY_APPEND(tmp, r->fill, r->nfill, i);
        }
    }

    if (!nregs) { talloc_free(tmp); return true; }   // no deferred runs in this item

    const int AW = 4096;
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
        // the blur halo padding (libass's exact expand amount), plus one px
        // per \be iteration (each box pass grows the support by one).
        r->margin = blur_expand_pad(MPMAX(r->blur_f, r->blur_b)) +
                    MPMAX(r->be_f, r->be_b);
        int rw = r->x1 - r->x0 + 2 * r->margin, rh = r->y1 - r->y0 + 2 * r->margin;
        if (r->nfill) ALLOC_REGION(r->fill_ax, r->fill_ay, rw, rh);
        if (r->nbord) ALLOC_REGION(r->bord_ax, r->bord_ay, rw, rh);
    }
    int atlas_w = MPMAX(max_w, 1), atlas_h = MPMAX(shelf_y + shelf_h, 1);

    if (!gc_ensure_pool(p, &entry->result_tex, r8, atlas_w, atlas_h, true, true, false, false, false) ||
        !gc_ensure_pool(p, &p->run_acc, p->osd_acc_fmt, run_acc_w, run_acc_h, true, false, false, false, false) ||
        !gc_ensure_pool(p, &p->run_tmp, r_tmp, run_acc_w, run_acc_h, true, true, false, false, false) ||
        !gc_ensure_pool(p, &p->run_cov_f, r8, run_acc_w, run_acc_h, true, true, false, false, false) ||
        !gc_ensure_pool(p, &p->run_cov_b, r8, run_acc_w, run_acc_h, true, true, false, false, false)) {
        talloc_free(tmp);
        return true;
    }

    for (int i = 0; i < nregs; i++) {
        // WP-E3 checkpoint: between regions. Each iteration only records
        // dispatches into this build buffer's result_tex and the shared
        // scratch (run_acc/run_cov_*), neither of which the previous
        // complete overlay state references.
        if (sub_guard_expired(p)) {
            talloc_free(tmp);
            return false;
        }
        struct gc_region *r = &regs[i];
        int bw = r->x1 - r->x0 + 2 * r->margin, bh = r->y1 - r->y0 + 2 * r->margin;
        // Reproduce the CPU pipeline order (ass_composite_construct):
        // ass_synth_blur (gaussian, then \be) runs on each layer BEFORE
        // ass_fix_outline subtracts the crisp fill from the (blurred) border.
        // The bordered fill is neither blurred nor be'd (libass zeroes its
        // sigma/be; blur_bm gating), so blurring it here is a no-op then.
        // \be is applied by mpv only in outline mode -- in GLYPHS mode libass
        // box-blurs the coverage on the CPU before deferring the gaussian.
        if (r->nfill) {
            gc_build_cov(p, item, r, r->fill, r->nfill, p->run_cov_f, bw, bh, cpos, gs);
            gc_blur(p, p->run_cov_f, bw, bh, r->blur_f);  // σ may be 0 (bordered fill)
#if HAVE_ASS_OUTLINE_DEFERRED
            if (is_outline && r->be_f > 0)
                gc_be_blur(p, p->run_cov_f, p->run_tmp, 0, 0, bw, bh, r->be_f);
            // Shadow runs: the whole-pixel part of libass's shadow offset is
            // in the parts' positions; this sub-pixel remainder is applied to
            // the final coverage AFTER the gaussian and \be, matching the CPU
            // order (ass_composite_construct blurs bm_s's source first, then
            // ass_shift_bitmap). run_cov_b is free scratch here: shadow runs
            // are fill-only, and a bordered region rebuilds it just below.
            if (is_outline && (r->run_flags & RUN_FLAG_SHADOW) &&
                (r->shift_x || r->shift_y)) {
                gc_subshift(p, p->run_cov_f, p->run_cov_b, bw, bh,
                            r->shift_x, r->shift_y);
                osd_copy(p, p->run_cov_b, p->run_cov_f, 0, 0, bw, bh);
            }
#endif
            if (r->nbord) {
                gc_build_cov(p, item, r, r->bord, r->nbord, p->run_cov_b, bw, bh, cpos, gs);
                gc_blur(p, p->run_cov_b, bw, bh, r->blur_b);
#if HAVE_ASS_OUTLINE_DEFERRED
                if (is_outline && r->be_b > 0)
                    gc_be_blur(p, p->run_cov_b, p->run_tmp, 0, 0, bw, bh, r->be_b);
#endif
                if (r->run_flags & 1)   // bit 0 = apply fix_outline (see libass)
                    osd_unop(p, p->run_cov_f, p->run_cov_b, bw, bh,
                             "fill", true, "bord", osd_fixoutline_body);
#if HAVE_ASS_OUTLINE_DEFERRED
                gc_apply_clip(p, p->run_cov_b, bw, bh, r, clips, nclips);
#endif
                osd_copy(p, p->run_cov_b, entry->result_tex, r->bord_ax, r->bord_ay, bw, bh);
            }
#if HAVE_ASS_OUTLINE_DEFERRED
            gc_apply_clip(p, p->run_cov_f, bw, bh, r, clips, nclips);
#endif
            osd_copy(p, p->run_cov_f, entry->result_tex, r->fill_ax, r->fill_ay, bw, bh);
        } else if (r->nbord) {
            gc_build_cov(p, item, r, r->bord, r->nbord, p->run_cov_b, bw, bh, cpos, gs);
            gc_blur(p, p->run_cov_b, bw, bh, r->blur_b);
#if HAVE_ASS_OUTLINE_DEFERRED
            if (is_outline && r->be_b > 0)
                gc_be_blur(p, p->run_cov_b, p->run_tmp, 0, 0, bw, bh, r->be_b);
            gc_apply_clip(p, p->run_cov_b, bw, bh, r, clips, nclips);
#endif
            osd_copy(p, p->run_cov_b, entry->result_tex, r->bord_ax, r->bord_ay, bw, bh);
        }
    }

    // Emit the overlay in libass's EXACT z-order: the ASS_Image list order,
    // which item->parts preserves. Within one event libass emits every run's
    // border image before any run's fill image (and shadow/background runs
    // before those); across events the list is back-to-front. Region-major
    // emission (border then fill per region) broke that for multi-run events
    // whose runs overlap -- adjacent \bord3 karaoke syllables, overlapping
    // 3D-tilted glyph runs -- a later run's border was painted OVER an
    // earlier run's fill, where libass paints all borders first. And a global
    // border-then-fill pass across all runs mislayered overlapping signs from
    // different events. Ordering each region-layer by its FIRST part index
    // reproduces the image-list order exactly, covering both cases.
    entry->num_run_parts = 0;
    struct gc_emit { int key, reg, layer; };   // layer: 0 border, 1 fill
    struct gc_emit *ems = talloc_array(tmp, struct gc_emit, 2 * (size_t) nregs);
    int nems = 0;
    for (int i = 0; i < nregs; i++) {
        if (regs[i].nbord)
            ems[nems++] = (struct gc_emit){ regs[i].bord[0], i, 0 };
        if (regs[i].nfill)
            ems[nems++] = (struct gc_emit){ regs[i].fill[0], i, 1 };
    }
    // insertion sort by key (part indices are distinct; nems is small)
    for (int i = 1; i < nems; i++) {
        struct gc_emit e = ems[i];
        int j = i;
        for (; j > 0 && ems[j - 1].key > e.key; j--)
            ems[j] = ems[j - 1];
        ems[j] = e;
    }
    for (int ei = 0; ei < nems; ei++) {
        struct gc_region *r = &regs[ems[ei].reg];
        int layer = ems[ei].layer;
        bool is_shadow = r->run_flags & RUN_FLAG_SHADOW;
        int dx0 = r->x0 - r->margin, dy0 = r->y0 - r->margin;
        int dx1 = r->x1 + r->margin, dy1 = r->y1 + r->margin;
        // Rectangular \clip on the dst (no shader, a plain rect crop), in the
        // same (capped) render space as the region coords: normal \clip keeps
        // one visible rect (the intersection); inverse \iclip subtracts the
        // excluded rect, leaving up to 4 visible strips. Non-outline items
        // carry no rect: keep the whole dst.
        int vr[4][4]; int nvr = 0;
        if (!is_outline) {
            vr[0][0]=dx0; vr[0][1]=dy0; vr[0][2]=dx1; vr[0][3]=dy1; nvr = 1;
        } else if (r->run_flags & RUN_FLAG_RECT_INVERSE) {
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
        int ax, ay; uint32_t c;
        if (layer == 0) { ax = r->bord_ax; ay = r->bord_ay; c = r->bord_color; }
        else            { ax = r->fill_ax; ay = r->fill_ay; c = r->fill_color; }
        for (int v = 0; v < nvr; v++) {
            int cx0 = vr[v][0], cy0 = vr[v][1], cx1 = vr[v][2], cy1 = vr[v][3];
            // \kf karaoke: split the fill at wipe_x into sung (fill_color,
            // left) and unsung (fill_color2, right). Else one segment.
            int sx0[2] = { cx0, 0 }, sx1[2] = { cx1, 0 };
            uint32_t sc[2] = { c, 0 }; int nseg = 1;
            if (is_outline && layer == 1 && !is_shadow &&
                (r->run_flags & RUN_FLAG_KF_WIPE)) {
                int w = MPMAX(cx0, MPMIN(cx1, r->wipe_x));
                sx0[0] = cx0; sx1[0] = w;   sc[0] = r->fill_color;
                sx0[1] = w;   sx1[1] = cx1; sc[1] = r->fill_color2;
                nseg = 2;
            }
            for (int s = 0; s < nseg; s++) {
                if (sx0[s] >= sx1[s]) continue;
                uint32_t sg = sc[s];
                // result_tex is composited in capped space; upscale the
                // region back to display coords (gs == 1 when uncapped ->
                // identity). dst is a float rect, so libplacebo
                // bilinear-upscales the sample.
                struct pl_overlay_part part = {
                    .src = { ax + (sx0[s] - dx0), ay + (cy0 - dy0),
                             ax + (sx1[s] - dx0), ay + (cy1 - dy0) },
                    .dst = { sx0[s] / gs, cy0 / gs, sx1[s] / gs, cy1 / gs },
                    .color = { (sg >> 24) / 255.0f, ((sg >> 16) & 0xFF) / 255.0f,
                               ((sg >> 8) & 0xFF) / 255.0f, (255 - (sg & 0xFF)) / 255.0f },
                };
                MP_TARRAY_APPEND(p, entry->run_parts, entry->num_run_parts, part);
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
    return true;
}

// WP-H1b: warm the glyph atlas for ONE upcoming render-ahead frame (a served
// ring copy the VO hasn't drawn yet): resolve its deferred glyphs against the
// cache and upload/rasterize the misses, exactly like the resolve+flush phase
// of compose_glyph_runs but with nothing composed. Runs in no-recycle mode so
// it can never evict the current frame's live glyphs (if the atlas is that
// tight it refuses and the entry is treated as done -- the real frame handles
// its misses with normal eviction). Budget: stop RESOLVING once deadline_ns
// passes, but always flush what was reserved (a reserved slot whose pixels
// are never written would cache-hit garbage forever -- same invariant as the
// resolve->flush span in compose_glyph_runs). Returns true when the item
// needs no further pre-fill passes; false = budget ran out mid-item (the
// caller re-peeks it on a later frame; resolved glyphs then hit).
static bool gc_prefill_item(struct priv *p, const struct sub_bitmaps *item,
                            int64_t deadline_ns)
{
    bool is_outline = item->format == SUBBITMAP_LIBASS_OUTLINES;
#if !HAVE_ASS_OUTLINE_DEFERRED
    if (is_outline)
        return true;
#endif
    ptrdiff_t pstride = 0;
    if (!is_outline) {
        if (!item->packed)
            return true;
        pstride = item->packed->stride[0];
    }
    if (!gc_ensure_atlas(p))
        return true;

    void *tmp = talloc_new(NULL);
    // Same eviction-pass semantics as a real compose item (a segment claimed
    // by this pass is never recycled mid-pass), plus no-recycle (see above).
    p->gc_pass++;
    p->gc_pass_claims = 0;
    p->gc_pass_wraps = 0;
    p->gc_no_recycle = true;
    p->gc_refused = false;
    struct gmiss *miss = NULL;
    int nmiss = 0;
    size_t miss_bytes = 0;
#if HAVE_ASS_OUTLINE_DEFERRED
    struct rjob *rjobs = NULL;
    int nrjobs = 0, ne = 0;
#endif
    bool complete = true;
    for (int i = 0; i < item->num_parts; i++) {
        if ((i & 15) == 0 && mp_time_ns() > deadline_ns) {
            complete = false;         // budget: stop resolving, flush, resume later
            break;
        }
        const struct sub_bitmap *b = &item->parts[i];
        if (b->libass.glyph_id == 0)
            continue;
        if (!gc_cacheable(p, b->w, b->h))
            continue;                 // WP-H1d: giant glyphs are never cached --
                                      // the frame rasterizes them transiently
        bool up;
        struct gpos pos;
        if (!gcache_reserve(p, b->libass.glyph_id, b->w, b->h, &pos, &up)) {
            if (p->gc_refused)
                break;                // atlas tight: give up on this entry
            continue;                 // table full: nothing to pre-fill
        }
        if (!up)
            continue;                 // already resident
#if HAVE_ASS_OUTLINE_DEFERRED
        if (is_outline) {
            // Queue the raster job (same blob unpacking as compose_glyph_runs).
            const int32_t *blob = b->libass.outline;
            if (!blob || b->libass.n_outline < 2)
                continue;
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
                ((struct rjob){ pos.ax, pos.ay, b->w, b->h, ne, nt, gtiles }));
            ne += ns;
            continue;
        }
#endif
        const uint8_t *gsrc = (const uint8_t *) item->packed->planes[0]
                            + (ptrdiff_t) b->src_y * pstride + b->src_x;
        MP_TARRAY_APPEND(tmp, miss, nmiss,
            ((struct gmiss){ pos.ax, pos.ay, b->w, b->h, gsrc }));
        miss_bytes += (size_t) b->w * b->h;
    }
    p->gc_no_recycle = false;
#if HAVE_ASS_OUTLINE_DEFERRED
    if (ne && gc_flush_edges(p, ne)) {
        gc_flush_raster(p, rjobs, nrjobs, p->glyph_atlas);
        p->cnt_prefill_glyphs += nrjobs;
    }
#endif
    if (nmiss) {
        gc_flush_misses(p, miss, nmiss, miss_bytes, pstride, p->glyph_atlas);
        p->cnt_prefill_glyphs += nmiss;
    }
    if (p->gc_refused)
        complete = true;              // never spin on a refused entry
    talloc_free(tmp);
    return complete;
}

// WP-H1b idle GPU pre-fill: on frames whose sub state didn't change (empty or
// static subs -- exactly the frames with headroom), warm the glyph atlas for
// upcoming frames already banked in the render-ahead ring, so the FIRST frame
// of a dense event doesn't pay the whole atlas fill (cold start, post-seek).
// The work is budgeted (--sub-prefill-budget-ms of wall clock per frame) and
// counts toward the frame's WP-E3 deadline: it only runs when at least twice
// its budget remains before the present-guard deadline, so it can never be
// what pushes a frame over. Runs after the frame's own overlay build + render
// recording; the extra dispatches ride the same submission.
static void sub_prefill_idle(struct vo *vo)
{
    struct priv *p = vo->priv;
    int budget_ms = p->next_opts->sub_prefill_budget_ms;
    if (budget_ms <= 0 || !p->gc_warmed || !p->glyph_atlas)
        return;
    if (!p->sub_state_cheap || p->guard_fired)
        return;
    int64_t now = mp_time_ns();
    int64_t budget_ns = budget_ms * INT64_C(1000000);
    if (p->guard_deadline_ns > 0 &&
        p->guard_t0 + p->guard_deadline_ns - now < 2 * budget_ns)
        return;
    double pts = 0;
    struct sub_bitmaps *item = osd_sub_peek_ahead(vo->osd, &pts);
    if (!item)
        return;
    bool done = true;
    if ((item->format == SUBBITMAP_LIBASS_GLYPHS ||
         item->format == SUBBITMAP_LIBASS_OUTLINES) && item->num_parts) {
        stats_time_start(p->stats, "sub-prefill");
        done = gc_prefill_item(p, item, now + budget_ns);
        stats_time_end(p->stats, "sub-prefill");
    }
    talloc_free(item);
    if (done)
        osd_sub_peek_ahead_done(vo->osd, pts);
}
#endif // HAVE_ASS_COMPOSITE_DEFERRED

// Build the overlay set for `frame` into one of g's ping-pong buffers.
// `present` marks the two draw_frame (presentation) callers: only those
// commit the build as the new "last complete" state and are subject to the
// WP-E3 present guard; video_screenshot passes false (always builds fully,
// never commits, so a screenshot at foreign geometry can't pollute the
// presentation snapshot). Returns false iff the guard bailed the build (the
// caller must then not treat the overlay state as freshly built, e.g. the
// blend-subs path leaves fp->osd_sync behind so the next frame retries).
static bool update_overlays(struct vo *vo, struct mp_osd_res res,
                            int flags, enum pl_overlay_coords coords,
                            struct osd_guard *g, struct pl_frame *frame,
                            struct mp_image *src, int stereo_mode, bool present)
{
    struct priv *p = vo->priv;
    double pts = src ? src->pts : 0;
    int div[2];
    mp_get_3d_side_by_side(stereo_mode, div);
    res.w /= div[0];
    res.h /= div[1];

    // WP-E3 present guard: build into the buffer NOT holding the last
    // complete overlay state, so that state stays untouched and presentable
    // until this build commits (see struct osd_guard). A build that follows
    // a bail runs deadline-free (must_complete) to guarantee progress.
    uint64_t sub_epoch = osd_sub_track_epoch(vo->osd);
    bool guard_on = present && p->guard_deadline_ns > 0 && !g->must_complete;
    p->guard_abs = guard_on ? p->guard_t0 + p->guard_deadline_ns : 0;
    if (present)
        g->must_complete = false;
    int build = g->good >= 0 ? 1 - g->good : 0;
    struct osd_state *state = &g->states[build];
    // WP-A3 slow-frame phase timers: only take wall-clock samples when the
    // MSGL_V [slowframe] path can fire. The dump-stats start/end phase events
    // (stats_time_*) are emitted unconditionally and self-gate on MSGL_STATS.
    bool want_t = mp_msg_test(vo->log, MSGL_V);
    p->dbg_subrender_ns = p->dbg_capcomp_ns = p->dbg_blur_ns = 0;
    // Advertise the deferred-composite/outline formats too; sd_ass only emits
    // them when --sub-gpu-composite / --sub-gpu-raster are set, otherwise it
    // falls back to plain LIBASS.
    static const bool gpu_sub_formats[SUBBITMAP_COUNT] = {
        [SUBBITMAP_LIBASS] = true, [SUBBITMAP_BGRA] = true,
        [SUBBITMAP_LIBASS_GLYPHS] = true,
#if HAVE_ASS_OUTLINE_DEFERRED
        [SUBBITMAP_LIBASS_OUTLINES] = true,
#endif
    };
    stats_time_start(p->stats, "sub-render");   // WP-A3: libass render + fetch
    int64_t sr0 = want_t ? mp_time_ns() : 0;
    struct sub_bitmap_list *subs = osd_render(vo->osd, res, pts, flags, gpu_sub_formats);
    if (want_t)
        p->dbg_subrender_ns = mp_time_ns() - sr0;
    stats_time_end(p->stats, "sub-render");

    // WP-H1b: an unchanged osd_render change_id means this frame's sub/OSD
    // state is static or empty -- the cheap frames the idle glyph pre-fill is
    // allowed to use. Main presentation consumer only (the blend-subs path
    // has its own per-frame sync and the screenshot path must not vote).
    if (present && g == &p->osd_guard) {
        p->sub_state_cheap = subs->change_id == p->last_sub_change_id;
        p->last_sub_change_id = subs->change_id;
    }

    frame->overlays = state->overlays;
    frame->num_overlays = 0;

    // WP-E3 checkpoint: after the fetch, before anything is mutated (covers a
    // stalled osd_render, e.g. a render-ahead miss-wait blown far past its
    // own bound).
    if (sub_guard_expired(p))
        goto bail;
    // WP-E3 debug (--sub-debug-stall-ms): inject an artificial stall after
    // the first checkpoint so the guard is deterministically testable. Main
    // presentation call only, once per frame.
    if (present && g == &p->osd_guard && p->next_opts->sub_debug_stall_ms > 0) {
        mp_sleep_ns(p->next_opts->sub_debug_stall_ms * INT64_C(1000000));
        if (sub_guard_expired(p))   // the checkpoint the injected stall hits
            goto bail;
    }

    for (int n = 0; n < subs->num_items; n++) {
        // WP-E3 checkpoint: between OSD items; the previous item's overlay is
        // fully recorded, this item's entry is untouched.
        if (sub_guard_expired(p))
            goto bail;
        const struct sub_bitmaps *item = subs->items[n];
#if HAVE_ASS_OUTLINE_DEFERRED
        // Outline mode has no packed atlas (the GPU rasterizes from the blobs).
        if (!item->num_parts ||
            (!item->packed && item->format != SUBBITMAP_LIBASS_OUTLINES))
            continue;
#else
        if (!item->num_parts || !item->packed)
            continue;
#endif
        struct osd_entry *entry = &state->entries[item->render_index];
#if HAVE_ASS_OUTLINE_DEFERRED
        if (item->format == SUBBITMAP_LIBASS_OUTLINES) {
            // Coverage is rasterized at the (capped) render_w; compose in that
            // space and upscale to display. gs == 1 when uncapped.
            double gs = item->render_w > 0 && subs->w > 0
                      ? (double) item->render_w / subs->w : 1.0;
            stats_time_start(p->stats, "sub-composite");
            bool done = compose_glyph_runs(p, item, entry, frame, state, coords, src, gs);
            stats_time_end(p->stats, "sub-composite");
            if (!done)
                goto bail;
            continue;   // no legacy fallback path (it would need a packed atlas)
        }
#endif // HAVE_ASS_OUTLINE_DEFERRED
#if HAVE_ASS_COMPOSITE_DEFERRED
        if (item->format == SUBBITMAP_LIBASS_GLYPHS) {
            // Glyph coverage is rendered at the (capped) render_w; compose the
            // runs in that space and upscale to display. gs == 1 when uncapped.
            double gs = item->render_w > 0 && subs->w > 0
                      ? (double) item->render_w / subs->w : 1.0;
            stats_time_start(p->stats, "sub-composite");   // WP-A3: GPU per-glyph composite
            bool done = compose_glyph_runs(p, item, entry, frame, state, coords, src, gs);
            stats_time_end(p->stats, "sub-composite");
            if (!done)
                goto bail;
            // Already-combined fallback parts (shadow/karaoke runs, glyph_id 0)
            // go through the legacy path below; skip it if there are none.
            bool has_fallback = false;
            for (int i = 0; i < item->num_parts; i++)
                if (item->parts[i].libass.glyph_id == 0) { has_fallback = true; break; }
            if (!has_fallback)
                continue;
        }
#endif // HAVE_ASS_COMPOSITE_DEFERRED
        pl_fmt tex_fmt = p->osd_fmt[item->format];
        if (!entry->tex)
            MP_TARRAY_POP(p->sub_tex, p->num_sub_tex, &entry->tex);
        // Round the OSD texture up and grow it monotonically so it isn't
        // reallocated every frame as the atlas grows through a dense scene
        // (each realloc stalls the display thread). WP-E: also floor both dims
        // so the first allocation already covers the whole scene's overlays --
        // otherwise the tex creeps up to the scene max over several frames, and
        // which frame first hits the max (hence the grow) is decided by
        // --untimed frame dropping, making tex-realloc-after-first spuriously
        // nonzero. WP-H1c: the floor is derived from the display size (the old
        // fixed 2048 was too small for 8K packed atlases -- the first OSD
        // overlay recreated the texture mid-playback) and matches the pool
        // textures pre-created at warm-up/reconfig, so popping one makes the
        // recreate below a no-op. A truly oversized overlay still grows once
        // via the counted fallback.
        int fl_w, fl_h;
        overlay_tex_floor(p, &fl_w, &fl_h);
        int want_w = MPMAX((item->packed_w + 255) & ~255, fl_w);
        int want_h = MPMAX((item->packed_h + 255) & ~255, fl_h);
        int prev_w = entry->tex ? entry->tex->params.w : -1;
        int prev_h = entry->tex ? entry->tex->params.h : -1;
        int new_w = MPMAX(want_w, entry->tex ? entry->tex->params.w : 0);
        int new_h = MPMAX(want_h, entry->tex ? entry->tex->params.h : 0);
        bool ok = pl_tex_recreate(p->gpu, &entry->tex, &(struct pl_tex_params) {
            .format = tex_fmt,
            .w = new_w,
            .h = new_h,
            .host_writable = true,
            .sampleable = true,
        });
        if (!ok) {
            MP_ERR(vo, "Failed recreating OSD texture!\n");
            break;
        }
        // WP-A3: count only real (re)allocations -- the tex is grown monotonically
        // so this fires on first alloc and on each grow, not every frame.
        if (new_w != prev_w || new_h != prev_h)
            vo_alloc_bump(p, &p->cnt_tex_realloc);
        stats_time_start(p->stats, "sub-upload");   // WP-A3: overlay atlas upload
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
        // Reuse the staging buffer whenever it's already big enough; only
        // (re)allocate on growth, rounded up, so it stops being reallocated as
        // the atlas grows frame-to-frame through a dense scene (that realloc was
        // the ~187ms VO-thread stall).
        bool buf_ok = (*ring) && (*ring)->params.size >= buf_size;
        if (!buf_ok) {
            size_t want = (buf_size + (4u << 20) - 1) & ~(size_t)((4u << 20) - 1);
            buf_ok = pl_buf_recreate(p->gpu, ring,
                                     pl_buf_params(.size = want, .host_writable = true));
            vo_alloc_bump(p, &p->cnt_overlay_buf_grow);   // WP-A3: overlay ring grow
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
        stats_time_end(p->stats, "sub-upload");
        if (!ok) {
            MP_ERR(vo, "Failed uploading OSD texture!\n");
            talloc_free(upload_params.priv);
            break;
        }

        // WP-E3 checkpoint: the atlas upload is recorded (async, targets only
        // this build buffer's entry->tex); the parts build below is next.
        if (sub_guard_expired(p))
            goto bail;

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

        pl_tex overlay_tex = entry->tex;
#if HAVE_ASS_BLUR_DEFERRED
        // Deferred-blur (see ass_set_blur_deferred): libass emitted unblurred
        // coverage in pre-expanded bounds plus a per-part gaussian std-dev; do
        // the blur here on the GPU instead of on the CPU display path.
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
                if (!entry->blur_tex)
                    MP_TARRAY_POP(p->sub_scratch, p->num_sub_scratch, &entry->blur_tex);
                if (!entry->tmp_tex)
                    MP_TARRAY_POP(p->sub_scratch, p->num_sub_scratch, &entry->tmp_tex);
                if (pl_tex_recreate(p->gpu, &entry->blur_tex, &bp) &&
                    pl_tex_recreate(p->gpu, &entry->tmp_tex, &bp))
                {
                    stats_time_start(p->stats, "sub-blur");   // WP-A3: GPU blur pass
                    int64_t bl0 = want_t ? mp_time_ns() : 0;
                    for (int i = 0; i < item->num_parts; i++) {
                        // WP-E3 checkpoint: between per-part blur dispatch
                        // pairs (all writes go to this build buffer's blur
                        // scratch textures).
                        if (sub_guard_expired(p)) {
                            stats_time_end(p->stats, "sub-blur");
                            goto bail;
                        }
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
                    if (want_t)
                        p->dbg_blur_ns += mp_time_ns() - bl0;
                    stats_time_end(p->stats, "sub-blur");
                    overlay_tex = entry->blur_tex;
                }
            }
        }
#endif // HAVE_ASS_BLUR_DEFERRED

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

        // Capped-resolution composite (see --sub-render-res-limit): the sub
        // overlay was rasterized at render_w x render_h but its parts are in
        // display space. Composite the parts into a small intermediate texture
        // at that resolution, then replace them with a single overlay that the
        // final render bilinear-upscales onto the (8K) target. This cuts the
        // per-frame composite fill cost by ~(render/display)^2 for big animated
        // signs, the dominant cost of blend-subtitles=no at 8K. SDR libass subs
        // on the target frame only; everything else keeps the direct path.
        // render_w > 0 is set only by sd_ass for capped subtitle overlays
        // (OSD/stats and image subs leave it 0), so it both detects the cap and
        // identifies the sub overlay.
        if (item->render_w > 0 && item->render_w < subs->w &&
            coords == PL_OVERLAY_COORDS_DST_FRAME &&
            item->format == SUBBITMAP_LIBASS &&
            !item->video_color_space &&
            !(src && pl_color_transfer_is_hdr(frame->color.transfer)) &&
            p->osd_inter_fmt && p->osd_rr &&
            !(div[0] > 1 || div[1] > 1) &&
            entry->num_parts > 0)
        {
            // WP-E3 checkpoint: before recording the capped-res composite
            // (a whole pl_render_image, the heaviest single call here).
            if (sub_guard_expired(p))
                goto bail;
            int rw = item->render_w, rh = item->render_h;
            int iw = (rw + 63) & ~63, ih = (rh + 63) & ~63;
            bool iok = pl_tex_recreate(p->gpu, &entry->inter_tex, &(struct pl_tex_params) {
                .format = p->osd_inter_fmt,
                .w = MPMAX(iw, entry->inter_tex ? entry->inter_tex->params.w : 0),
                .h = MPMAX(ih, entry->inter_tex ? entry->inter_tex->params.h : 0),
                .renderable = true,
                .sampleable = true,
            });
            if (iok) {
                // Display-space bounding box of the parts, so the upscale pass
                // touches only the sub region (not the whole 8K frame).
                float bx0 = 1e9f, by0 = 1e9f, bx1 = -1e9f, by1 = -1e9f;
                for (int i = 0; i < entry->num_parts; i++) {
                    struct pl_overlay_part *pp = &entry->parts[i];
                    bx0 = MPMIN(bx0, pp->dst.x0); by0 = MPMIN(by0, pp->dst.y0);
                    bx1 = MPMAX(bx1, pp->dst.x1); by1 = MPMAX(by1, pp->dst.y1);
                }
                float rx = (float) rw / subs->w, ry = (float) rh / subs->h;
                for (int i = 0; i < entry->num_parts; i++) {
                    entry->parts[i].dst.x0 *= rx; entry->parts[i].dst.x1 *= rx;
                    entry->parts[i].dst.y0 *= ry; entry->parts[i].dst.y1 *= ry;
                }
                struct pl_overlay inter_ol = *ol;
                inter_ol.parts = entry->parts;
                inter_ol.num_parts = entry->num_parts;
                inter_ol.coords = PL_OVERLAY_COORDS_DST_FRAME;
                struct pl_frame inter = {
                    .num_planes = 1,
                    .planes[0] = {
                        .texture = entry->inter_tex,
                        .components = 4,
                        .component_mapping = {0, 1, 2, 3},
                    },
                    .repr = { .sys = PL_COLOR_SYSTEM_RGB,
                              .levels = PL_COLOR_LEVELS_FULL,
                              .alpha = PL_ALPHA_PREMULTIPLIED },
                    .color = pl_color_space_srgb,
                    .crop = { 0, 0, rw, rh },
                    .overlays = &inter_ol,
                    .num_overlays = 1,
                };
                struct pl_render_params op = pl_render_fast_params;
                op.background = PL_CLEAR_COLOR;
                op.border = PL_CLEAR_COLOR;
                op.background_transparency = 1.0f; // clear to transparent
                int64_t cc0 = want_t ? mp_time_ns() : 0;   // WP-A3: capped composite
                pl_render_image(p->osd_rr, NULL, &inter, &op);
                if (want_t)
                    p->dbg_capcomp_ns += mp_time_ns() - cc0;

                // Replace the N display-space parts with one upscale of the
                // composited intermediate over the parts' display bounding box.
                entry->parts[0] = (struct pl_overlay_part) {
                    .src = { bx0 * rx, by0 * ry, bx1 * rx, by1 * ry },
                    .dst = { bx0, by0, bx1, by1 },
                    .color = {1, 1, 1, 1},
                };
                entry->num_parts = 1;
                ol->tex = entry->inter_tex;
                ol->parts = entry->parts;
                ol->num_parts = 1;
                ol->mode = PL_OVERLAY_NORMAL;
                ol->repr.alpha = PL_ALPHA_PREMULTIPLIED;
                ol->color = pl_color_space_srgb;
            }
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

    talloc_free(subs);

    // WP-E3 commit: the build ran to completion; it becomes the presentable
    // snapshot. Record everything needed to validate a later stale-serve.
    if (present) {
        g->good = build;
        g->good_num = frame->num_overlays;
        g->good_pts = src ? src->pts : MP_NOPTS_VALUE;
        g->good_epoch = sub_epoch;
        g->good_res = res;
        g->good_flags = flags;
        g->good_coords = coords;
    }
    return true;

bail:
    // WP-E3 guard fired: abandon the partial build (the next build for this
    // consumer reuses the same buffer from scratch) and present the previous
    // complete overlay state -- IF it is still valid for this frame. It is
    // invalidated by: VOCTRL_RESET/reconfig/resize (good = -1, set in
    // control()/reconfig()/resize()), a sub track change (sub_epoch), any
    // geometry/flags/coords mismatch, and a pts discontinuity the reset path
    // might not have covered (backstop: never serve across a pts regression
    // or a > 0.5 s forward jump). When invalid, this frame simply presents
    // no overlays -- late-but-correct beats stale-but-wrong.
    talloc_free(subs);
    p->guard_fired = true;
    g->must_complete = true;
    frame->num_overlays = 0;
    if (g->good >= 0 && flags == g->good_flags && coords == g->good_coords &&
        osd_res_equals(res, g->good_res) && sub_epoch == g->good_epoch)
    {
        double now = src ? src->pts : MP_NOPTS_VALUE;
        bool pts_ok = now == g->good_pts ||    // same frame / OSD-only redraw
                      (now != MP_NOPTS_VALUE && g->good_pts != MP_NOPTS_VALUE &&
                       now >= g->good_pts && now - g->good_pts <= 0.5);
        if (pts_ok) {
            frame->overlays = g->states[g->good].overlays;
            frame->num_overlays = g->good_num;
        }
    }
    // Off the fast path (fires are exceptional); lets tests attribute exactly
    // what a guard engagement presented (stale snapshot pts vs no overlays).
    // A presented-nothing bail (no valid previous snapshot: cold start,
    // post-seek/reset, pts discontinuity, or >0.5 s content gap) is the visible
    // vanish -- count it as guard-empty, not stale-present. Sticky per frame.
    if (frame->num_overlays) {
        MP_VERBOSE(vo, "[present-guard] deadline exceeded at pts %f: presenting "
                   "previous overlays (pts %f)\n",
                   src ? src->pts : MP_NOPTS_VALUE, g->good_pts);
    } else {
        p->guard_presented_empty = true;
        MP_VERBOSE(vo, "[present-guard] deadline exceeded at pts %f: presenting "
                   "no overlays\n", src ? src->pts : MP_NOPTS_VALUE);
    }
    return false;
}

struct frame_priv {
    struct vo *vo;
    struct osd_guard subs;
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

// Effective reference white luminance in nits to assume for SDR content.
static float get_ref_luma(struct priv *p)
{
    const struct gl_video_opts *opts = p->opts_cache->opts;
    if (opts->hdr_reference_white)
        return opts->hdr_reference_white;

    // auto: follow the system reference white, if available
    struct ra_swapchain *sw = p->ra_ctx->swapchain;
    if (sw->fns->target_ref_luma)
        return sw->fns->target_ref_luma(sw);

    return 0;
}

static bool use_ref_luma(const struct pl_color_space *csp, const struct pl_color_space *target_csp)
{
    if (!pl_color_transfer_is_hdr(csp->transfer))
        return true;
#if PL_API_VER >= 362
    if (csp->transfer == PL_COLOR_TRC_SCRGB && target_csp && !pl_color_transfer_is_hdr(target_csp->transfer))
        return true;
#endif
    return false;
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
    float ref_luma;
    if (!pl_color_transfer_is_hdr(frame->color.transfer) && (ref_luma = get_ref_luma(p)))
        frame->color.hdr.max_luma = ref_luma;

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
    for (int s = 0; s < 2; s++) {   // WP-E3: both ping-pong buffers
        struct osd_state *state = &fp->subs.states[s];
        for (int i = 0; i < MP_ARRAY_SIZE(state->entries); i++) {
            pl_tex tex = state->entries[i].tex;
            if (tex)
                MP_TARRAY_APPEND(p, p->sub_tex, p->num_sub_tex, tex);
            pl_tex bl = state->entries[i].blur_tex;
            if (bl)
                MP_TARRAY_APPEND(p, p->sub_scratch, p->num_sub_scratch, bl);
            pl_tex tm = state->entries[i].tmp_tex;
            if (tm)
                MP_TARRAY_APPEND(p, p->sub_scratch, p->num_sub_scratch, tm);
            // Not pooled; destroy rather than leak with the frame.
            pl_tex_destroy(gpu, &state->entries[i].inter_tex);
            pl_tex_destroy(gpu, &state->entries[i].result_tex);
        }
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
                                 float min_luma, bool hint, float target_ref_luma,
                                 const struct pl_color_space *target_csp)
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
    if (target_ref_luma && (!target->color.hdr.max_luma || !hint) &&
        use_ref_luma(&target->color, target_csp)) {
        target->color.hdr.max_luma = target_ref_luma;
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
        fp->subs.good = -1;   // WP-E3: no complete overlay build yet

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
    float target_ref_luma = 0;
    if (target_unknown) {
        target_csp = (struct pl_color_space){
            .transfer = opts->target_trc ? opts->target_trc : pl_color_space_hdr10.transfer };
    } else {
        target_ref_luma = get_ref_luma(p);
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
        if (target_ref_luma && use_ref_luma(&hint, &target_csp))
            hint.hdr.max_luma = target_ref_luma;
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
    apply_target_options(p, &target, hint.hdr.min_luma, strict_sw_params,
                         target_ref_luma, &target_csp);
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
    // WP-A3: slow-frame timing. Only sample the clock when the MSGL_V slowframe
    // line can fire; otherwise this is just the (already-present) stats events.
    bool want_slow = mp_msg_test(vo->log, MSGL_V);
    // WP-E3 present guard: one deadline for ALL of this frame's overlay
    // builds (the main osd-update below plus the blend-subs updates in the
    // mix loop) so presentation never waits on the sub path longer than one
    // frame's worth of time. Default (-1) = the frame's duration; when the
    // frame carries none (redraws, stills) fall back to the cadence-inferred
    // approx_duration (there is no container fps at the vo level), then to a
    // ~24 fps budget.
    p->guard_fired = false;
    p->guard_presented_empty = false;
    int64_t gdl = 0;
    int gms = p->next_opts->sub_present_guard_ms;
    if (gms > 0) {
        gdl = gms * INT64_C(1000000);
    } else if (gms < 0) {
        if (frame->duration > 0) {
            gdl = (int64_t) frame->duration;
        } else if (frame->approx_duration > 0) {
            gdl = (int64_t) frame->approx_duration;
        } else {
            gdl = INT64_C(42000000);
        }
    }
    p->guard_deadline_ns = gdl;
    p->guard_t0 = gdl ? mp_time_ns() : 0;
    int64_t osd_t0 = want_slow ? mp_time_ns() : 0;
    stats_time_start(p->stats, "osd-update");
    update_overlays(vo, p->osd_res,
                    (frame->current && opts->blend_subs) ? OSD_DRAW_OSD_ONLY : 0,
                    PL_OVERLAY_COORDS_DST_FRAME, &p->osd_guard, &target, frame->current,
                    frame->current ? frame->current->params.stereo3d : 0, true);
    stats_time_end(p->stats, "osd-update");
    // Snapshot the phase timers now, before the blend-subs update_overlays loop
    // below can overwrite them (they live in priv, reset per update_overlays call).
    int64_t osd_ns = want_slow ? mp_time_ns() - osd_t0 : 0;
    int64_t sr_ns = p->dbg_subrender_ns, cc_ns = p->dbg_capcomp_ns, bl_ns = p->dbg_blur_ns;
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
                    bool done = update_overlays(vo, res, OSD_DRAW_SUB_ONLY,
                                                rel, &fp->subs, image, mpi,
                                                mpi->params.stereo3d, true);
                    stats_time_end(p->stats, "osd-blend-update");
                    // WP-E3: a bailed build presented this frame's previous
                    // complete overlays; leave osd_sync behind so the next
                    // frame rebuilds normally.
                    if (done)
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
    int64_t submit_t0 = want_slow ? mp_time_ns() : 0;   // WP-A3: render-submit wall time
    stats_time_start(p->stats, "render");
    bool render_ok = pl_render_image_mix(p->rr, &mix, &target, &params);
    stats_time_end(p->stats, "render");
    int64_t submit_ns = want_slow ? mp_time_ns() - submit_t0 : 0;
    if (!render_ok) {
        MP_ERR(vo, "Failed rendering frame!\n");
        goto done;
    }

    struct pl_frame ref_frame;
    pl_frames_infer_mix(p->rr, &mix, &target, &ref_frame);

#if HAVE_ASS_COMPOSITE_DEFERRED
    // WP-H1b: idle GPU glyph pre-fill for upcoming render-ahead frames. After
    // the frame's own work is recorded, so its budget check sees the true
    // remaining headroom before the present-guard deadline.
    sub_prefill_idle(vo);
#endif

    // WP-A3: per-frame counter samples (emit-on-change; no-op unless --dump-stats)
    // and the MSGL_V [slowframe] breakdown line (format kept byte-compatible with
    // the historical logs + TOOLS/subtest/parse_stats.py).
    emit_counter(vo, "gcache-flush",              p->cnt_gcache_flush,        &p->stat_gcache_flush);
    emit_counter(vo, "atlas-overflow",            p->cnt_atlas_overflow,      &p->stat_atlas_overflow);
    emit_counter(vo, "staging-grow",              p->cnt_staging_grow,        &p->stat_staging_grow);
    emit_counter(vo, "overlay-buf-grow",          p->cnt_overlay_buf_grow,    &p->stat_overlay_buf_grow);
    emit_counter(vo, "tex-realloc",               p->cnt_tex_realloc,         &p->stat_tex_realloc);
    emit_counter(vo, "raster-pool-grow",          p->cnt_raster_pool_grow,    &p->stat_raster_pool_grow);
    emit_counter(vo, "vo-alloc-after-first-frame", p->cnt_vo_alloc_after_first, &p->stat_vo_alloc_after_first);
    emit_counter(vo, "raster-dispatches",         p->cnt_raster_dispatches,   &p->stat_raster_dispatches);
    emit_counter(vo, "raster-tiles",              p->cnt_raster_tiles,        &p->stat_raster_tiles);
    emit_counter(vo, "gcache-epoch-advance",      p->cnt_gcache_epoch_advance, &p->stat_gcache_epoch_advance);
    emit_counter(vo, "gcache-evict-n",            p->cnt_gcache_evict_n,      &p->stat_gcache_evict_n);
    emit_counter(vo, "gcache-overcommit",         p->cnt_gcache_overcommit,   &p->stat_gcache_overcommit);
    emit_counter(vo, "shader-warmups",            p->cnt_shader_warmups,      &p->stat_shader_warmups);
    // WP-E3: guard engagements, once per frame no matter how many overlay
    // sets bailed (ra-stale counts dec_sub-level staleness separately; the
    // two layers are independent and never double-count). A bail that served
    // the previous overlays is stale-present; one that presented nothing (no
    // valid snapshot -- cold start / post-seek / discontinuity) is guard-empty.
    if (p->guard_fired) {
        if (p->guard_presented_empty)
            p->cnt_guard_empty++;
        else
            p->cnt_stale_present++;
    }
    emit_counter(vo, "stale-present",             p->cnt_stale_present,       &p->stat_stale_present);
    emit_counter(vo, "guard-empty",               p->cnt_guard_empty,         &p->stat_guard_empty);
    emit_counter(vo, "prefill-glyphs",            p->cnt_prefill_glyphs,      &p->stat_prefill_glyphs);
    emit_counter(vo, "glyphs-uncached",           p->cnt_glyph_uncached,      &p->stat_glyph_uncached);
    if (want_slow) {
        int64_t gpu_ns = 0;
        for (int i = 0; i < p->perf_fresh.count; i++)
            gpu_ns += p->perf_fresh.info[i].last;
        double osd_ms = osd_ns / 1e6, sr_ms = sr_ns / 1e6, cc_ms = cc_ns / 1e6,
               bl_ms = bl_ns / 1e6, sub_ms = submit_ns / 1e6, gpu_ms = gpu_ns / 1e6;
        double other_ms = osd_ms - sr_ms - cc_ms - bl_ms; // upload + overlay build
        double budget_ms = frame->duration > 0 ? frame->duration / 1e6 : 41.7;
        if (osd_ms > budget_ms)
            MP_VERBOSE(vo, "[slowframe] osd-update=%.1f (subrender=%.1f capcomp=%.1f "
                       "blur=%.1f other=%.1f) render-submit=%.1f gpu-passes=%.1f ms\n",
                       osd_ms, sr_ms, cc_ms, bl_ms, other_ms, sub_ms, gpu_ms);
    }

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
    p->first_frame_drawn = true; // WP-A3: gate vo-alloc-after-first-frame counting
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
    // WP-E3: geometry changed; a pre-resize overlay snapshot must not be
    // presented (the res equality check in the bail path is the backstop).
    p->osd_guard.good = -1;
}

static int reconfig(struct vo *vo, struct mp_image_params *params)
{
    struct priv *p = vo->priv;
    if (!p->ra_ctx->fns->reconfig(p->ra_ctx))
        return -1;

    // WP-E3: new video params; never serve an overlay snapshot across a
    // reconfigure (resize() below only invalidates when geometry changed).
    p->osd_guard.good = -1;
    resize(vo);
    mp_mutex_lock(&vo->params_mutex);
    vo->target_params = NULL;
    mp_mutex_unlock(&vo->params_mutex);
#if HAVE_ASS_COMPOSITE_DEFERRED
    // WP-E: cold-start warm-up on first config (off the hot path, before the
    // first frame). Preallocates the glyph atlas + upload rings and compiles
    // every OSD compute pass, so no first-use alloc/compile spikes the VO thread
    // once playback begins.
    gc_warmup(p);
    // WP-H1c: re-floor the legacy overlay pool + upload ring for the (possibly
    // changed) display size. Reconfig only -- before playback -- never
    // mid-playback; no-op when already at the right floor.
    ensure_overlay_pool(p);
    // WP-H5a: likewise re-floor the raster/compose pools to the (possibly
    // larger) display size. gc_warmup runs once; a later display-size increase
    // reaches this on reconfig. Uses plain gc_ensure (monotonic MPMAX), off the
    // hot path, so it never bumps raster-pool-grow.
    gc_prealloc_pools(p);
#endif
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
        apply_target_options(p, &target, 0, false, 0, NULL);
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
    // WP-E3: screenshots are not presentation -- always build fully (no
    // deadline) and never commit (present=false below), so a screenshot at
    // foreign geometry can't pollute the presentation snapshot.
    p->guard_deadline_ns = 0;
    p->guard_t0 = 0;
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
                        mpi->params.stereo3d, false);
    } else {
        // Disable overlays when blend_subs is disabled
        update_overlays(vo, osd, osd_flags, PL_OVERLAY_COORDS_DST_FRAME,
                        &p->osd_guard, &target, mpi,
                        mpi->params.stereo3d, false);
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
        // WP-E3: VOCTRL_RESET is the canonical playback-discontinuity signal
        // to a VO -- vo.c delivers it on the VO thread on every seek/playback
        // restart, before the first post-seek draw_frame. Invalidating here
        // guarantees a guard fire on the first post-seek frame can never
        // present a pre-seek overlay snapshot (the pts window in the bail
        // path is only a backstop behind this). The blend-subs snapshots
        // (frame_priv) need no invalidation: they die with their frames when
        // the queue is reset, and each is only ever served for its own frame.
        p->osd_guard.good = -1;
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
    for (int s = 0; s < 2; s++) {   // WP-E3: both ping-pong buffers
        struct osd_state *state = &p->osd_guard.states[s];
        for (int i = 0; i < MP_ARRAY_SIZE(state->entries); i++) {
            pl_tex_destroy(p->gpu, &state->entries[i].tex);
            pl_tex_destroy(p->gpu, &state->entries[i].blur_tex);
            pl_tex_destroy(p->gpu, &state->entries[i].tmp_tex);
            pl_tex_destroy(p->gpu, &state->entries[i].inter_tex);
            pl_tex_destroy(p->gpu, &state->entries[i].result_tex);
        }
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
    pl_tex_destroy(p->gpu, &p->work_tex);
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
    pl_renderer_destroy(&p->osd_rr);
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
    // WP-A3: seed emit-on-change sentinels so the first frame emits a baseline
    // value sample for every counter (so e.g. --assert-value gcache-flush==0
    // finds a sample on a clean run).
    p->stat_gcache_flush = p->stat_atlas_overflow = p->stat_staging_grow = -1;
    p->stat_overlay_buf_grow = p->stat_tex_realloc = p->stat_vo_alloc_after_first = -1;
    p->stat_raster_pool_grow = -1;
    p->stat_raster_dispatches = p->stat_raster_tiles = -1;
    p->stat_gcache_epoch_advance = p->stat_gcache_evict_n = p->stat_gcache_overcommit = -1;
    p->stat_shader_warmups = -1;
    p->stat_stale_present = -1;
    p->stat_prefill_glyphs = -1;
    p->stat_glyph_uncached = -1;
    p->osd_guard.good = -1;   // WP-E3: no complete overlay build yet

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
    p->blur_cache_sigma = -1.0f;   // force computing weights for the first sigma
    p->osd_sync = 1;

    // Dedicated renderer + RGBA target for the capped-resolution subtitle
    // overlay composite (--sub-render-res-limit). A separate renderer keeps the
    // tiny composite from thrashing the main 8K renderer's caches.
    p->osd_rr = pl_renderer_create(p->pllog, p->gpu);
    p->osd_inter_fmt = pl_find_fmt(p->gpu, PL_FMT_UNORM, 4, 8, 0,
        PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_LINEAR |
        PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_BLENDABLE);
    if (!p->osd_inter_fmt)
        MP_VERBOSE(vo, "No RGBA composite format; --sub-render-res-limit will "
                       "only cap subtitle rasterization, not compositing.\n");

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

    struct bstr shadername = mp_strip_ext(mp_basename_bstr(bstr0(shaderpath)));

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
