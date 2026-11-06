// Standalone adapter for the MVTools path used by AJN 3.2.1 rife_cuda.py:
// Super(pel=1), Analyse(isb=True), SCDetection(thscd1=240, thscd2=130).
// SPDX-License-Identifier: GPL-2.0-or-later
#include "ajn_mvtools.h"
#include "vendor/GroupOfPlanes.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <memory>
#include <vector>
#include "enhanced.h"

namespace {
struct Super {
    MVGroupOfFrames frames{};
    std::vector<uint8_t> storage[3];
    uint8_t *planes[3]{};
    ptrdiff_t strides[3]{};
    double pts = NAN;
    Super(const ajn_mvtools_frame &f, int levels, int mode) {
        int bytes = (f.bits + 7) / 8;
        int x = 1 << f.sub_w, y = 1 << f.sub_h;
        // Same padded pyramid layout as mv.Super. Only pel=1 is required.
        for (int p = 0; p < (mode == YPLANE ? 1 : 3); ++p) {
            int xp = p ? x : 1, yp = p ? y : 1;
            strides[p] = ((f.width / xp + 32 / xp) * bytes + 63) & ~63;
            auto size = PlaneSuperOffset(p, f.height / yp, levels, 1,
                                        16 / yp, strides[p], y);
            storage[p].resize(static_cast<size_t>(size));
            planes[p] = storage[p].data();
        }
        mvgofInit(&frames, levels, f.width, f.height, 1, 16, 16, mode, 0,
                  x, y, f.bits);
        mvgofUpdate(&frames, planes, strides);
    }
    ~Super() { mvgofDeinit(&frames); }
    void fill(const ajn_mvtools_frame &f, MVPlaneSet mode) {
        mvgofResetState(&frames);
        for (int p = 0; p < (mode == YPLANE ? 1 : 3); ++p)
            mvfFillPlane(frames.frames[0], f.planes[p], f.strides[p], p);
        mvgofReduce(&frames, mode, 2); // Super's default bilinear rfilter.
        mvgofPad(&frames, mode);
        mvgofRefine(&frames, mode, 2); // No-op refinement for pel=1.
        pts = f.pts;
    }
};
struct Motion {
    GroupOfPlanes gop{};
    explicit Motion(const MVAnalysisData &a) {
        gopInit(&gop, 8, 8, a.nLvCount, 1, a.nMotionFlags, 0, 0, 0,
                a.nBlkX, a.nBlkY, a.xRatioUV, a.yRatioUV, 0, a.bitsPerSample);
    }
    ~Motion() { gopDeinit(&gop); }
    void reset() {
        // Analyse starts each pair with zero predictors. Reusing the previous
        // pair's vectors would introduce temporal prediction and change cuts.
        for (int i = 0; i < gop.nLevelCount; ++i) {
            auto *plane = gop.planes[i];
            std::memset(plane->vectors, 0, plane->nBlkCount * sizeof(VECTOR));
        }
    }
};
struct Detector {
    ajn_mvtools_frame shape{};
    std::unique_ptr<Super> previous, current;
    std::unique_ptr<Motion> motion;
    std::vector<uint8_t> vectors;
    MVAnalysisData analysis{};
    MVPlaneSet mode = YPLANE;
    double enhanced_pts = NAN;
    double recent_raw[4]{}, recent_residual[4]{};
    int recent_next = 0;
    bool last_cut = false;
    void reset_enhanced() {
        enhanced_pts = NAN; recent_next = 0; last_cut = false;
        std::fill(std::begin(recent_raw), std::end(recent_raw), 0);
        std::fill(std::begin(recent_residual), std::end(recent_residual), 0);
    }
    bool matches(const ajn_mvtools_frame &f) const {
        return previous && shape.width == f.width && shape.height == f.height &&
            shape.bits == f.bits && shape.sub_w == f.sub_w && shape.sub_h == f.sub_h &&
            (shape.planes[1] != nullptr) == (f.planes[1] != nullptr);
    }
    void initialize(const ajn_mvtools_frame &f) {
        mode = f.planes[1] ? YUVPLANES : YPLANE;
        int x = 1 << f.sub_w, y = 1 << f.sub_h;
        int superLevels = 0;
        while (PlaneHeightLuma(f.height, superLevels, y, 16) >= y * 2 &&
               PlaneWidthLuma(f.width, superLevels, x, 16) >= x * 2) ++superLevels;
        analysis = {};
        auto &a = analysis;
        a.nBlkSizeX = a.nBlkSizeY = 8;
        a.nPel = a.nDeltaFrame = a.isBackward = 1;
        a.nWidth = f.width; a.nHeight = f.height;
        a.nBlkX = f.width / 8; a.nBlkY = f.height / 8;
        a.nHPadding = a.nVPadding = 16;
        a.xRatioUV = x; a.yRatioUV = y; a.bitsPerSample = f.bits;
        a.nMotionFlags = MOTION_IS_BACKWARD | (mode == YUVPLANES ? MOTION_USE_CHROMA_MOTION : 0);
#ifdef AJN_MVTOOLS_SSE2
        a.nMotionFlags |= MOTION_USE_SIMD;
#endif
        while (((a.nBlkX * 8) >> a.nLvCount) / 8 > 0 &&
               ((a.nBlkY * 8) >> a.nLvCount) / 8 > 0) ++a.nLvCount;
        previous = std::make_unique<Super>(f, superLevels, mode);
        current = std::make_unique<Super>(f, superLevels, mode);
        motion = std::make_unique<Motion>(a);
        vectors.resize(gopGetArraySize(&motion->gop));
        shape = f;
        reset_enhanced();
    }
};
bool valid(const ajn_mvtools_frame &f) {
    if (f.width < 16 || f.height < 16 || f.width > 8192 || f.height > 8192 ||
        f.bits < 8 || f.bits > 16 || f.sub_w < 0 || f.sub_w > 1 ||
        f.sub_h < 0 || f.sub_h > 1 || !std::isfinite(f.pts) ||
        (f.width % (1 << f.sub_w)) || (f.height % (1 << f.sub_h)) ||
        (!!f.planes[1] != !!f.planes[2])) return false;
    for (int p = 0; p < (f.planes[1] ? 3 : 1); ++p)
        if (!f.planes[p] || f.strides[p] < (f.width >> (p ? f.sub_w : 0)) * ((f.bits + 7) / 8)) return false;
    return true;
}
bool identical(const ajn_mvtools_frame &a, const ajn_mvtools_frame &b) {
    for (int p = 0; p < (a.planes[1] ? 3 : 1); ++p) {
        int width = (a.width >> (p ? a.sub_w : 0)) * ((a.bits + 7) / 8);
        int height = a.height >> (p ? a.sub_h : 0);
        for (int y = 0; y < height; ++y)
            if (std::memcmp(a.planes[p] + y * a.strides[p],
                            b.planes[p] + y * b.strides[p], width)) return false;
    }
    return true;
}
}
int ajn_mvtools_version(void) { return AJN_MVTOOLS_ABI; }
void *ajn_mvtools_create(void) { return new (std::nothrow) Detector; }
void ajn_mvtools_destroy(void *context) { delete static_cast<Detector *>(context); }
void ajn_mvtools_reset(void *context) {
    auto *d = static_cast<Detector *>(context);
    if (d) {
        if (d->previous) d->previous->pts = d->current->pts = NAN;
        d->reset_enhanced();
    }
}
int ajn_mvtools_analyze_enhanced(void *context, const ajn_mvtools_frame *a,
    const ajn_mvtools_frame *b, int thscd1, int thscd2, ajn_mvtools_result *out) {
    // This read-only shortcut precedes the original search. Validate its
    // inputs before reading pixels; every other pair follows the legacy path.
    if (context && a && b && out && valid(*a) && valid(*b) &&
        a->width == b->width && a->height == b->height && a->bits == b->bits &&
        a->sub_w == b->sub_w && a->sub_h == b->sub_h &&
        (!!a->planes[1] == !!b->planes[1]) && b->pts > a->pts &&
        thscd1 >= 0 && thscd1 <= 16320 && thscd2 >= 0 && thscd2 <= 255 && identical(*a, *b)) {
        auto &d = *static_cast<Detector *>(context);
        if (!d.matches(*a) || d.enhanced_pts != a->pts) d.reset_enhanced();
        // Preserve the normal history update for a zero-difference pair.
        d.recent_raw[d.recent_next] = d.recent_residual[d.recent_next] = 0;
        d.recent_next = (d.recent_next + 1) % 4;
        // A cached pyramid of a is also a valid pyramid of its exact repeat b.
        if (d.matches(*a)) {
            if (d.current->pts == a->pts) d.current->pts = b->pts;
            if (d.previous->pts == a->pts) d.previous->pts = b->pts;
        }
        int64_t sad = thscd1;
        if (a->planes[1]) sad += sad / ((1 << a->sub_w) * (1 << a->sub_h)) * 2;
        sad = static_cast<int64_t>(sad * ((1 << a->bits) - 1) / 255.0 + .5);
        int total = (a->width / 8) * (a->height / 8);
        *out = {0, 0, total, thscd2 * total / 256, sad};
        d.last_cut = false;
        d.enhanced_pts = b->pts;
        return 0;
    }
    int error = ajn_mvtools_analyze(context, a, b, thscd1, thscd2, out);
    if (error) return error;
    try {
        auto &d = *static_cast<Detector *>(context);
        if (d.enhanced_pts != a->pts) d.reset_enhanced();
        auto s = enhanced::measure(*a, *b, d.motion->gop.planes[0]->vectors,
                                  d.analysis.nBlkX, d.analysis.nBlkY, out->block_sad_threshold);
        double raw = s.all.difference(), residual = s.matched.difference();
        double spatial = s.changed.correlation(), aligned = s.matched.correlation();
        double histogram = s.matched.distance();
        double old_raw = std::max(2.0, *std::max_element(std::begin(d.recent_raw), std::end(d.recent_raw)));
        double old_residual = std::max(3.0, *std::max_element(std::begin(d.recent_residual), std::end(d.recent_residual)));
        // A block search can find similar patches even in unrelated pictures.
        // Require agreement in the unwarped changed region as well. For
        // deforming low-contrast textures, require stable appearance and motion
        // energy over time; this also carries a fade through its final frame.
        bool continuous = spatial > .95 || (spatial > .9 && aligned > .95) ||
            (spatial > .6 && aligned > .85 && s.all.distance() < .12 && s.changed.distance() < .22) ||
            (aligned > .65 && histogram < .22 && residual < .6 * s.changed.difference() &&
             s.all.distance() < .15 && s.changed.distance() < .22 && spatial > .5) ||
            (aligned > .65 && histogram < .22 && residual < .65 * s.changed.difference() &&
             s.all.distance() < .12 && s.changed.distance() < .22 && raw < 1.6 * old_raw) ||
            (std::max(s.all.variance(0), s.all.variance(1)) < 45 * 45 &&
             s.all.distance() < .2 && histogram < .25 && aligned > .55 &&
             raw < 1.5 * old_raw && residual < 1.5 * old_residual) ||
            (std::max(s.changed.variance(0), s.changed.variance(1)) < 16 &&
             (s.changed.difference() < 12 || raw < 1.5 * old_raw));
        bool enough_blocks = out->changed_blocks > int64_t(thscd2) * out->total_blocks * 75 / (130 * 256);
        bool flat_cut = enough_blocks && raw > std::max(12.0, 2 * old_raw) &&
                        std::min(s.all.variance(0), s.all.variance(1)) < 1;
        // Dark shots may have too few corners for feature correspondence.
        // After quiet history, anticorrelated (not flat) changed regions still
        // provide evidence of an edit. Undefined correlation during a fade is
        // zero, so it cannot take this path.
        bool sparse_dark = old_raw == 2 && old_residual == 3 && raw > 5 &&
            spatial < 0 && aligned < 0 && s.all.n &&
            s.all.sum[0] < 32 * s.all.n && s.all.sum[1] < 32 * s.all.n;
        bool weak = enough_blocks && !d.last_cut && spatial < .4 && aligned < .88 &&
            (s.all.distance() > .12 || s.changed.distance() > .3) && histogram > .15 &&
            raw > 1.8 * old_raw && residual > 2.1 * old_residual &&
            raw * residual > 5.5 * old_raw * old_residual && (!s.all.dominated() || sparse_dark);
        // A pan/zoom can change most pixels while retaining the same picture.
        // Require more feature matches to overturn a strong MVTools decision
        // than to reject a weak candidate. Insufficient detail is not evidence.
        if (out->cut && !continuous && s.all.distance() < .3) {
            continuous = enhanced::correspondence(*a, *b, 8) == enhanced::Correspondence::Moving;
            // Smaller image patches can track a deforming face that coarse
            // descriptors lose. Restrict refinement to already aligned,
            // ongoing motion; abrupt edits need the stronger coarse evidence.
            if (!continuous && spatial > .5 && aligned > .85 &&
                raw <= 2 * old_raw && residual <= 2 * old_residual)
                continuous = enhanced::correspondence(*a, *b, 8, true, d.motion->gop.planes[0]->vectors) == enhanced::Correspondence::Moving;
            // Effects can deform too much for sparse corner correspondence.
            // Require distributed patch continuity and an established motion
            // history before overriding a cut with this evidence.
            if (!continuous && aligned > .85 && raw <= 2 * old_raw && residual <= 2 * old_residual)
                continuous = enhanced::patch_motion(*a, *b, d.motion->gop.planes[0]->vectors,
                    d.analysis.nBlkX, d.analysis.nBlkY, out->block_sad_threshold);
        }
        // Shared borders or a similar palette can mask an abrupt edit. Do not
        // override the original decision without actual feature continuity.
        if (out->cut && continuous && spatial < .9 && histogram > .15 &&
            raw > 2 * old_raw && residual > 2 * old_residual)
            continuous = enhanced::correspondence(*a, *b, 8) == enhanced::Correspondence::Moving;
        // Luma continuity cannot disprove a cut detected only in chroma.
        if (raw < 1) continuous = false;
        bool cut = (out->cut && !continuous) || flat_cut;
        if (!cut && weak) {
            auto relation = enhanced::correspondence(*a, *b, 4);
            cut = relation == enhanced::Correspondence::Distinct ||
                  (sparse_dark && relation == enhanced::Correspondence::Unknown);
        }
        out->cut = cut;
        // Learn ordinary motion, not the discontinuity at an accepted cut.
        // There is no fixed quiet-shot warmup: a later cut may follow quickly.
        if (!out->cut) {
            d.recent_raw[d.recent_next] = raw;
            d.recent_residual[d.recent_next] = out->changed_blocks > out->total_blocks / 50 ? residual : 0;
            d.recent_next = (d.recent_next + 1) % 4;
        }
        d.last_cut = out->cut;
        d.enhanced_pts = b->pts;
        return 0;
    } catch (const std::exception &) {
        return -2;
    }
}
int ajn_mvtools_analyze(void *context, const ajn_mvtools_frame *a,
    const ajn_mvtools_frame *b, int thscd1, int thscd2, ajn_mvtools_result *out) {
    if (!context || !a || !b || !out || !valid(*a) || !valid(*b) ||
        a->width != b->width || a->height != b->height || a->bits != b->bits ||
        a->sub_w != b->sub_w || a->sub_h != b->sub_h ||
        (!!a->planes[1] != !!b->planes[1]) || b->pts <= a->pts ||
        thscd1 < 0 || thscd1 > 16320 || thscd2 < 0 || thscd2 > 255) return -1;
    try {
        auto &d = *static_cast<Detector *>(context);
        if (!d.matches(*a)) d.initialize(*a);
        if (d.current->pts == a->pts) d.previous.swap(d.current);
        if (d.previous->pts != a->pts) d.previous->fill(*a, d.mode);
        d.current->fill(*b, d.mode);
        auto &motion = *d.motion;
        motion.reset();
        auto scale = [bits = a->bits](int value) {
            return static_cast<int>(value * ((1 << bits) - 1) / 255.0 + 0.5);
        };
        // isb=True compares frame n with n+1. Preserve that direction.
        gopSearchMVs(&motion.gop, &d.previous->frames, &d.current->frames,
            SearchHex2, 2, 1, scale(1000), scale(1200), 50, 2, 1,
            d.vectors.data(), 0, nullptr, 0, 50, 0, scale(10000), 24, 1, 0,
            SearchExhaustive);
        // scaleThSCD and fpobIsSceneChange, with their original integer rounding.
        int64_t sad = thscd1;
        if (d.mode == YUVPLANES)
            sad += sad / (d.analysis.xRatioUV * d.analysis.yRatioUV) * 2;
        sad = static_cast<int64_t>(sad * ((1 << a->bits) - 1) / 255.0 + 0.5);
        int total = d.analysis.nBlkX * d.analysis.nBlkY;
        int count = 0;
        for (int i = 0; i < total; ++i)
            count += motion.gop.planes[0]->vectors[i].sad > sad;
        int threshold = thscd2 * total / 256;
        *out = {count > threshold, count, total, threshold, sad};
        return 0;
    } catch (const std::exception &) {
        return -2; // A C ABI cannot propagate a C++ exception into the player.
    }
}
