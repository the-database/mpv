// Focused regression checks for the legacy algorithm adapter.
#include "ajn_mvtools.h"
#include "vendor/SADFunctions.h"
#include "vendor/Luma.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "vendor/MVAnalysisData.h"
#include "enhanced.h"

static void require(bool value, const char *message) {
    if (!value) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
static void corner_order() {
    // Equal-strength corners at the score cutoff: preserve row-major ties
    // and the five-pixel spacing produced by neighbourhood suppression.
    std::vector<float> a(32 * 32), b(a.size());
    for (int y = 0; y < 32; ++y) for (int x = 0; x < 32; ++x) {
        a[y * 32 + x] = 2 * (x + y);
        b[y * 32 + x] = a[y * 32 + x] + 8;
    }
    auto points = enhanced::points(a, b, 32, 32);
    require(points.size() == 16, "corner score cutoff and suppression coverage");
    for (int i = 0; i < 16; ++i)
        require(points[i].x == 6 + 5 * (i % 4) && points[i].y == 6 + 5 * (i / 4),
                "equal-strength corners must retain row-major selection order");
}
static void patch_tracking() {
    // The destination corner is displaced from the physical patch centre.
    // Local refinement must recover the translation despite illumination,
    // then reject an equally good copy elsewhere in the picture.
    constexpr int w = 96, h = 80;
    std::vector<float> a(w * h), b(w * h);
    uint32_t random = 17;
    for (auto &v : a) {
        random = random * 1664525u + 1013904223u;
        v = float(20 + (random >> 24) / 2);
    }
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x)
        b[y * w + x] = a[std::max(0, y - 2) * w + std::max(0, x - 7)] + 25;
    enhanced::Point source{}, seed{}, other{}, found{};
    require(enhanced::patch(a, w, h, 32, 32, source) &&
            enhanced::patch(b, w, h, 37, 33, seed) &&
            enhanced::patch(b, w, h, 70, 34, other), "tracking fixture has textured patches");
    require(enhanced::track(source, b, {seed, other}, w, h, seed, found) &&
            found.x == 39 && found.y == 34, "track the physical patch beyond the selected corner");
    for (int y = -5; y <= 5; ++y) for (int x = -5; x <= 5; ++x)
        b[(34 + y) * w + 70 + x] = b[(34 + y) * w + 39 + x];
    enhanced::patch(b, w, h, 70, 34, other);
    require(!enhanced::track(source, b, {seed, other}, w, h, seed, found),
            "equally good repeated patches cannot establish a confident track");
}
static void regional_motion() {
    // Reciprocal tracks from the remaining head turn. The two eye regions
    // agree locally, while mouth/hair outliers prevent one global motion.
    const int head[][4] = {
        {184,105,162,109},{168,128,170,95},{171,123,166,102},{166,20,132,44},
        {204,33,173,42},{204,49,175,62},{196,60,160,78},{152,26,118,52},
        {190,74,153,92},{191,25,154,42},{209,27,177,36},{152,41,163,88},
        {168,97,133,45},{138,115,114,116},{168,14,136,38},{189,88,162,98},
        {217,73,169,98},{135,45,184,23},{44,128,70,125},{136,110,112,111}
    };
    std::vector<std::pair<enhanced::Point, enhanced::Point>> matches;
    for (const auto &m : head) matches.push_back({{m[0],m[1],{}},{m[2],m[3],{}}});
    require(!enhanced::moving_matches(matches, 8) && enhanced::regional_matches(matches, 45),
            "independent eye motions must confirm the deforming head turn");
    const int cut[][4] = {
        {191,112,228,35},{163,119,230,54},{161,109,227,37},{214,47,112,120},
        {170,103,99,78},{158,51,233,7},{195,99,227,41}
    };
    matches.clear();
    for (const auto &m : cut) matches.push_back({{m[0],m[1],{}},{m[2],m[3],{}}});
    require(!enhanced::regional_matches(matches, 45), "nearby real edit has no agreeing regions");
    matches.clear();
    for (int i = 0; i < 8; ++i)
        matches.push_back({{20 + i * 10,30,{}},{30 + i * 10,40,{}}});
    require(!enhanced::regional_matches(matches, 45), "collinear edge tracks cannot establish regions");
}
static void deforming_patches() {
    constexpr int w = 640, h = 384;
    std::vector<uint8_t> a(w * h), b(a.size());
    uint32_t random = 39;
    for (auto &v : a) { random = random * 1664525u + 1013904223u; v = random >> 24; }
    auto motion = [](int x, int y) { return y < h / 2 ? (x < w / 2 ? 8 : -8) : (x < w / 2 ? -12 : 12); };
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x)
        b[y * w + x] = a[y * w + std::clamp(x - motion(x, y), 0, w - 1)];
    std::vector<VECTOR> vectors(w / 8 * (h / 8));
    for (int y = 0; y < h / 8; ++y) for (int x = 0; x < w / 8; ++x)
        vectors[y * (w / 8) + x] = {motion(x * 8 + 4, y * 8 + 4), 0, 4096};
    ajn_mvtools_frame fa{}, fb{};
    fa.width = fb.width = w; fa.height = fb.height = h; fa.bits = fb.bits = 8;
    fa.planes[0] = a.data(); fb.planes[0] = b.data(); fa.strides[0] = fb.strides[0] = w;
    require(enhanced::patch_motion(fa, fb, vectors.data(), w / 8, h / 8, 360),
            "independent textured regions moving in opposing directions remain continuous");
    std::fill(b.begin() + w * (h / 3), b.end(), 0);
    require(!enhanced::patch_motion(fa, fb, vectors.data(), w / 8, h / 8, 360),
            "a shared moving strip cannot hide a change in the rest of the picture");
    for (auto &v : b) { random = random * 1664525u + 1013904223u; v = random >> 24; }
    require(!enhanced::patch_motion(fa, fb, vectors.data(), w / 8, h / 8, 360),
            "unrelated textures do not establish motion from plausible vectors alone");
}
static void kernels_match() {
    // Every block shape used by 8x8 analysis and its chroma planes. Source
    // blocks are packed by Analyse; reference rows have their own stride.
    for (int bits : {8, 16}) for (int width : {4, 8}) for (int height : {4, 8}) {
        int bytes = bits / 8, pitch = (width + 5) * bytes;
        std::vector<uint8_t> a(width * height * bytes), b(pitch * (height + 1));
        auto scalar = selectSADFunction(width, height, bits, 0, 0);
        auto simd = selectSADFunction(width, height, bits, 1, 0);
#ifdef AJN_MVTOOLS_SSE2
        require(simd != scalar, "SSE2 SAD dispatch must be active");
#endif
        for (int pattern = 0; pattern < 3; ++pattern) {
            for (size_t i = 0; i < a.size(); ++i)
                a[i] = pattern == 0 ? 0 : pattern == 1 ? 255 : static_cast<uint8_t>(i * 73 + 19);
            for (size_t i = 0; i < b.size(); ++i)
                b[i] = pattern == 0 ? 255 : pattern == 1 ? 0 : static_cast<uint8_t>(i * 37 + 29);
            require(scalar(a.data(), width * bytes, b.data() + bytes, pitch) ==
                    simd(a.data(), width * bytes, b.data() + bytes, pitch), "SSE2 SAD must equal scalar, including 16-bit extremes");
            if (width == 8 && height == 8) {
                auto luma = selectLumaFunction(width, height, bits, 0);
                auto fastLuma = selectLumaFunction(width, height, bits, 1);
                require(luma(b.data() + bytes, pitch) == fastLuma(b.data() + bytes, pitch), "SSE2 luma must equal scalar");
            }
        }
    }
}
static void matches_fresh(void *reused, const ajn_mvtools_frame &a, const ajn_mvtools_frame &b) {
    void *fresh = ajn_mvtools_create();
    ajn_mvtools_result expected{}, actual{};
    require(ajn_mvtools_analyze(fresh, &a, &b, 240, 130, &expected) == 0 &&
            ajn_mvtools_analyze(reused, &a, &b, 240, 130, &actual) == 0, "fresh and reused analysis");
    require(actual.cut == expected.cut && actual.changed_blocks == expected.changed_blocks &&
            actual.total_blocks == expected.total_blocks && actual.block_count_threshold == expected.block_count_threshold &&
            actual.block_sad_threshold == expected.block_sad_threshold, "reused motion workspace must not retain previous predictors");
    ajn_mvtools_destroy(fresh);
}
struct Image {
    ajn_mvtools_frame f{};
    std::vector<uint8_t> p[3];
    Image(int w, int h, int bits, bool chroma, double pts) {
        f.width = w; f.height = h; f.bits = bits; f.pts = pts;
        f.sub_w = f.sub_h = chroma ? 1 : 0;
        for (int n = 0; n < (chroma ? 3 : 1); ++n) {
            f.strides[n] = (w >> (n ? f.sub_w : 0)) * ((bits + 7) / 8);
            p[n].resize(f.strides[n] * (h >> (n ? f.sub_h : 0)));
            f.planes[n] = p[n].data();
        }
    }
    void fill(int plane, int value) {
        if (f.bits == 8) std::fill(p[plane].begin(), p[plane].end(), static_cast<uint8_t>(value));
        else for (size_t i = 0; i < p[plane].size(); i += 2) {
            p[plane][i] = value & 255; p[plane][i + 1] = value >> 8;
        }
    }
};
static void enhanced_cuts() {
    for (int bits : {8, 16}) {
        // A localized hard cut: the rest of the image stays dark, so fewer
        // than half of the blocks change. Test the full analysis path.
        Image a(128, 64, bits, false, 0), b(128, 64, bits, false, 1);
        a.fill(0, 0); b.fill(0, 0);
        int value = 80 * ((1 << bits) - 1) / 255;
        for (int y = 0; y < 64; ++y) for (int x = 40; x < 88; ++x) {
            size_t at = y * a.f.strides[0] + x * (bits / 8);
            a.p[0][at] = value & 255;
            if (bits == 16) a.p[0][at + 1] = value >> 8;
        }
        void *d = ajn_mvtools_create();
        ajn_mvtools_result legacy{}, enhanced{};
        require(ajn_mvtools_analyze(d, &a.f, &b.f, 240, 130, &legacy) == 0 && !legacy.cut &&
                legacy.changed_blocks > 75 * legacy.total_blocks / 256, "localized cut fixture must be missed by legacy");
        ajn_mvtools_reset(d);
        require(ajn_mvtools_analyze_enhanced(d, &a.f, &b.f, 240, 130, &enhanced) == 0 && enhanced.cut,
                "local picture cut must be detected immediately after reset");
        ajn_mvtools_reset(d);
        for (int i = 0; i < 3; ++i) {
            ajn_mvtools_frame next = a.f; a.f.pts = i; next.pts = i + 1;
            require(ajn_mvtools_analyze_enhanced(d, &a.f, &next, 240, 130, &enhanced) == 0 && !enhanced.cut,
                    "static warmup must remain continuous");
        }
        a.f.pts = 3; b.f.pts = 4;
        require(ajn_mvtools_analyze_enhanced(d, &a.f, &b.f, 240, 130, &enhanced) == 0 && enhanced.cut &&
                enhanced.changed_blocks == legacy.changed_blocks, "enhanced mode must recover a localized cut at both depths");
        // A seek preserves pair-local classification.
        ajn_mvtools_reset(d); a.f.pts = 0; b.f.pts = 1;
        require(ajn_mvtools_analyze_enhanced(d, &a.f, &b.f, 240, 130, &enhanced) == 0 && enhanced.cut,
                "enhanced seek must not suppress the first cut");
        b.fill(0, (1 << bits) - 1); a.f.pts = 10; b.f.pts = 11;
        require(ajn_mvtools_analyze_enhanced(d, &a.f, &b.f, 240, 130, &enhanced) == 0 && enhanced.cut,
                "large change to a flat frame is a cut without warmup");
        // A fade onset can have undefined correlation: both frames are flat.
        a.fill(0, 0); b.fill(0, 5 * ((1 << bits) - 1) / 255);
        a.f.pts = 12; b.f.pts = 13;
        require(ajn_mvtools_analyze_enhanced(d, &a.f, &b.f, 240, 130, &enhanced) == 0 && !enhanced.cut,
                "small fade out of black must not hold interpolation");
        // Global illumination changes preserve structure despite high SAD.
        for (int y = 0; y < 64; ++y) for (int x = 0; x < 128; ++x) {
            int v = 20 + (x * 7 + y * 13) % 100;
            for (int i = 0; i < 2; ++i) {
                auto &image = i ? b : a;
                int sample = (v + i * 30) * ((1 << bits) - 1) / 255;
                size_t at = y * image.f.strides[0] + x * (bits / 8);
                image.p[0][at] = sample & 255;
                if (bits == 16) image.p[0][at + 1] = sample >> 8;
            }
        }
        a.f.pts = 14; b.f.pts = 15;
        require(ajn_mvtools_analyze_enhanced(d, &a.f, &b.f, 240, 130, &enhanced) == 0 && !enhanced.cut,
                "illumination must not masquerade as a cut at either depth");
        ajn_mvtools_reset(d);
        for (int step = 4; step > 0; --step) {
            for (int y = 0; y < 64; ++y) for (int x = 0; x < 128; ++x) {
                for (int i = 0; i < 2; ++i) {
                    auto &image = i ? b : a;
                    int v = (200 + (x + y) % 32) * (step - i) / 4;
                    int sample = v * ((1 << bits) - 1) / 255;
                    size_t at = y * image.f.strides[0] + x * (bits / 8);
                    image.p[0][at] = sample & 255;
                    if (bits == 16) image.p[0][at + 1] = sample >> 8;
                }
            }
            a.f.pts = 20 + 4 - step; b.f.pts = a.f.pts + 1;
            require(ajn_mvtools_analyze_enhanced(d, &a.f, &b.f, 240, 130, &enhanced) == 0 && !enhanced.cut,
                    "a continuous fade must reach black without a final hold");
        }
        ajn_mvtools_destroy(d);
    }
}
static void repeated_frames() {
    for (int bits : {8, 16}) {
        Image a(128, 64, bits, true, 0), b(128, 64, bits, true, 1);
        int scale = ((1 << bits) - 1) / 255;
        a.fill(0, 40 * scale); b.fill(0, 40 * scale);
        a.fill(1, 128 * scale); b.fill(1, 128 * scale);
        a.fill(2, 128 * scale); b.fill(2, 128 * scale);
        void *fast = ajn_mvtools_create(), *legacy = ajn_mvtools_create();
        ajn_mvtools_result expected{}, actual{};
        require(ajn_mvtools_analyze(legacy, &a.f, &b.f, 240, 130, &expected) == 0 &&
                ajn_mvtools_analyze_enhanced(fast, &a.f, &b.f, 240, 130, &actual) == 0 &&
                actual.cut == expected.cut && actual.changed_blocks == expected.changed_blocks &&
                actual.total_blocks == expected.total_blocks &&
                actual.block_count_threshold == expected.block_count_threshold &&
                actual.block_sad_threshold == expected.block_sad_threshold,
                "an exact repeat preserves the full legacy result at both depths");
        // Luma equality alone must never bypass a chroma edit.
        a.f.pts = 1; b.f.pts = 2; b.fill(1, 0); b.fill(2, 255 * scale);
        require(ajn_mvtools_analyze_enhanced(fast, &a.f, &b.f, 240, 130, &actual) == 0 && actual.cut,
                "identical luma with changed chroma still needs analysis");
        // Reuse a populated pyramid through repeats, then switch API modes.
        a.f.pts = 2; b.f.pts = 3; a.fill(1, 0); a.fill(2, 255 * scale);
        require(ajn_mvtools_analyze_enhanced(fast, &a.f, &b.f, 240, 130, &actual) == 0 && !actual.cut,
                "repeat after a cut remains continuous");
        a.f.pts = 3; b.f.pts = 4; b.fill(0, 120 * scale);
        matches_fresh(fast, a.f, b.f);
        ajn_mvtools_reset(fast);
        a.f.pts = 0; b.f.pts = 1; b.fill(0, 40 * scale);
        require(ajn_mvtools_analyze_enhanced(fast, &a.f, &b.f, -1, 130, &actual) == -1,
                "exact repeats must not bypass parameter validation");
        ajn_mvtools_destroy(fast); ajn_mvtools_destroy(legacy);
    }
}
int main(int argc, char **argv) {
    corner_order();
    patch_tracking();
    regional_motion();
    deforming_patches();
    kernels_match();
    enhanced_cuts();
    repeated_frames();
    void *d = ajn_mvtools_create(); require(d, "create");
    ajn_mvtools_result r{};
    Image a(64, 64, 8, false, 0), b(64, 64, 8, false, 1);
    require(ajn_mvtools_analyze(d, &a.f, &b.f, 240, 130, &r) == 0 && !r.cut, "identical frames");
    b.fill(0, 4); ajn_mvtools_reset(d);
    require(ajn_mvtools_analyze(d, &a.f, &b.f, 240, 130, &r) == 0 && r.cut && r.changed_blocks == 64 &&
            r.block_sad_threshold == 240 && r.block_count_threshold == 32, "legacy thresholds");
    require(ajn_mvtools_analyze(d, &a.f, &b.f, 256, 130, &r) == 0 && !r.cut, "strict SAD comparison");
    Image ca(64, 64, 10, true, 0), cb(64, 64, 10, true, 1);
    cb.fill(1, 400); cb.fill(2, 400);
    require(ajn_mvtools_analyze(d, &ca.f, &cb.f, 240, 130, &r) == 0 && r.cut &&
            r.block_sad_threshold == 1444, "10-bit chroma and threshold rounding");
    require(ajn_mvtools_analyze_enhanced(d, &ca.f, &cb.f, 240, 130, &r) == 0 && r.cut,
            "luma-only continuity must preserve a chroma-only cut");
    // A seek must discard a cached super frame even when timestamps repeat.
    cb.fill(1, 0); cb.fill(2, 0); ajn_mvtools_reset(d);
    require(ajn_mvtools_analyze(d, &ca.f, &cb.f, 240, 130, &r) == 0 && !r.cut, "seek reset");
    require(ajn_mvtools_analyze(d, &b.f, &a.f, 240, 130, &r) == -1, "reversed timestamps");
    require(ajn_mvtools_analyze(d, &a.f, &b.f, 16321, 130, &r) == -1, "threshold boundary");
    int width = argc > 1 ? std::atoi(argv[1]) : 1280;
    int height = argc > 2 ? std::atoi(argv[2]) : 720;
    Image panA(width, height, 8, true, 0), panB(width, height, 8, true, 1);
    // Textured content translated by four pixels; the border is replicated.
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x)
        panA.p[0][y * width + x] = static_cast<uint8_t>((x / 8 * 17 + y / 8 * 31) % 220 + 16);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x)
        panB.p[0][y * width + x] = panA.p[0][y * width + std::max(0, x - 4)];
    panA.fill(1, 128); panA.fill(2, 128); panB.fill(1, 128); panB.fill(2, 128);
    auto start = std::chrono::steady_clock::now();
    require(ajn_mvtools_analyze(d, &panA.f, &panB.f, 240, 130, &r) == 0 && !r.cut, "pan must remain continuous");
    require(ajn_mvtools_analyze_enhanced(d, &panA.f, &panB.f, 240, 130, &r) == 0 && !r.cut,
            "enhanced mode must keep a translated shot continuous");
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    Image next(width, height, 8, true, 2);
    next.fill(1, 128); next.fill(2, 128);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x)
        next.p[0][y * width + x] = panB.p[0][y * width + std::max(0, x - 4)];
    matches_fresh(d, panB.f, next.f);
    panA.f.pts = 3;
    matches_fresh(d, next.f, panA.f);
    // Reinitialization and seek reset must also preserve fresh-pair results.
    matches_fresh(d, ca.f, cb.f);
    ajn_mvtools_reset(d);
    next.f.pts = 4;
    matches_fresh(d, panA.f, next.f);
    std::printf("PASS enhanced localized cuts/motion/reset, SIMD/scalar parity, workspace reuse, legacy thresholds, chroma, depth, reset and pan; %dx%d first pair %.3f ms (%d/%d changed blocks)\n",
                 width, height, ms, r.changed_blocks, r.total_blocks);
    ajn_mvtools_destroy(d);
}
