/* SPDX-License-Identifier: LGPL-2.1-or-later */
#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "video/out/display_rate.h"

static double choose_rate(struct mp_display_rate rate, const unsigned *modes, int n)
{
    double best = 0, score = -1;
    for (int i = 0; i < n; i++) {
        double hz = mp_display_rate_from_gdi(modes[i]);
        double s = mp_display_rate_score(rate, hz);
        if (s > score) { best = hz; score = s; }
    }
    return best;
}

static double choose(double fps, const unsigned *modes, int n)
{
    return choose_rate((struct mp_display_rate){.fps = fps, .variable = fps <= 0},
                       modes, n);
}

static struct mp_display_rate analyze(double *pts, int count,
                                      struct mp_display_cadence *cadences)
{
    int runs = mp_display_rate_analyze(pts, count, cadences);
    return (struct mp_display_rate){
        .fps = runs == 1 ? 1 / cadences[0].interval : 0,
        .variable = runs != 1,
        .cadences = cadences,
        .num_cadences = runs,
        .scale = 1,
    };
}

static void mixed_pts(double *pts, int first, int second, double fps1, double fps2)
{
    for (int n = 0; n <= first + second; n++) {
        double t = n <= first ? n / fps1 : first / fps1 + (n - first) / fps2;
        pts[n] = round(t * 1000) / 1000;
    }
}

int main(void)
{
    const unsigned tv[] = {23, 24, 25, 29, 30, 50, 59, 60};
    const unsigned integer_modes[] = {60, 48, 30, 24, 72, 120, 100, 75, 96, 144};
    const unsigned exact_higher[] = {24, 47, 144};
    double pts[600];
    struct mp_display_cadence cadences[600];
    for (int n = 0; n < 600; n++)
        pts[n] = round(n * 1001.0 / 24) / 1000;
    // Millisecond-rounded 23.976 timestamps remain CFR, including B-frame
    // packet reordering. Exact higher modes outrank low near-multiples.
    double temp = pts[25]; pts[25] = pts[28]; pts[28] = temp;
    struct mp_display_rate rate = analyze(pts, 600, cadences);
    assert(!rate.variable && rate.num_cadences == 1);
    double fps = rate.fps;
    assert(fabs(fps - 24000.0 / 1001) < 0.001);
    assert(fabs(choose(fps, tv, 8) - 24000.0 / 1001) < 0.001);
    assert(choose(fps, integer_modes, 10) == 24);
    assert(fabs(choose(fps, exact_higher, 3) - 48000.0 / 1001) < 0.001);

    // Two long CFR sections are mixed as a whole. Sampling just the opening
    // would choose a low rate and force a later switch.
    for (int n = 300; n < 600; n++)
        pts[n] = round(300 * 1001.0 / 24 + (n - 300) * 1001.0 / 30) / 1000;
    rate = analyze(pts, 600, cadences);
    assert(rate.variable && rate.num_cadences > 1);
    const unsigned common[] = {60, 120, 144};
    const unsigned fractional[] = {120, 143, 119, 144};
    assert(choose_rate(rate, common, 3) == 120);
    assert(fabs(choose_rate(rate, fractional, 4) - 120000.0 / 1001) < 0.001);
    // A missing/incomplete profile, unlike measured VFR, still uses maximum Hz.
    assert(choose(0, tv, 8) == 60);
    assert(choose(0, integer_modes, 10) == 144);

    double t = 0;
    for (int n = 0; n < 600; n++) {
        pts[n] = round(t * 1000) / 1000;
        t += n % 2 ? 1.0 / 24 : 1.0 / 30;
    }
    rate = analyze(pts, 600, cadences);
    assert(rate.variable && rate.num_cadences > 1);
    assert(choose_rate(rate, common, 3) == 120);
    rate.scale = 2;
    const unsigned doubled[] = {120, 144, 240};
    assert(choose_rate(rate, doubled, 3) == 240);
    assert(mp_display_rate_analyze(NULL, 0, cadences) == 0);
    pts[10] = NAN;
    assert(mp_display_rate_analyze(pts, 600, cadences) == 0);
    const unsigned ntsc_only[] = {23, 24, 59, 60};
    assert(choose(25, ntsc_only, 4) == 60);

    // A 60 Hz-limited display has no common multiple for 24/30. Prefer the
    // mode with less whole-episode error, rather than always choosing 60 Hz.
    const unsigned limited[] = {24, 30, 48, 60};
    mixed_pts(pts, 480, 30, 24, 30); // 20 seconds at 24, one second at 30
    rate = analyze(pts, 511, cadences);
    assert(choose_rate(rate, limited, 4) == 48);
    mixed_pts(pts, 24, 570, 24, 30); // one second at 24, 19 seconds at 30
    rate = analyze(pts, 595, cadences);
    assert(choose_rate(rate, limited, 4) == 60);

    // Nearby cadences must not be collapsed by millisecond rounding.
    mixed_pts(pts, 240, 250, 24, 25);
    rate = analyze(pts, 491, cadences);
    assert(rate.variable && rate.num_cadences > 1);
    const unsigned pal_mix[] = {48, 50, 120, 144, 600};
    assert(choose_rate(rate, pal_mix, 5) == 600);

    // Continuous irregular timing remains measured VFR, not an unknown input.
    t = 0;
    for (int n = 0; n < 600; n++) {
        pts[n] = t;
        t += (27 + n % 17) / 1000.0;
    }
    rate = analyze(pts, 600, cadences);
    assert(rate.variable && rate.num_cadences > 1);
    double duration = 0;
    int intervals = 0;
    for (int n = 0; n < rate.num_cadences; n++) {
        duration += cadences[n].interval * cadences[n].frames;
        intervals += cadences[n].frames;
    }
    assert(intervals == 599 && fabs(duration - pts[599] + pts[0]) < 1e-9);
    assert(mp_display_rate_score(rate, 144) > mp_display_rate_score(rate, 60));
    puts("PASS whole-file CFR/VFR; rounded/reordered PTS; common multiples; duration-weighted selection; unknown fallback; playback scale");
    return 0;
}
