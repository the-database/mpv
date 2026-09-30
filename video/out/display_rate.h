/* SPDX-License-Identifier: LGPL-2.1-or-later */
#ifndef MP_DISPLAY_RATE_H
#define MP_DISPLAY_RATE_H

#include <stdbool.h>
#include <math.h>
#include <stdlib.h>

struct mp_display_cadence {
    double interval;
    int frames;
};

struct mp_display_rate {
    double fps;
    bool variable;
    bool apply;
    // Owned by the stream header, which outlives the pending mode change.
    const struct mp_display_cadence *cadences;
    int num_cadences;
    double scale; // initial playback speed and declared filter multiplier
};

// A difference between two millisecond-rounded PTS can be off by 1 ms.
#define MP_DISPLAY_PTS_TOLERANCE 0.001000001

static inline int mp_display_pts_compare(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

// Analyze a complete scan, in presentation order, into consecutive cadence
// runs. A run must fit a uniform timestamp grid within container rounding, not
// merely have similar individual intervals (which could merge 24 and 25 fps).
// Short irregular runs are retained too. The caller supplies count - 1 entries.
// Return zero for unverified timestamps; a single run establishes CFR.
static inline int mp_display_rate_analyze(double *pts, int count,
                                         struct mp_display_cadence *cadences)
{
    if (count < 12)
        return 0;
    for (int i = 0; i < count; i++) {
        if (!isfinite(pts[i]))
            return 0;
    }
    qsort(pts, count, sizeof(*pts), mp_display_pts_compare);
    for (int i = 1; i < count; i++) {
        double dt = pts[i] - pts[i - 1];
        if (dt <= 0 || !isfinite(dt))
            return 0;
    }

    int start = 0, runs = 0;
    double low = 0, high = INFINITY;
    for (int i = 1; i < count; i++) {
        double span = pts[i] - pts[start];
        double next_low = fmax(low, (span - MP_DISPLAY_PTS_TOLERANCE) / (i - start));
        double next_high = fmin(high, (span + MP_DISPLAY_PTS_TOLERANCE) / (i - start));
        if (next_low > next_high) {
            int frames = i - 1 - start;
            cadences[runs++] = (struct mp_display_cadence){
                .interval = (pts[i - 1] - pts[start]) / frames,
                .frames = frames,
            };
            start = i - 1;
            span = pts[i] - pts[start];
            next_low = span - MP_DISPLAY_PTS_TOLERANCE;
            next_high = span + MP_DISPLAY_PTS_TOLERANCE;
        }
        low = next_low;
        high = next_high;
    }
    cadences[runs++] = (struct mp_display_cadence){
        .interval = (pts[count - 1] - pts[start]) / (count - 1 - start),
        .frames = count - 1 - start,
    };
    return runs;
}

// For measured VFR, prefer a mode fitting every run, then minimize the
// duration-weighted frame-hold error. An interval spanning n + f refreshes
// needs n/n+1 holds; their mean squared duration error is f*(1-f)/hz^2.
// Accounting for PTS rounding prevents 41/42 ms timestamps from being mistaken
// for genuine timing variation. No frames are decoded or retimed here.
static inline double mp_display_rate_vfr_score(struct mp_display_rate rate,
                                               double hz)
{
    bool exact = true, near_match = true;
    double error = 0, duration = 0;
    for (int i = 0; i < rate.num_cadences; i++) {
        struct mp_display_cadence c = rate.cadences[i];
        double interval = c.interval / rate.scale;
        double ticks = interval * hz;
        double nearest = round(ticks);
        double rounding = MP_DISPLAY_PTS_TOLERANCE / (c.frames * rate.scale);
        double distance = fmax(0, fabs(nearest / hz - interval) - rounding);
        double relative = distance / interval;
        bool fits = nearest >= 1 && relative < 0.0005;
        exact &= fits;
        near_match &= nearest >= 1 && relative < 0.002;
        double f = fits ? 0 : fmin(0.5, distance * hz);
        double weight = interval * c.frames;
        error += weight * f * (1 - f) / (hz * hz);
        duration += weight;
    }
    if (exact)
        return 1000000 - hz;
    if (near_match)
        return 500000 - hz;
    // Milliseconds squared keep the score well-scaled. Higher remains better.
    return 1 / (1 + 1000000 * error / duration);
}

// Prefer the lowest exact multiple for CFR, then a near multiple. Measured
// VFR uses its whole-file profile; unverified inputs retain the highest-rate
// fallback. Exact matches always win over near ones.
static inline double mp_display_rate_score(struct mp_display_rate rate, double hz)
{
    if (!(hz > 1) || (!rate.variable && !(rate.fps > 0)))
        return -1;
    if (rate.variable && rate.num_cadences > 0 &&
        isfinite(rate.scale) && rate.scale > 0)
        return mp_display_rate_vfr_score(rate, hz);
    if (rate.variable)
        return hz;
    double multiple = round(hz / rate.fps);
    if (!rate.variable && multiple >= 1) {
        double error = fabs(hz / (rate.fps * multiple) - 1);
        if (error < 0.0005)
            return 1000000 - hz;
        // Allow NTSC/integer pairs plus millisecond timestamp rounding, but
        // never PAL/NTSC mismatches such as 25 fps on a 24 Hz display.
        if (error < 0.002)
            return 500000 - hz;
    }
    return hz;
}

// GDI mode enumeration uses integer labels for fractional NTSC modes, as does
// mpv's existing GDI refresh-rate reader. Keep those modes distinct from 24/60.
static inline double mp_display_rate_from_gdi(unsigned hz)
{
    switch (hz) {
    case 23: case 29: case 47: case 59: case 71: case 89: case 95:
    case 119: case 143: case 164: case 239: case 359: case 479:
        return (hz + 1) / 1.001;
    default:
        return hz > 1 ? hz : 0;
    }
}
#endif
