// AJN's optional scene classifier. The legacy MVTools search is unchanged.
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <array>
#include <cmath>
#include <vector>

namespace enhanced {
struct Moments {
    int histogram[2][32]{};
    int n = 0;
    int64_t sum[2]{}, square[2]{}, product = 0, absolute = 0;
    void add(int a, int b) {
        ++n; ++histogram[0][a / 8]; ++histogram[1][b / 8];
        sum[0] += a; sum[1] += b;
        square[0] += a * a; square[1] += b * b;
        product += a * b; absolute += std::abs(a - b);
    }
    double difference() const { return n ? double(absolute) / n : 0; }
    double variance(int i) const {
        return n ? std::max(0.0, double(square[i]) / n - std::pow(double(sum[i]) / n, 2)) : 0;
    }
    double correlation() const {
        double va = double(n) * square[0] - double(sum[0]) * sum[0];
        double vb = double(n) * square[1] - double(sum[1]) * sum[1];
        return va > 0 && vb > 0 ? (double(n) * product - double(sum[0]) * sum[1]) / std::sqrt(va * vb) : 0;
    }
    double distance() const {
        int d = 0;
        for (int i = 0; i < 32; ++i) d += std::abs(histogram[0][i] - histogram[1][i]);
        return n ? d / (2.0 * n) : 0;
    }
    bool dominated() const {
        for (const auto &h : histogram)
            for (int count : h) if (count > n * .85) return true;
        return false;
    }
};
inline int pixel(const ajn_mvtools_frame &f, int x, int y) {
    const auto *row = f.planes[0] + y * f.strides[0];
    if (f.bits == 8) return row[x];
    uint16_t v; std::memcpy(&v, row + x * 2, sizeof(v));
    return v >> (f.bits - 8);
}
struct Appearance { Moments all, changed, matched; };
inline Appearance measure(const ajn_mvtools_frame &a, const ajn_mvtools_frame &b,
                          const VECTOR *vectors, int nx, int ny, int64_t threshold) {
    Appearance s;
    int sx = (a.width + 239) / 240, sy = (a.height + 134) / 135;
    for (int y = 0; y < ny * 8; y += sy) for (int x = 0; x < nx * 8; x += sx) {
        int av = pixel(a, x, y), bv = pixel(b, x, y);
        s.all.add(av, bv);
        const auto &v = vectors[(y / 8) * nx + x / 8];
        if (v.sad <= threshold) continue;
        s.changed.add(av, bv);
        // MVTools pel=1 vectors map the previous picture into the current one.
        s.matched.add(av, pixel(b, std::clamp(x + v.x, 0, b.width - 1),
                                  std::clamp(y + v.y, 0, b.height - 1)));
    }
    return s;
}

inline bool patch_motion(const ajn_mvtools_frame &a, const ajn_mvtools_frame &b,
                         const VECTOR *vectors, int nx, int ny, int64_t threshold) {
    // Expanding effects need not retain corners or one rigid motion. Verify
    // the existing block vectors against larger, textured neighbourhoods.
    // Independent moving patches must cover a two-dimensional area in both
    // pictures; a shared border or one matched object is insufficient.
    int source_cells[12]{}, target_cells[12]{}, good = 0;
    auto spread = [](const int *cells) {
        for (int i = 0; i < 12; ++i) if (cells[i] >= 3)
            for (int j = 0; j < i; ++j) if (cells[j] >= 3)
                for (int k = 0; k < j; ++k) if (cells[k] >= 3 &&
                    (i % 4 - k % 4) * (j / 4 - k / 4) !=
                    (j % 4 - k % 4) * (i / 4 - k / 4)) return true;
        return false;
    };
    for (int by = 2; by < ny - 2; by += 4) for (int bx = 2; bx < nx - 2; bx += 4) {
        const auto &v = vectors[by * nx + bx];
        if (v.sad <= threshold || int64_t(v.x) * v.x + int64_t(v.y) * v.y <= 16) continue;
        int x = bx * 8 + 4, y = by * 8 + 4, u = x + v.x, z = y + v.y;
        if (u < 16 || z < 16 || u + 16 >= b.width || z + 16 >= b.height) continue;
        Moments matched;
        int direct = 0;
        for (int yy = -16; yy <= 16; yy += 4) for (int xx = -16; xx <= 16; xx += 4) {
            int value = pixel(a, x + xx, y + yy);
            direct += std::abs(value - pixel(b, x + xx, y + yy));
            matched.add(value, pixel(b, u + xx, z + yy));
        }
        if (std::min(matched.variance(0), matched.variance(1)) <= 100 ||
            matched.correlation() <= .9 || matched.absolute >= .6 * direct) continue;
        ++good;
        ++source_cells[(y * 3 / a.height) * 4 + x * 4 / a.width];
        ++target_cells[(z * 3 / b.height) * 4 + u * 4 / b.width];
        if (good >= 12 && spread(source_cells) && spread(target_cells)) return true;
    }
    return false;
}

struct Point { int x, y; std::array<float, 36> descriptor; };
inline std::vector<float> thumbnail(const ajn_mvtools_frame &f, int w, int h) {
    std::vector<float> out(w * h);
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
        int x0 = x * f.width / w, x1 = (x + 1) * f.width / w;
        int y0 = y * f.height / h, y1 = (y + 1) * f.height / h, sum = 0;
        for (int yy = y0; yy < y1; ++yy) for (int xx = x0; xx < x1; ++xx) sum += pixel(f, xx, yy);
        out[y * w + x] = float(sum) / ((x1 - x0) * (y1 - y0));
    }
    return out;
}
inline std::vector<Point> points(const std::vector<float> &a,
                                 const std::vector<float> &b, int w, int h) {
    std::vector<float> score(w * h);
    std::vector<int> candidates;
    for (int y = 6; y < h - 6; ++y) for (int x = 6; x < w - 6; ++x) {
        int i = y * w + x;
        if (std::abs(a[i] - b[i]) <= 3) continue;
        float gx = (a[i + 1] - a[i - 1]) * .5f, gy = (a[i + w] - a[i - w]) * .5f;
        score[i] = std::min(gx * gx, gy * gy);
        if (score[i] >= 4) candidates.push_back(i);
    }
    // Match max_element's first-pixel tie break without rescanning the image
    // for each corner. Suppressed candidates are skipped in this fixed order.
    std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
        return score[a] != score[b] ? score[a] > score[b] : a < b;
    });
    std::vector<Point> out;
    int selected = 0;
    for (int i : candidates) {
        if (!score[i]) continue;
        if (selected++ == 96) break;
        int x = i % w, y = i / w;
        Point p{x, y, {}}; int k = 0; float sum = 0, norm = 0;
        for (int yy = y - 5; yy <= y + 5; yy += 2)
            for (int xx = x - 5; xx <= x + 5; xx += 2) sum += (p.descriptor[k++] = a[yy * w + xx]);
        for (auto &v : p.descriptor) { v -= sum / 36; norm += v * v; }
        if (norm > 100) {
            norm = std::sqrt(norm);
            for (auto &v : p.descriptor) v /= norm;
            out.push_back(p);
        }
        for (int yy = y - 4; yy <= y + 4; ++yy)
            for (int xx = x - 4; xx <= x + 4; ++xx) score[yy * w + xx] = 0;
    }
    return out;
}
// Refine a sparse corner match at nearby pixel positions. A corner detector
// need not pick the same physical location in both pictures. Compare a
// competing patch outside this neighbourhood to reject repeated detail.
inline bool patch(const std::vector<float> &image, int w, int h, int x, int y, Point &p) {
    if (x < 5 || y < 5 || x >= w - 5 || y >= h - 5) return false;
    p.x = x; p.y = y;
    float sum = 0, norm = 0; int k = 0;
    for (int yy = y - 5; yy <= y + 5; yy += 2)
        for (int xx = x - 5; xx <= x + 5; xx += 2) sum += (p.descriptor[k++] = image[yy * w + xx]);
    for (auto &v : p.descriptor) { v -= sum / 36; norm += v * v; }
    if (norm <= 100) return false;
    norm = std::sqrt(norm);
    for (auto &v : p.descriptor) v /= norm;
    return true;
}
inline float patch_distance(const Point &a, const Point &b) {
    float dot = 0;
    for (int k = 0; k < 36; ++k) dot += a.descriptor[k] * b.descriptor[k];
    return 1 - dot;
}
inline bool track(const Point &source, const std::vector<float> &target,
                  const std::vector<Point> &corners, int w, int h, Point prediction, Point &found) {
    std::vector<std::pair<float, Point>> candidates;
    candidates.reserve(corners.size() + 99);
    for (const auto &p : corners) candidates.emplace_back(patch_distance(source, p), p);
    if (patch(target, w, h, prediction.x, prediction.y, prediction))
        candidates.emplace_back(patch_distance(source, prediction), prediction);
    std::partial_sort(candidates.begin(), candidates.begin() + 2, candidates.end(),
                      [](const auto &a, const auto &b) { return a.first < b.first; });
    Point seeds[2] = {candidates[0].second, candidates[1].second};
    for (const auto &seed : seeds) for (int y = seed.y - 3; y <= seed.y + 3; ++y)
        for (int x = seed.x - 3; x <= seed.x + 3; ++x) {
            Point p{};
            if (patch(target, w, h, x, y, p)) candidates.emplace_back(patch_distance(source, p), p);
        }
    auto best = std::min_element(candidates.begin(), candidates.end(),
                                [](const auto &a, const auto &b) { return a.first < b.first; });
    if (best->first >= .18f) return false;
    found = best->second;
    float second = 2;
    for (const auto &c : candidates) {
        int dx = c.second.x - found.x, dy = c.second.y - found.y;
        if (dx * dx + dy * dy > 25) second = std::min(second, c.first);
    }
    return best->first < .65f * second;
}
// Feature correspondence is reserved for uncertain candidates. Weak-cut
// recovery needs distinct detail; overturning a strong cut needs more matching
// features. Ignore stationary matches: shared titles/borders are not continuity.
enum class Correspondence { Unknown, Distinct, Moving };
inline bool moving_matches(const std::vector<std::pair<Point, Point>> &matches, int required_matches) {
    auto inliers = [&](size_t anchor, double cs, double sn) {
        int count = 0; const auto &o = matches[anchor];
        for (const auto &m : matches) {
            double x = m.first.x - o.first.x, y = m.first.y - o.first.y;
            double dx = cs * x - sn * y + o.second.x - m.second.x;
            double dy = sn * x + cs * y + o.second.y - m.second.y;
            count += dx * dx + dy * dy < 9;
        }
        return count;
    };
    for (size_t i = 0; i < matches.size(); ++i) {
        if (inliers(i, 1, 0) >= required_matches) return true;
        for (size_t j = 0; j < i; ++j) {
            double x = matches[i].first.x - matches[j].first.x, y = matches[i].first.y - matches[j].first.y;
            double u = matches[i].second.x - matches[j].second.x, v = matches[i].second.y - matches[j].second.y;
            double d = x * x + y * y;
            if (d < 25) continue;
            double cs = (x * u + y * v) / d, sn = (x * v - y * u) / d, scale = cs * cs + sn * sn;
            if (scale > .25 && scale < 4 && inliers(i, cs, sn) >= required_matches) return true;
        }
    }
    return false;
}
inline bool regional_matches(const std::vector<std::pair<Point, Point>> &matches,
                             int radius) {
    // Two independently verified regions can explain deformation that a
    // single whole-picture similarity cannot. Each needs three non-collinear
    // reciprocal tracks, without sharing evidence with the other region.
    if (matches.size() < 6) return false;
    struct Region { double x, y, u, v; std::vector<size_t> tracks; };
    std::vector<Region> regions;
    for (size_t i = 0; i < matches.size(); ++i) for (size_t j = 0; j < i; ++j) {
        const auto &a = matches[i], &b = matches[j];
        double x = b.first.x - a.first.x, y = b.first.y - a.first.y;
        double u = b.second.x - a.second.x, v = b.second.y - a.second.y;
        double distance = x * x + y * y;
        if (distance < 25 || distance > radius * radius) continue;
        double cs = (x * u + y * v) / distance, sn = (x * v - y * u) / distance;
        double scale = cs * cs + sn * sn;
        if (scale <= .25 || scale >= 4) continue;
        Region region{};
        bool area = false;
        for (size_t k = 0; k < matches.size(); ++k) {
            const auto &m = matches[k];
            double px = m.first.x - a.first.x, py = m.first.y - a.first.y;
            if (px * px + py * py > radius * radius) continue;
            double dx = cs * px - sn * py + a.second.x - m.second.x;
            double dy = sn * px + cs * py + a.second.y - m.second.y;
            if (dx * dx + dy * dy >= 9) continue;
            region.tracks.push_back(k);
            region.x += m.first.x; region.y += m.first.y;
            region.u += m.second.x; region.v += m.second.y;
            // Similar patches along one straight edge are insufficient.
            double cross = x * py - y * px;
            area |= cross * cross > 6.25 * distance;
        }
        if (region.tracks.size() < 3 || !area) continue;
        region.x /= region.tracks.size(); region.y /= region.tracks.size();
        region.u /= region.tracks.size(); region.v /= region.tracks.size();
        for (const auto &other : regions) {
            double dx = region.x - other.x, dy = region.y - other.y;
            double du = region.u - other.u, dv = region.v - other.v;
            double separation = dx * dx + dy * dy;
            if (separation < 24 * 24 || du * du + dv * dv < 24 * 24) continue;
            // Independent parts may deform, but must retain their layout:
            // relative displacement stays below half their separation.
            if ((du - dx) * (du - dx) + (dv - dy) * (dv - dy) > .25 * separation) continue;
            bool shared = false;
            for (size_t k : region.tracks)
                shared |= std::find(other.tracks.begin(), other.tracks.end(), k) != other.tracks.end();
            if (!shared) return true;
        }
        regions.push_back(std::move(region));
    }
    return false;
}
inline Correspondence correspondence(const ajn_mvtools_frame &a, const ajn_mvtools_frame &b,
                                     int required_matches, bool finer = false, const VECTOR *vectors = nullptr) {
    int max_w = finer ? 240 : 120, max_h = finer ? 135 : 68;
    int w = std::min(max_w, a.width), h = std::max(1, std::min(a.height, (a.height * w + a.width / 2) / a.width));
    if (h > max_h) { w = std::max(1, w * max_h / h); h = max_h; }
    if (w < 13 || h < 13) return Correspondence::Unknown;
    auto ta = thumbnail(a, w, h), tb = thumbnail(b, w, h);
    auto pa = points(ta, tb, w, h), pb = points(tb, ta, w, h);
    if (pa.size() < 12 || pb.size() < 12) return Correspondence::Unknown;
    std::vector<float> distance(pa.size() * pb.size());
    for (size_t i = 0; i < pa.size(); ++i) for (size_t j = 0; j < pb.size(); ++j) {
        float dot = 0;
        for (int k = 0; k < 36; ++k) dot += pa[i].descriptor[k] * pb[j].descriptor[k];
        distance[i * pb.size() + j] = 1 - dot;
    }
    std::vector<std::pair<Point, Point>> matches;
    for (size_t i = 0; i < pa.size(); ++i) {
        float best = 2, second = 2; size_t at = 0;
        for (size_t j = 0; j < pb.size(); ++j) {
            float d = distance[i * pb.size() + j];
            if (d < best) { second = best; best = d; at = j; }
            else second = std::min(second, d);
        }
        if (best >= .18f || best >= .65f * second) continue;
        bool mutual = true;
        for (size_t k = 0; k < pa.size(); ++k) if (distance[k * pb.size() + at] < best) { mutual = false; break; }
        int dx = pb[at].x - pa[i].x, dy = pb[at].y - pa[i].y;
        if (mutual && dx * dx + dy * dy > 6.25) matches.emplace_back(pa[i], pb[at]);
    }
    if (moving_matches(matches, required_matches)) return Correspondence::Moving;
    if (vectors) {
        // Only the fine, uncertain-motion path performs dense patch tracking.
        // Keep exact corner matches above; require reciprocal tracks for new
        // evidence so a one-way match on a flat edge cannot establish motion.
        matches.clear();
        int nx = a.width / 8, ny = a.height / 8;
        for (const auto &p : pa) {
            int x = p.x * a.width / w, y = p.y * a.height / h;
            const auto &v = vectors[std::min(y / 8, ny - 1) * nx + std::min(x / 8, nx - 1)];
            Point predicted{p.x + v.x * w / a.width, p.y + v.y * h / a.height, {}}, to{}, back{};
            if (!track(p, tb, pb, w, h, predicted, to)) continue;
            int dx = to.x - p.x, dy = to.y - p.y;
            if (dx * dx + dy * dy <= 6.25) continue;
            if (!track(to, ta, pa, w, h, p, back)) continue;
            dx = back.x - p.x; dy = back.y - p.y;
            if (dx * dx + dy * dy <= 4) matches.emplace_back(p, to);
        }
        if (moving_matches(matches, required_matches)) return Correspondence::Moving;
        // A deforming face can have several local motions. Use only reciprocal
        // tracks, with non-collinear neighbours confirming each local model.
        // The caller still requires aligned, ongoing motion before this path.
        if (regional_matches(matches, std::min(w, h) / 3))
            return Correspondence::Moving;
    }
    return Correspondence::Distinct;
}
}
