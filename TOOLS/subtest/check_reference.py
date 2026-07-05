#!/usr/bin/env python3
"""
check_reference.py -- self-test / verification gate for TOOLS/subtest.

Runs parse_stats against the corpus of mpv --dump-stats captures and compares
the computed video-draw statistics against the reference table taken from the
project postmortem (which was derived from these exact files).

Tolerances (per the work package):
    frames    -- must match EXACTLY
    ms values -- within +-0.15 ms
    %over     -- within +-0.1 percentage points

Also checks two worst-stall figures:
    stats_composite_async  video-draw max ~= 8614.8 ms
    ep09stats3             video-draw max ~= 972    ms

Exit code 0 iff every row passes.

Python 3.12 stdlib only.
"""

from __future__ import annotations

import argparse
import os
import sys

import parse_stats

DEFAULT_CORPUS = "/mnt/y/Video/kobayashi"
BUDGET = 41.7
EVENT = "video-draw"

MS_TOL = 0.15
PCT_TOL = 0.1

# filename (without .txt) -> (frames, mean, p50, p95, p99, max, %over)
REFERENCE = {
    "stats_subsoff":          (720, 0.9, 0.8, 1.2, 1.4, 14.6, 0.0),
    "stats_off":              (664, 20.1, 14.9, 69.6, 78.3, 189.4, 14.0),
    "stats_on":               (679, 18.4, 15.1, 61.4, 68.0, 149.6, 15.3),
    "stats_noblur":           (684, 16.3, 9.3, 60.8, 71.4, 135.8, 13.5),
    "stats_composite_on":     (645, 11.2, 7.6, 28.4, 95.1, 438.6, 2.6),
    "stats_composite_off":    (720, 6.1, 4.6, 14.8, 35.4, 87.1, 0.7),
    "stats_ra":               (678, 16.2, 7.1, 59.9, 82.6, 144.7, 13.9),
    "stats_ra5":              (699, 8.6, 4.1, 43.0, 74.3, 168.4, 5.2),
    "stats_strided_ra":       (714, 10.9, 4.2, 19.5, 61.0, 1372.2, 1.5),
    "stats_eizouken_raster":  (461, 16.8, 17.2, 33.7, 39.0, 57.0, 0.9),
    "stats_eizouken_batched": (479, 11.5, 12.4, 19.0, 21.7, 253.9, 0.2),
    "stats_gpuraster":        (681, 5.9, 4.2, 15.6, 18.5, 60.6, 0.15),
    "stats_gpuraster_binned": (720, 6.9, 4.3, 16.4, 18.6, 519.2, 0.14),
}

# filename (without .txt) -> approx expected video-draw max (ms), abs tol
WORST_STALL = {
    "stats_composite_async": (8614.8, 0.15),
    "ep09stats3":            (972.0, 1.5),   # postmortem quotes "~972"
}


def check_row(name: str, corpus: str) -> tuple[bool, str]:
    path = os.path.join(corpus, name + ".txt")
    if not os.path.exists(path):
        return False, f"{name:<24} MISSING FILE: {path}"

    st = parse_stats.parse_stats_file(path)
    rep = parse_stats.summarize(EVENT, st.durations.get(EVENT, []), BUDGET)
    ref = REFERENCE[name]

    got = (rep.frames, rep.mean, rep.p50, rep.p95, rep.p99, rep.maximum, rep.pct_over)
    labels = ["frames", "mean", "p50", "p95", "p99", "max", "%over"]
    tols = [0, MS_TOL, MS_TOL, MS_TOL, MS_TOL, MS_TOL, PCT_TOL]

    diffs = []
    ok = True
    for i, (g, r, tol, lab) in enumerate(zip(got, ref, tols, labels)):
        if i == 0:
            passed = g == r
        else:
            passed = abs(g - r) <= tol + 1e-9
        if not passed:
            ok = False
            diffs.append(f"{lab}: got {g} want {r}")

    status = "PASS" if ok else "FAIL"
    line = (
        f"{name:<24} {status}  "
        f"fr={rep.frames:<4} mean={rep.mean:5.1f} p50={rep.p50:5.1f} "
        f"p95={rep.p95:6.1f} p99={rep.p99:6.1f} max={rep.maximum:8.1f} "
        f"%over={rep.pct_over:5.2f}"
    )
    if diffs:
        line += "  <-- " + "; ".join(diffs)
    return ok, line


def check_stall(name: str, corpus: str) -> tuple[bool, str]:
    path = os.path.join(corpus, name + ".txt")
    if not os.path.exists(path):
        return False, f"{name:<24} MISSING FILE: {path}"
    st = parse_stats.parse_stats_file(path)
    durs = st.durations.get(EVENT, [])
    if not durs:
        return False, f"{name:<24} FAIL  no {EVENT} pairs"
    mx = max(durs)
    want, tol = WORST_STALL[name]
    ok = abs(mx - want) <= tol
    status = "PASS" if ok else "FAIL"
    line = f"{name:<24} {status}  max={mx:.1f} ms (expect ~{want} +-{tol})"
    return ok, line


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS,
                    help=f"directory holding the stats_*.txt corpus (default {DEFAULT_CORPUS})")
    args = ap.parse_args(argv)

    print(f"Corpus: {args.corpus}")
    print(f"Event: {EVENT}   budget: {BUDGET} ms   "
          f"percentile: numpy-nearest round((n-1)*q)")
    print()
    print("=== reference table ===")

    all_ok = True
    for name in REFERENCE:
        ok, line = check_row(name, args.corpus)
        all_ok = all_ok and ok
        print(line)

    print()
    print("=== worst-stall spot checks ===")
    for name in WORST_STALL:
        ok, line = check_stall(name, args.corpus)
        all_ok = all_ok and ok
        print(line)

    print()
    if all_ok:
        print("RESULT: ALL PASS")
        return 0
    print("RESULT: FAILURES PRESENT")
    return 1


if __name__ == "__main__":
    sys.exit(main())
