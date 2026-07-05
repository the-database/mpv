#!/usr/bin/env python3
"""
check_acceptance.py -- adjudicate an 8K zero-drop subtitle acceptance run.

Thin wrapper over parse_stats.py (imported from the same directory). Given a
results directory (or one/more --dump-stats files) it certifies every scene
capture against the WP-G1 acceptance bar and emits a machine-readable
verdict.json plus a human-readable table.

ACCEPTANCE BAR (a scene PASSES only if all three hold):

  1. ZERO video-draw events longer than the per-frame budget.
     The budget is auto-derived from the file's own frame-duration value
     samples (frame-duration-approx preferred, then frame-duration); if the
     capture carries none, it falls back to 41.7 ms (23.976 fps).

  2. ZERO mpv frame drops -- counted as "drop-*" singular markers
     (the corpus/mpv emits a bare "drop-vo" line per dropped frame).

  3. ALL of these integrity counters == 0:
       gcache-flush atlas-overflow staging-grow overlay-buf-grow tex-realloc
       vo-alloc-after-first-frame ra-miss ra-inline ra-stale stale-present
     mpv emits these with emit_counter() (vo_gpu_next.c) / MP_STATS value lines
     (sub_ahead.c). vo_gpu_next's emit_counter is *emit-on-change* starting from
     0, so a counter that never left 0 for the whole run produces NO line at
     all. Therefore an ABSENT counter means "never fired" == 0 == GOOD; a
     PRESENT counter's last sample is its final cumulative value and must be 0.
     (This is also why the pre-instrumentation corpus captures -- which carry
     none of these counters -- pass the counter check and fail only on bars 1/2.)

OPTIONAL screenshot A/B (correctness spot-check, not part of the numeric bar):
  If <results>/shots/<scene>_acc_<pts>.png and <scene>_cpu_<pts>.png pairs
  exist, they are pixel-diffed. Exact parity is NOT expected on real hardware
  for blur frames; the gate is only for GROSS divergence (the sign-bleed class
  of bug), default > 16/255. The tight <=3/255 lavapipe bar is reported for
  information but not enforced here.

Python 3 stdlib only. Windows: run with the "py -3" launcher.
"""

from __future__ import annotations

import argparse
import array
import datetime
import json
import os
import struct
import sys
import zlib

# Import the sibling parser (works regardless of CWD / how py -3 was invoked).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parse_stats  # noqa: E402


DEFAULT_BUDGET_MS = 41.7

# The 10 integrity counters that must be 0 to certify a run.
COUNTERS = [
    "gcache-flush",
    "atlas-overflow",
    "staging-grow",
    "overlay-buf-grow",
    "tex-realloc",
    "vo-alloc-after-first-frame",
    "ra-miss",
    "ra-inline",
    "ra-stale",
    "stale-present",
]

# The mis-linked-build tell: stock (non-fork) libass makes mpv warn-and-ignore
# the GPU sub options. If this shows up in a scene log the capture is invalid.
STOCK_LIBASS_WARN = "built without forked-libass deferred API; ignoring"


# --------------------------------------------------------------------------
# budget / drops / counters
# --------------------------------------------------------------------------

def detect_budget(st: parse_stats.Stats) -> tuple[float, float, str]:
    """Return (budget_ms, fps, source). Prefer frame-duration-approx, then
    frame-duration, else the 41.7 ms default. frame-duration* samples are in
    seconds in the corpus; a value < 1 is treated as seconds, else as ms."""
    for name in ("frame-duration-approx", "frame-duration"):
        samples = st.values.get(name)
        if samples:
            vals = sorted(v for _ts, v in samples)
            med = vals[len(vals) // 2]
            if med <= 0:
                continue
            budget_ms = med * 1000.0 if med < 1.0 else med
            if 1.0 <= budget_ms <= 1000.0:
                return budget_ms, 1000.0 / budget_ms, name
    return DEFAULT_BUDGET_MS, 1000.0 / DEFAULT_BUDGET_MS, "default(41.7)"


def count_drops(st: parse_stats.Stats) -> tuple[int, dict[str, int]]:
    """Total frame-drop markers and their per-name breakdown (any drop-* )."""
    breakdown = {name: n for name, n in st.signals.items()
                 if name.startswith("drop")}
    return sum(breakdown.values()), breakdown


def read_counters(st: parse_stats.Stats) -> dict[str, float]:
    """counter name -> final value (last value sample; 0.0 if absent)."""
    out = {}
    for name in COUNTERS:
        samples = st.values.get(name)
        out[name] = samples[-1][1] if samples else 0.0
    return out


def scan_log(path: str) -> dict:
    """Scan a sibling mpv log for the stock-libass warning."""
    info = {"path": path, "stock_libass_warn": False, "warn_line": ""}
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                if STOCK_LIBASS_WARN in raw:
                    info["stock_libass_warn"] = True
                    info["warn_line"] = raw.strip()
                    break
    except OSError:
        return None
    return info


# --------------------------------------------------------------------------
# adjudication
# --------------------------------------------------------------------------

def adjudicate(stats_path: str, default_budget: float,
               logs_dir: str | None = None) -> dict:
    st = parse_stats.parse_stats_file(stats_path)
    budget, fps, budget_src = detect_budget(st)
    if budget_src == "default(41.7)":
        budget = default_budget
        fps = 1000.0 / budget

    rep = parse_stats.summarize("video-draw", st.durations.get("video-draw", []),
                                budget)
    drops, drop_breakdown = count_drops(st)
    counters = read_counters(st)

    reasons = []
    if rep.frames == 0:
        reasons.append("no video-draw start/end pairs (broken/empty capture)")
    if rep.over > 0:
        reasons.append(
            f"video-draw: {rep.over} frame(s) exceeded the "
            f"{budget:.2f} ms budget (max {rep.maximum:.1f} ms)")
    if drops > 0:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(drop_breakdown.items()))
        reasons.append(f"frame drops: {drops} ({detail})")
    fired = {k: v for k, v in counters.items() if round(v) != 0}
    if fired:
        detail = ", ".join(f"{k}={int(round(v))}" for k, v in sorted(fired.items()))
        reasons.append(f"integrity counters fired: {detail}")

    # log cross-check (optional)
    log_info = None
    if logs_dir:
        scene = scene_name(stats_path)
        cand = os.path.join(logs_dir, f"mpv_{scene}.log")
        if os.path.isfile(cand):
            log_info = scan_log(cand)
            if log_info and log_info["stock_libass_warn"]:
                reasons.append(
                    "STOCK-LIBASS build detected in log -- GPU sub options were "
                    "warned-and-ignored; this capture does NOT exercise the fork")

    verdict = "PASS" if not reasons else "FAIL"
    return {
        "file": stats_path,
        "scene": scene_name(stats_path),
        "fps": round(fps, 3),
        "budget_ms": round(budget, 3),
        "budget_source": budget_src,
        "video_draw": {
            "frames": rep.frames,
            "mean": round(rep.mean, 1),
            "p50": round(rep.p50, 1),
            "p95": round(rep.p95, 1),
            "p99": round(rep.p99, 1),
            "max": round(rep.maximum, 1),
            "over": rep.over,
            "pct_over": round(rep.pct_over, 3),
        },
        "frame_drops": drops,
        "drop_breakdown": drop_breakdown,
        "counters": {k: int(round(v)) for k, v in counters.items()},
        "log": log_info,
        "verdict": verdict,
        "reasons": reasons,
        "_report": rep,  # kept for printing; stripped before JSON
    }


def scene_name(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith(".txt"):
        base = base[:-4]
    if base.startswith("stats_"):
        base = base[len("stats_"):]
    return base


# --------------------------------------------------------------------------
# screenshot A/B (optional, ported from abdiff.py's proven zlib PNG reader)
# --------------------------------------------------------------------------

_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}


class _Img:
    __slots__ = ("w", "h", "channels", "bitdepth", "maxval", "raw")

    def __init__(self, w, h, channels, bitdepth, raw):
        self.w, self.h, self.channels, self.bitdepth = w, h, channels, bitdepth
        self.maxval = (1 << bitdepth) - 1
        self.raw = raw

    @property
    def row_bytes(self):
        return self.w * self.channels * (self.bitdepth // 8)

    def row_samples(self, y):
        rb = self.row_bytes
        chunk = self.raw[y * rb:(y + 1) * rb]
        if self.bitdepth == 8:
            return array.array("B", chunk)
        a = array.array("H")
        a.frombytes(chunk)
        if sys.byteorder == "little":
            a.byteswap()
        return a


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def read_png(path):
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:8] != _PNG_SIG:
        raise ValueError(f"{path}: not a PNG")
    pos = 8
    width = height = bitdepth = colortype = interlace = None
    idat = bytearray()
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            (width, height, bitdepth, colortype, _c, _f, interlace) = \
                struct.unpack(">IIBBBBB", body)
        elif ctype == b"IDAT":
            idat += body
        elif ctype == b"IEND":
            break
    if width is None:
        raise ValueError(f"{path}: no IHDR")
    if interlace != 0:
        raise ValueError(f"{path}: interlaced not supported")
    if colortype not in _CHANNELS or colortype == 3:
        raise ValueError(f"{path}: unsupported colour type {colortype}")
    if bitdepth not in (8, 16):
        raise ValueError(f"{path}: unsupported bit depth {bitdepth}")
    channels = _CHANNELS[colortype]
    raw = zlib.decompress(bytes(idat))
    bpp = channels * (bitdepth // 8)
    stride = width * bpp
    out = bytearray(height * stride)
    src = 0
    prev = bytearray(stride)
    for y in range(height):
        ftype = raw[src]
        src += 1
        line = bytearray(raw[src:src + stride])
        src += stride
        if ftype == 0:
            pass
        elif ftype == 1:
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 0xFF
        elif ftype == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ftype == 3:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                c = prev[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + _paeth(a, prev[i], c)) & 0xFF
        else:
            raise ValueError(f"{path}: bad filter {ftype} row {y}")
        out[y * stride:(y + 1) * stride] = line
        prev = line
    return _Img(width, height, channels, bitdepth, bytes(out))


def diff_png(pa, pb):
    """Return maxdiff normalised to /255, or raise on decode/geometry error."""
    a, b = read_png(pa), read_png(pb)
    if (a.w, a.h, a.channels, a.bitdepth) != (b.w, b.h, b.channels, b.bitdepth):
        raise ValueError(
            f"geometry mismatch {a.w}x{a.h}c{a.channels}b{a.bitdepth} vs "
            f"{b.w}x{b.h}c{b.channels}b{b.bitdepth}")
    if a.raw == b.raw:
        return 0.0
    ch, rb, maxdiff = a.channels, a.row_bytes, 0
    for y in range(a.h):
        if a.raw[y * rb:(y + 1) * rb] == b.raw[y * rb:(y + 1) * rb]:
            continue
        sa, sb = a.row_samples(y), b.row_samples(y)
        for i in range(len(sa)):
            d = sa[i] - sb[i]
            if d < 0:
                d = -d
            if d > maxdiff:
                maxdiff = d
    return maxdiff * 255.0 / a.maxval


def adjudicate_shots(shots_dir: str, gross8: float) -> list[dict]:
    """Diff every <scene>_acc_<pts>.png / <scene>_cpu_<pts>.png pair."""
    if not os.path.isdir(shots_dir):
        return []
    accs = {}
    for fn in sorted(os.listdir(shots_dir)):
        if fn.endswith(".png") and "_acc_" in fn:
            accs[fn] = fn.replace("_acc_", "_cpu_")
    out = []
    for acc, cpu in accs.items():
        pa = os.path.join(shots_dir, acc)
        pb = os.path.join(shots_dir, cpu)
        rec = {"acc": acc, "cpu": cpu}
        if not os.path.isfile(pb):
            rec.update(status="MISSING", note="no matching cpu-baseline shot")
        else:
            try:
                md = diff_png(pa, pb)
                rec["maxdiff8"] = round(md, 2)
                rec["tight_bar_ok"] = md <= 3.0
                rec["status"] = "GROSS" if md > gross8 else "ok"
            except Exception as e:  # decode / geometry -> compare manually
                rec.update(status="ERROR", note=str(e))
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------

def print_scene(r: dict) -> None:
    rep = r["_report"]
    print(f"\n=== SCENE: {r['scene']}   ({os.path.basename(r['file'])}) ===")
    mins = rep.frames * r["budget_ms"] / 1000.0 / 60.0
    print(f"fps={r['fps']}  budget={r['budget_ms']:.2f} ms  "
          f"[{r['budget_source']}]  run={rep.frames} frames (~{mins:.1f} min "
          f"of decoded video)")
    hdr = (f"{'event':<12} {'frames':>6} {'mean':>7} {'p50':>7} {'p95':>7} "
           f"{'p99':>7} {'max':>9} {'over':>6} {'%over':>7}")
    print(hdr)
    print("-" * len(hdr))
    print(f"{'video-draw':<12} {rep.frames:>6} {rep.mean:>7.1f} {rep.p50:>7.1f} "
          f"{rep.p95:>7.1f} {rep.p99:>7.1f} {rep.maximum:>9.1f} {rep.over:>6} "
          f"{rep.pct_over:>6.2f}%")

    d = r["frame_drops"]
    dtxt = "0" if d == 0 else \
        f"{d}  (" + ", ".join(f"{k}={v}" for k, v in sorted(r['drop_breakdown'].items())) + ")"
    print(f"\nframe drops: {dtxt}")

    print(f"\n{'counter':<28} {'value':>7}   status")
    print("-" * 48)
    for name in COUNTERS:
        v = r["counters"][name]
        status = "ok" if v == 0 else "FIRED"
        note = "  (never fired)" if v == 0 else ""
        print(f"{name:<28} {v:>7}   {status}{note}")

    if r.get("log") and r["log"].get("stock_libass_warn"):
        print(f"\n!! STOCK-LIBASS WARNING in log: {r['log']['warn_line']}")

    print(f"\nVERDICT: {r['verdict']}")
    for reason in r["reasons"]:
        print(f"  - {reason}")


def print_shots(shots: list[dict], gross8: float) -> None:
    if not shots:
        return
    print(f"\n=== SCREENSHOT A/B (gross-divergence gate > {gross8:.0f}/255; "
          f"tight bar <=3/255 informational) ===")
    hdr = f"{'acc shot':<34} {'maxdiff/255':>11}  {'<=3':>4}  status"
    print(hdr)
    print("-" * len(hdr))
    for s in shots:
        if s["status"] in ("MISSING", "ERROR"):
            print(f"{s['acc']:<34} {'--':>11}  {'--':>4}  {s['status']}: {s.get('note','')}")
        else:
            tight = "yes" if s["tight_bar_ok"] else "no"
            print(f"{s['acc']:<34} {s['maxdiff8']:>11.2f}  {tight:>4}  {s['status']}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def gather_stats_files(paths: list[str]) -> tuple[list[str], str | None]:
    """Expand args into stats files. Returns (files, results_dir_if_single)."""
    files: list[str] = []
    results_dir = None
    for p in paths:
        if os.path.isdir(p):
            results_dir = p
            found = sorted(fn for fn in os.listdir(p)
                           if fn.startswith("stats_") and fn.endswith(".txt"))
            if not found:  # fall back to any non-log .txt
                found = sorted(fn for fn in os.listdir(p)
                               if fn.endswith(".txt") and "log" not in fn.lower())
            files += [os.path.join(p, fn) for fn in found]
        else:
            files.append(p)
    return files, results_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Adjudicate 8K zero-drop subtitle acceptance captures.")
    ap.add_argument("paths", nargs="+",
                    help="a results directory and/or individual stats files")
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET_MS,
                    help="fallback budget ms when a capture has no "
                         "frame-duration value (default 41.7)")
    ap.add_argument("--gross", type=float, default=16.0,
                    help="screenshot gross-divergence gate, /255 (default 16)")
    ap.add_argument("--json", help="path for verdict.json "
                    "(default: <results-dir>/verdict.json, else ./verdict.json)")
    ap.add_argument("--no-shots", action="store_true",
                    help="skip the screenshot A/B even if shots/ exists")
    args = ap.parse_args(argv)

    files, results_dir = gather_stats_files(args.paths)
    if not files:
        print("error: no stats files found", file=sys.stderr)
        return 2

    logs_dir = results_dir  # sibling mpv_<scene>.log lives next to the stats
    results = [adjudicate(f, args.budget, logs_dir) for f in files]
    for r in results:
        print_scene(r)

    shots = []
    if not args.no_shots and results_dir:
        shots = adjudicate_shots(os.path.join(results_dir, "shots"), args.gross)
        print_shots(shots, args.gross)

    n_fail = sum(1 for r in results if r["verdict"] != "PASS")
    n_gross = sum(1 for s in shots if s["status"] == "GROSS")
    overall = "PASS" if (n_fail == 0 and n_gross == 0) else "FAIL"

    print("\n" + "=" * 60)
    print("SUMMARY")
    for r in results:
        print(f"  {r['scene']:<20} {r['verdict']}")
    if shots:
        gs = [s for s in shots if s["status"] == "GROSS"]
        print(f"  screenshots: {len(shots)} pair(s), "
              f"{len(gs)} gross-divergent")
    print(f"OVERALL: {overall}   ({len(results)} scene(s), {n_fail} failing)")
    print("=" * 60)

    # verdict.json
    out_json = args.json
    if not out_json:
        out_json = os.path.join(results_dir, "verdict.json") if results_dir \
            else "verdict.json"
    payload = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "criteria": {
            "video_draw_over_budget": 0,
            "frame_drops": 0,
            "counters_all_zero": COUNTERS,
            "screenshot_gross_gate_over_255": args.gross,
        },
        "overall": overall,
        "scenes": [{k: v for k, v in r.items() if k != "_report"}
                   for r in results],
        "screenshots": shots,
    }
    try:
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {out_json}")
    except OSError as e:
        print(f"warning: could not write {out_json}: {e}", file=sys.stderr)

    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
