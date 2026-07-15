#!/usr/bin/env python3
"""temporal_ab.py -- frame-step A/B temporal gate (WP-H7 deliverable 3).

The frame-SAMPLED abdiff.py proves per-PTS equality but can miss TEMPORAL
defects: content that renders correctly at the sampled instants yet corrupts
or vanishes on frames in between (round-4 field defect: a sign losing its
outline, then vanishing entirely, mid-lifetime -- state-dependent, so no
sampled PTS caught it). This tool frame-steps two configs across a window,
screenshots EVERY frame via the bundled temporal_step.lua (no IPC -- the same
driver works for a local Linux mpv and for a Windows mpv.exe launched through
WSL interop), and diffs the runs frame-by-frame:

  * zero-divergence gate: every frame byte-identical (or within --tol8 for
    blur-carrying content),
  * vanish detector: any frame where the A/B differing-pixel area jumps past
    --vanish-pct of the frame is flagged VANISH (one config lost content the
    other still renders),
  * forensics: per-run --dump-stats counters are snapshotted per captured
    frame (the lua logs a timestamped tstep-shot line; the stats dump shares
    that clock), so the first divergent frame comes annotated with exactly
    which counters fired in the interval that produced it.

Usage (local, lavapipe):
  python3 temporal_ab.py --mpv ../../build/mpv \
      --media /path/ep7.mkv --start 246.0 --frames 122 \
      --config-a /path/mpv-cpu-baseline.conf \
      --config-b /path/mpv-acceptance.conf \
      --geometry 7680x4320 --out out_t --tag unfazed --runs 3

Rig (Windows mpv.exe from WSL; paths auto-translated /mnt/X/.. -> X:/..):
  python3 temporal_ab.py --mpv /mnt/c/Users/jsoos/8k-rig/mpv/mpv.exe \
      --media "/mnt/y/Video/kobayashi/....mkv" --start 246.0 --frames 122 \
      --config-a /mnt/y/Video/kobayashi/8k-acceptance-kit/mpv-cpu-baseline.conf \
      --config-b /mnt/y/Video/kobayashi/8k-acceptance-kit/mpv-acceptance.conf \
      --geometry 7680x4320 --out out_rig --tag unfazed_rig \
      --win-scratch /mnt/c/Users/jsoos/8k-rig/out
  (rig runs open a real window on the desktop; keep them short/batched)

Python 3.12, stdlib only. PNG decode + diff come from abdiff.py (same dir).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abdiff import read_png, write_diff_png, Image  # noqa: E402

LUA_NAME = "temporal_step.lua"

# Determinism / window pinning shared by every run (abdiff.py conventions,
# minus geometry/gpu-sw which are parameterized, minus high-bit-depth: the
# temporal gate works at 8-bit -- a vanished sign is thousands of LSBs, and
# 8K x 16-bit x every-frame would be tens of GB per run).
BASE_FLAGS = [
    "--no-config",
    "--vo=gpu-next",
    "--gpu-api=vulkan",
    "--no-audio",
    "--window-scale=1",
    "--no-border",
    "--no-osc",
    "--no-osd-bar",
    "--osd-level=0",
    "--no-input-default-bindings",
    "--force-window=yes",
    "--keep-open=yes",
    "--pause",
    "--dither=no",
    "--deband=no",
    "--icc-profile-auto=no",
    "--hr-seek=yes",
    "--sub-auto=no",
    "--blend-subtitles=no",
    "--screenshot-format=png",
    "--screenshot-png-compression=1",
    "--screenshot-png-filter=0",
    "--screenshot-high-bit-depth=no",
    "--screenshot-tag-colorspace=no",
]

# Counters worth snapshotting for temporal forensics (last cumulative value).
COUNTERS = [
    "compose-reuse", "result-spill", "raster-pool-pregrow", "raster-pool-grow",
    "gcache-overcommit", "glyphs-uncached", "gcache-epoch-advance",
    "stale-present", "guard-empty", "guard-first-late",
    "ra-miss", "ra-stale", "ra-inline", "ra-fetch-stall",
    "staging-wrap", "blob-hash-hit", "tex-realloc", "vo-alloc-after-first-frame",
]


def is_windows_binary(path: str) -> bool:
    return path.lower().endswith(".exe")


def to_win_path(p: str) -> str:
    """Translate a /mnt/<d>/... WSL path to <D>:/... (forward slashes: mpv on
    Windows accepts them everywhere, and they survive --script-opts)."""
    m = re.match(r"^/mnt/([a-zA-Z])(/.*)?$", p)
    if not m:
        raise ValueError(f"cannot translate to a Windows path: {p}")
    drive = m.group(1).upper()
    rest = m.group(2) or "/"
    return f"{drive}:{rest}"


def parse_config(arg: str | None) -> list[str]:
    """A .conf path becomes --include=...; anything else splits as raw opts."""
    if not arg:
        return []
    if arg.endswith(".conf") and os.path.exists(arg):
        return [f"--include={arg}"]
    return arg.split()


def translate_flags_for_win(flags: list[str]) -> list[str]:
    out = []
    for f in flags:
        if f.startswith("--include=/mnt/"):
            out.append("--include=" + to_win_path(f[len("--include="):]))
        else:
            out.append(f)
    return out


class StatsSnapshot:
    """Parse a --dump-stats file into (ts_ns, name, value) counter samples and
    provide 'cumulative value as of time T' lookups for the forensic table."""

    def __init__(self, path: str):
        self.samples: dict[str, list[tuple[int, float]]] = {}
        if not os.path.exists(path):
            return
        with open(path, "r", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 4 and parts[1] == "value":
                    try:
                        ts = int(parts[0]); v = float(parts[2])
                    except ValueError:
                        continue
                    self.samples.setdefault(parts[3], []).append((ts, v))

    def last(self, name: str):
        s = self.samples.get(name)
        return s[-1][1] if s else None

    def value_at(self, name: str, ts_ns: int):
        s = self.samples.get(name)
        if not s:
            return None
        v = None
        for t, val in s:
            if t <= ts_ns:
                v = val
            else:
                break
        return v


def parse_shot_times(log_path: str) -> dict[int, float]:
    """Map captured frame idx -> mpv log timestamp (seconds, mp_time clock)."""
    out: dict[int, float] = {}
    if not os.path.exists(log_path):
        return out
    pat = re.compile(r"\[\s*(\d+\.\d+)\]\[.\].*tstep-shot idx=(\d+)")
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                out[int(m.group(2))] = float(m.group(1))
    return out


def run_one(mpv: str, media: str, cfg_flags: list[str], run_dir: str,
            args, win: bool) -> dict:
    """Launch one frame-step capture run. Returns paths + parsed artifacts."""
    os.makedirs(run_dir, exist_ok=True)
    shots_dir = os.path.join(run_dir, "shots")
    os.makedirs(shots_dir, exist_ok=True)
    stats_path = os.path.join(run_dir, "stats.txt")
    log_path = os.path.join(run_dir, "mpv.log")
    for p in (stats_path, log_path, os.path.join(shots_dir, "DONE")):
        if os.path.exists(p):
            os.unlink(p)

    if win:
        lua = os.path.join(run_dir, LUA_NAME)
        shutil.copyfile(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     LUA_NAME), lua)
        media_arg = to_win_path(media) if media.startswith("/mnt/") else media
        script_arg = to_win_path(lua)
        out_arg = to_win_path(shots_dir)
        stats_arg = to_win_path(stats_path)
        log_arg = to_win_path(log_path)
        cfg_flags = translate_flags_for_win(cfg_flags)
    else:
        media_arg = media
        script_arg = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  LUA_NAME)
        out_arg = shots_dir
        stats_arg = stats_path
        log_arg = log_path

    gpu_sw = args.gpu_sw
    if gpu_sw == "auto":
        gpu_sw = "no" if win else "yes"

    # cfg_flags first, then the pinned run controls: a kit conf sets e.g. `fs`,
    # and the temporal gate needs the window at exactly --geometry (the window
    # screenshot size) regardless of what the conf says.
    cmd = [mpv, media_arg] + cfg_flags + BASE_FLAGS + [
        f"--gpu-sw={gpu_sw}",
        "--fs=no",
        f"--geometry={args.geometry}",
        f"--script={script_arg}",
        ("--script-opts=" +
         f"tstep-start={args.start},tstep-frames={args.frames}," +
         f"tstep-out={out_arg},tstep-settle={args.settle}," +
         f"tstep-settle0={args.settle0}"),
        f"--dump-stats={stats_arg}",
        f"--log-file={log_arg}",
        "-v",
    ]
    timeout = args.run_timeout or (120 + args.frames * args.per_frame_timeout)
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, timeout=timeout)
    dt = time.time() - t0
    done = os.path.join(shots_dir, "DONE")
    frames = sorted(f for f in os.listdir(shots_dir)
                    if re.fullmatch(r"f\d{5}\.png", f))
    return {
        "dir": run_dir, "shots": shots_dir, "frames": frames,
        "complete": os.path.exists(done), "wall_s": round(dt, 1),
        "rc": proc.returncode,
        "stats": stats_path, "log": log_path,
    }


def rows_differing(a: Image, b: Image) -> int:
    """Number of pixel rows that are not byte-identical (C-speed slicing)."""
    rb = a.row_bytes
    n = 0
    for y in range(a.h):
        if a.raw[y * rb:(y + 1) * rb] != b.raw[y * rb:(y + 1) * rb]:
            n += 1
    return n


def sampled_diff(a: Image, b: Image, gross_tol: int, stride: int = 4):
    """maxdiff + differing-pixel estimates on a strided grid of the differing
    rows only. Exactness is decided by full row equality (rows_differing==0);
    this is the QUANTIFIER for frames already known to differ -- full
    per-pixel stats on an 8K frame in pure Python would take minutes.

    Returns (maxdiff, ndiff_any, ndiff_gross): ndiff_gross counts only pixels
    differing by more than gross_tol (native units) -- the VANISH metric must
    not trip on the legitimate small blur residual spread across a big sign
    (the m5 bar allows maxdiff8 <= 3 on blur-carrying content)."""
    ch = a.channels
    rb = a.row_bytes
    maxd = 0
    nd = 0
    nd_gross = 0
    for y in range(0, a.h, stride):
        ra = a.raw[y * rb:(y + 1) * rb]
        rbb = b.raw[y * rb:(y + 1) * rb]
        if ra == rbb:
            continue
        sa = a.row_samples(y)
        sb = b.row_samples(y)
        for px in range(0, a.w, stride):
            base = px * ch
            pd = 0
            for k in range(ch):
                d = sa[base + k] - sb[base + k]
                if d < 0:
                    d = -d
                if d > pd:
                    pd = d
            if pd:
                nd += 1
                if pd > gross_tol:
                    nd_gross += 1
                if pd > maxd:
                    maxd = pd
    scale = stride * stride
    return maxd, nd * scale, nd_gross * scale  # estimates over the full frame


def compare_runs(ra: dict, rb: dict, args, tag: str, out_dir: str) -> dict:
    """Frame-by-frame diff of two capture runs -> verdict dict."""
    n = min(len(ra["frames"]), len(rb["frames"]))
    frames = []
    first_div = None
    vanish_frames = []
    sa = StatsSnapshot(rb["stats"])
    shot_t = parse_shot_times(rb["log"])
    total_px = None
    for i in range(n):
        fa = os.path.join(ra["shots"], ra["frames"][i])
        fb = os.path.join(rb["shots"], rb["frames"][i])
        ia = read_png(fa)
        ib = read_png(fb)
        if total_px is None:
            total_px = ia.w * ia.h
        if ia.raw == ib.raw:
            rec = {"i": i, "exact": True, "maxdiff8": 0, "ndiff_est": 0}
            if not args.keep_all:
                os.unlink(fb)   # A frames may serve later --runs; main prunes
        else:
            nrows = rows_differing(ia, ib)
            gross_tol = int(args.tol8 * ia.maxval / 255.0)
            maxd, ndiff, ndiff_gross = sampled_diff(ia, ib, gross_tol)
            maxd8 = maxd * 255.0 / ia.maxval
            pct = 100.0 * ndiff / total_px
            pct_gross = 100.0 * ndiff_gross / total_px
            rec = {"i": i, "exact": False, "rows_differing": nrows,
                   "maxdiff8": round(maxd8, 2), "ndiff_est": ndiff,
                   "pct_est": round(pct, 3),
                   "ndiff_gross_est": ndiff_gross,
                   "pct_gross_est": round(pct_gross, 3)}
            if maxd8 > args.tol8:
                if first_div is None:
                    first_div = i
                    # forensic annotation: counter deltas across the interval
                    # that produced this frame
                    if i in shot_t and (i - 1) in shot_t:
                        t1 = int(shot_t[i - 1] * 1e9)
                        t2 = int(shot_t[i] * 1e9)
                        deltas = {}
                        for c in COUNTERS:
                            v1 = sa.value_at(c, t1)
                            v2 = sa.value_at(c, t2)
                            if v1 is not None and v2 is not None and v2 != v1:
                                deltas[c] = v2 - v1
                        rec["counter_deltas"] = deltas
                    if args.diff_png:
                        write_diff_png(os.path.join(
                            out_dir, f"{tag}_firstdiv_{i:05d}.png"),
                            ia, ib, args.amp)
                if pct_gross > args.vanish_pct:
                    rec["vanish"] = True
                    vanish_frames.append(i)
            elif not args.keep_all:
                # within tolerance: don't hoard 8K PNGs (30+ MB each)
                os.unlink(fb)
        frames.append(rec)
        del ia, ib

    counters = {c: sa.last(c) for c in COUNTERS if sa.last(c) is not None}
    ndiv = sum(1 for f in frames if not f["exact"]
               and f["maxdiff8"] > args.tol8)
    verdict = "PASS" if ndiv == 0 and ra["complete"] and rb["complete"] \
        else "FAIL"
    return {
        "tag": tag, "frames_compared": n,
        "a_complete": ra["complete"], "b_complete": rb["complete"],
        "divergent": ndiv, "first_divergent": first_div,
        "vanish_frames": vanish_frames,
        "verdict": verdict,
        "counters_b": counters,
        "frames": frames,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mpv", required=True,
                    help="mpv binary; a .exe is auto-driven via WSL interop "
                         "with /mnt/X path translation")
    ap.add_argument("--media", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--frames", type=int, required=True,
                    help="frames to capture (duration*fps)")
    ap.add_argument("--config-a", default="", help=".conf path or raw opts "
                    "(reference, e.g. the CPU baseline)")
    ap.add_argument("--config-b", default="", help=".conf path or raw opts "
                    "(config under test)")
    ap.add_argument("--extra-a", default="", help="raw opts appended after "
                    "config-a (e.g. local-only overrides)")
    ap.add_argument("--extra-b", default="", help="raw opts appended after "
                    "config-b (e.g. --sub-present-guard-ms=0 on the lavapipe "
                    "box, where the auto deadline fires on every heavy frame "
                    "and would pollute a content A/B with stale serves)")
    ap.add_argument("--geometry", default="1280x720")
    ap.add_argument("--gpu-sw", choices=["yes", "no", "auto"], default="auto",
                    help="auto: yes for a local binary (lavapipe box), no for "
                         "a .exe (real GPU rig)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="tab")
    ap.add_argument("--runs", type=int, default=1,
                    help="repeat the B (under-test) run N times against one A "
                         "reference run (state-dependent defects)")
    ap.add_argument("--settle", type=float, default=0.25)
    ap.add_argument("--settle0", type=float, default=3.0)
    ap.add_argument("--tol8", type=float, default=0.0,
                    help="per-frame maxdiff tolerance on the 8-bit scale "
                         "(0 = exact; 3 = the m5 blur bar)")
    ap.add_argument("--vanish-pct", type=float, default=0.05,
                    help="flag VANISH when the A/B differing area exceeds "
                         "this %% of the frame")
    ap.add_argument("--win-scratch", default="/mnt/c/Users/jsoos/8k-rig/out",
                    help="WSL-visible scratch for .exe runs (never SMB/Y:)")
    ap.add_argument("--keep-all", action="store_true",
                    help="keep matching PNGs too (default: deleted)")
    ap.add_argument("--diff-png", action="store_true", default=True)
    ap.add_argument("--no-diff-png", dest="diff_png", action="store_false")
    ap.add_argument("--amp", type=int, default=8)
    ap.add_argument("--run-timeout", type=int, default=0)
    ap.add_argument("--per-frame-timeout", type=int, default=20,
                    help="seconds/frame for the default run timeout")
    args = ap.parse_args(argv)

    win = is_windows_binary(args.mpv)
    os.makedirs(args.out, exist_ok=True)
    base = os.path.join(args.win_scratch, "temporal", args.tag) if win \
        else os.path.join(args.out, args.tag)

    cfg_a = parse_config(args.config_a) + args.extra_a.split()
    cfg_b = parse_config(args.config_b) + args.extra_b.split()

    print(f"[temporal_ab] tag={args.tag} start={args.start} "
          f"frames={args.frames} geometry={args.geometry} "
          f"mode={'rig(.exe)' if win else 'local'}")
    if win:
        print("[temporal_ab] NOTE: rig runs open a real mpv window on the "
              "Windows desktop for their duration.")

    ra = run_one(args.mpv, args.media, cfg_a, os.path.join(base, "a"),
                 args, win)
    print(f"  A: {len(ra['frames'])} frames in {ra['wall_s']}s "
          f"complete={ra['complete']}")
    results = []
    ok = True
    for k in range(1, args.runs + 1):
        rb = run_one(args.mpv, args.media, cfg_b,
                     os.path.join(base, f"b_run{k}"), args, win)
        print(f"  B#{k}: {len(rb['frames'])} frames in {rb['wall_s']}s "
              f"complete={rb['complete']}")
        r = compare_runs(ra, rb, args, f"{args.tag}_run{k}", args.out)
        results.append(r)
        fd = r["first_divergent"]
        print(f"  B#{k}: {r['verdict']}  divergent={r['divergent']}/"
              f"{r['frames_compared']}"
              + (f"  FIRST DIVERGENT frame={fd}" if fd is not None else "")
              + (f"  VANISH at {r['vanish_frames'][:8]}"
                 if r["vanish_frames"] else ""))
        if fd is not None:
            rec = r["frames"][fd]
            print(f"     frame {fd}: maxdiff8={rec['maxdiff8']} "
                  f"ndiff~{rec['ndiff_est']} ({rec.get('pct_est', 0)}%)  "
                  f"counters fired in interval: "
                  f"{rec.get('counter_deltas', {})}")
        if r["verdict"] != "PASS":
            ok = False

    # prune A frames that were within tolerance in EVERY run (divergent pairs
    # keep both sides for inspection)
    if not args.keep_all:
        keep = set()
        for r in results:
            for rec in r["frames"]:
                if not rec["exact"] and rec["maxdiff8"] > args.tol8:
                    keep.add(rec["i"])
        for i, fname in enumerate(ra["frames"]):
            if i not in keep:
                p = os.path.join(ra["shots"], fname)
                if os.path.exists(p):
                    os.unlink(p)

    summary = os.path.join(args.out, f"{args.tag}_summary.json")
    with open(summary, "w") as f:
        json.dump(results, f, indent=1)
    print(f"[temporal_ab] {'PASS' if ok else 'FAIL'}  summary: {summary}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
