#!/usr/bin/env python3
"""
run_matrix.py -- acceptance runner for the subtitle-performance matrix.

Given a config matrix (name -> list of extra mpv options) and a video, either
  (a) run each config locally with a real mpv binary (--mpv PATH), capturing
      --dump-stats and then adjudicating with parse_stats.py, or
  (b) emit a Windows batch + per-config .conf files (--emit-windows DIR) that a
      human runs on a Windows mpv, copying the stats_*.txt back for offline
      adjudication with parse_stats.py.

Both modes exercise a scripted mid-run seek (see SEEK, below).

Python 3.12 stdlib only.

--------------------------------------------------------------------------
CONFIG MATRIX FORMAT
--------------------------------------------------------------------------
Preferred: JSON object mapping config-name -> options.  Options may be a list
of strings or a single whitespace-separated string:

    {
      "off":        ["--sub-visibility=no"],
      "on":         "--vf=  --sub-ass-override=force",
      "composite":  ["--vo=gpu-next", "--gpu-composite=yes"]
    }

Also accepted (TOML-ish / ini-ish fallback, one config per line):

    off       = --sub-visibility=no
    on        : --sub-ass-override=force
    composite = --vo=gpu-next --gpu-composite=yes

Lines beginning with '#' are comments.  A "base" key (if present) is prepended
to every other config's options.

--------------------------------------------------------------------------
SEEK
--------------------------------------------------------------------------
Local mode:  mpv is launched with --input-ipc-server=<socket>; a background
thread waits --seek-at seconds then sends {"command":["seek", S, "absolute"]}
to force a subtitle/render discontinuity mid-run.

Windows/emit mode:  IPC sockets are awkward to script portably in a .bat, so
the emitted run_tests.bat documents a manual seek: it prints an instruction to
press the seek key (default: right-arrow a few times, or drag the seek bar)
around the halfway mark.  Simpler still, pass --seek-at 0 to skip the seek and
instead run the tail of the file via --start (documented in the .bat).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time

import parse_stats


# --------------------------------------------------------------------------
# config loading
# --------------------------------------------------------------------------

def _as_opt_list(v) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        return v.split()
    raise ValueError(f"bad option value: {v!r}")


def load_matrix(path: str) -> dict[str, list[str]]:
    """Load a config matrix (JSON preferred, ini-ish fallback)."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    raw: dict[str, object]
    try:
        raw = json.loads(text)
        if not isinstance(raw, dict):
            raise ValueError("top-level JSON must be an object")
    except (json.JSONDecodeError, ValueError):
        raw = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^([^=:]+?)\s*[=:]\s*(.*)$", line)
            if not m:
                continue
            raw[m.group(1).strip()] = m.group(2).strip()

    base = _as_opt_list(raw.pop("base")) if "base" in raw else []
    matrix: dict[str, list[str]] = {}
    for name, val in raw.items():
        matrix[name] = base + _as_opt_list(val)
    return matrix


# --------------------------------------------------------------------------
# local run
# --------------------------------------------------------------------------

def _seek_thread(sock_path: str, seek_at: float, seek_to: float, log) -> None:
    """After seek_at seconds, send an absolute seek over the mpv IPC socket."""
    time.sleep(seek_at)
    for attempt in range(20):
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(sock_path)
            cmd = json.dumps({"command": ["seek", seek_to, "absolute"]}) + "\n"
            s.sendall(cmd.encode())
            s.close()
            log(f"  [seek] sent seek -> {seek_to}s")
            return
        except OSError:
            time.sleep(0.25)
    log("  [seek] WARNING: could not connect to IPC socket; seek skipped")


def run_one_local(name: str, opts: list[str], args) -> str:
    """Run a single config with a real mpv; return the stats file path."""
    stats_path = os.path.join(args.out, f"stats_{name}.txt")
    sock_path = os.path.join(tempfile.gettempdir(), f"mpv-subtest-{name}-{os.getpid()}.sock")

    cmd = [
        args.mpv,
        args.video,
        "--no-audio",
        "--no-config" if args.no_config else "--config=yes",
        f"--dump-stats={stats_path}",
        f"--input-ipc-server={sock_path}",
    ]
    if args.frames:
        cmd.append(f"--frames={args.frames}")
    elif args.duration:
        cmd.append(f"--length={args.duration}")
    cmd += opts

    def log(m):
        print(m, flush=True)

    log(f"[{name}] {' '.join(cmd)}")

    seeker = None
    if args.seek_at and args.seek_at > 0:
        seeker = threading.Thread(
            target=_seek_thread, args=(sock_path, args.seek_at, args.seek_to, log),
            daemon=True,
        )
        seeker.start()

    try:
        subprocess.run(cmd, check=False)
    finally:
        try:
            os.unlink(sock_path)
        except OSError:
            pass
    if seeker is not None:
        seeker.join(timeout=1.0)
    return stats_path


def run_local(matrix: dict[str, list[str]], args) -> int:
    os.makedirs(args.out, exist_ok=True)
    reports = []
    for name, opts in matrix.items():
        stats_path = run_one_local(name, opts, args)
        if not os.path.exists(stats_path):
            print(f"[{name}] WARNING: no stats file produced", file=sys.stderr)
            reports.append(parse_stats.summarize(name, [], args.budget))
            continue
        st = parse_stats.parse_stats_file(stats_path)
        rep = parse_stats.summarize(args.event, st.durations.get(args.event, []), args.budget)
        rep.event = name  # label the row by config name
        reports.append(rep)

    print()
    print(f"=== matrix results: event={args.event} budget={args.budget:.1f} ms ===")
    parse_stats.print_table(reports)
    return 0


# --------------------------------------------------------------------------
# windows emit
# --------------------------------------------------------------------------

BAT_HEADER = r"""@echo off
REM ------------------------------------------------------------------
REM  subtitle-performance acceptance matrix -- generated by run_matrix.py
REM
REM  1. Edit MPV and VIDEO below to match this machine.
REM  2. Run this .bat.  It plays VIDEO once per config, writing
REM     stats_<name>.txt next to this file.
REM  3. SCRIPTED SEEK: during each run, press RIGHT ARROW a few times
REM     around the halfway point (or drag the seek bar) to force a
REM     subtitle/render discontinuity.  (mpv IPC sockets are painful to
REM     drive from a .bat, so the seek is manual here.)
REM     -- OR -- set SEEKSTART below to a non-empty "--start=SECONDS" to
REM     instead sample the tail of the file with no manual step.
REM  4. Copy the stats_*.txt files back and adjudicate with:
REM        python3 parse_stats.py stats_<name>.txt --assert-under video-draw=41.7
REM ------------------------------------------------------------------

set MPV={mpv}
set VIDEO={video}
set COMMON=--no-audio {lenflag}
set SEEKSTART=
"""

BAT_RUN = r"""
echo === {name} ===
"%MPV%" "%VIDEO%" %COMMON% %SEEKSTART% --include="{conf}" --dump-stats=stats_{name}.txt
"""


def emit_windows(matrix: dict[str, list[str]], args) -> int:
    d = args.emit_windows
    os.makedirs(d, exist_ok=True)

    if args.frames:
        lenflag = f"--frames={args.frames}"
    elif args.duration:
        lenflag = f"--length={args.duration}"
    else:
        lenflag = ""

    bat = BAT_HEADER.format(
        mpv=args.mpv or "mpv.exe",
        video=args.video or "VIDEO.mkv",
        lenflag=lenflag,
    )

    for name, opts in matrix.items():
        conf_name = f"{name}.conf"
        conf_path = os.path.join(d, conf_name)
        with open(conf_path, "w", encoding="utf-8", newline="\r\n") as fh:
            fh.write(f"# config: {name}\n")
            for o in opts:
                # "--foo=bar" -> "foo=bar"; "--foo" -> "foo=yes"
                s = o.lstrip("-")
                if "=" in s:
                    fh.write(s + "\n")
                else:
                    fh.write(s + "=yes\n")
        bat += BAT_RUN.format(name=name, conf=conf_name)

    bat += "\necho.\necho All runs done. Copy stats_*.txt back and run parse_stats.py.\npause\n"

    bat_path = os.path.join(d, "run_tests.bat")
    with open(bat_path, "w", encoding="utf-8", newline="\r\n") as fh:
        fh.write(bat)

    print(f"wrote {bat_path}")
    for name in matrix:
        print(f"wrote {os.path.join(d, name + '.conf')}")
    print("\nEdit MPV/VIDEO at the top of run_tests.bat, then run it on Windows.")
    print("Perform the mid-run seek manually as documented in the .bat header.")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run / emit a subtitle-performance acceptance matrix.")
    p.add_argument("config", help="matrix config file (JSON or ini-ish)")
    p.add_argument("--video", help="video path (required for --mpv; template for --emit-windows)")
    p.add_argument("--event", default="video-draw", help="event to report (default video-draw)")
    p.add_argument("--budget", type=float, default=41.7, help="over-budget ms (default 41.7)")
    p.add_argument("--fps", type=float, help="set budget to 1000/FPS")

    # run length
    p.add_argument("--duration", type=float, help="seconds to play per config")
    p.add_argument("--frames", type=int, help="frames to play per config (overrides --duration)")

    # seek
    p.add_argument("--seek-at", type=float, default=0.0,
                   help="local mode: seconds into playback to fire a seek (0=off)")
    p.add_argument("--seek-to", type=float, default=0.0,
                   help="local mode: absolute seek target in seconds (default 0)")

    # modes (mutually exclusive-ish)
    p.add_argument("--mpv", help="path to mpv binary -> run locally")
    p.add_argument("--emit-windows", metavar="DIR",
                   help="write run_tests.bat + .conf files into DIR")
    p.add_argument("--out", default="matrix-out",
                   help="local mode: output dir for stats files (default matrix-out)")
    p.add_argument("--no-config", action="store_true",
                   help="local mode: pass --no-config to mpv")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fps:
        args.budget = 1000.0 / args.fps

    matrix = load_matrix(args.config)
    if not matrix:
        print("error: empty config matrix", file=sys.stderr)
        return 2
    print(f"loaded {len(matrix)} config(s): {', '.join(matrix)}")

    if args.emit_windows:
        return emit_windows(matrix, args)

    if args.mpv:
        if not args.video:
            print("error: --video is required with --mpv", file=sys.stderr)
            return 2
        return run_local(matrix, args)

    print("error: choose a mode -- --mpv PATH (local) or --emit-windows DIR",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
