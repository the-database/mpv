#!/usr/bin/env python3
"""Windows D3D11 regression for live ASS reinit with retained outline frames."""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import time
import uuid

from test_live_ass_options import ASS, IPC


def fixture(events):
    header = ASS[:ASS.index("Dialogue:")]
    return header + "".join(
        f"Dialogue: {i % 3},0:00:00.00,0:00:20.00,Default,,0,0,0,,"
        f"{{\\pos({15 + i % 20 * 31},{15 + i // 20 * 21})"
        f"\\fs14\\bord1\\blur0.5}}Glyph {i}\n"
        for i in range(events)
    )


def run_case(args, events, ahead, gpu, threads):
    name = f"events{events}-ahead{ahead}-gpu{int(gpu)}-threads{threads}"
    out = args.out / name
    out.mkdir()
    ass = out / "overlap.ass"
    ass.write_text(fixture(events), encoding="utf-8")
    pipe = r"\\.\pipe\ajn-ass-lifetime-" + uuid.uuid4().hex
    cmd = [str(args.mpv), "--no-config", "--idle=yes", "--keep-open=yes",
           "--pause=yes", "--force-window=yes", "--window-minimized=yes",
           "--vo=gpu-next", "--gpu-api=d3d11", "--gpu-context=d3d11",
           "--audio=no", "--osd-level=0", "--sub-ass-override=force",
           "--sub-font-size=25", f"--sub-render-ahead-frames={ahead}",
           f"--sub-render-ahead-threads={threads}", f"--sub-ass-render-threads={threads}",
           f"--sub-gpu-raster={'yes' if gpu else 'no'}",
           f"--sub-gpu-composite={'yes' if gpu else 'no'}",
           "--input-ipc-server=" + pipe, "--log-file=" + str(out / "mpv.log"),
           "--sub-file=" + str(ass), str(args.video)]
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    channel = None
    result = {"case": name, "toggles": 0}
    try:
        deadline = time.monotonic() + 15
        while channel is None:
            if process.poll() is not None:
                raise RuntimeError(f"mpv exited {process.returncode} before IPC connected")
            try:
                channel = open(pipe, "r+b", buffering=0)
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(.1)
        ipc = IPC(channel)
        time.sleep(1)
        result["mpvVersion"] = ipc.call("get_property", "mpv-version")
        ipc.call("set_property", "pause", False)
        for i in range(args.toggles):
            ipc.call("set_property", "sub-ass-override", "no" if i % 2 == 0 else "force")
            result["toggles"] += 1
            time.sleep(.05)
        # Let queued renders and their final releases run before shutdown.
        time.sleep(.5)
        ipc.call("quit")
        process.wait(timeout=10)
        result["exitCode"] = process.returncode
    except Exception as error:
        result["error"] = str(error)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        result["exitCode"] = process.poll()
    finally:
        if channel:
            channel.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
    result["pass"] = ("error" not in result and result["exitCode"] == 0
                      and result["toggles"] == args.toggles)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mpv", type=lambda p: Path(p).resolve(), required=True)
    ap.add_argument("--video", type=lambda p: Path(p).resolve(), required=True)
    ap.add_argument("--out", type=lambda p: Path(p).resolve(), required=True)
    ap.add_argument("--toggles", type=int, default=80)
    args = ap.parse_args()
    if not 1 <= args.toggles <= 200:
        ap.error("--toggles must be between 1 and 200")
    args.out.mkdir(parents=True, exist_ok=False)
    # Inherited by these child players only; do not change system crash settings.
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    previous = kernel.SetErrorMode(0x0002)  # SEM_NOGPFAULTERRORBOX
    results = []
    try:
        for case in [(1, 60, True, 4), (320, 60, True, 4), (320, 0, True, 4),
                     (320, 60, False, 4), (320, 0, False, 4), (320, 60, True, 1)]:
            results.append(run_case(args, *case))
            (args.out / "results.json").write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(results[-1]), flush=True)
    finally:
        kernel.SetErrorMode(previous)
    return 0 if all(result["pass"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
