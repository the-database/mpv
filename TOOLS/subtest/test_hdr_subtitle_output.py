"""Render fixed PQ output to files; does not measure physical HDR display nits."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import cast
import uuid

assert sys.platform == "win32", "This D3D11/named-pipe test requires Windows."
if sys.platform == "win32":
    from PIL import Image, ImageChops, ImageStat
    from test_live_ass_options import ASS, IPC

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--mpv", type=lambda x: Path(x).resolve(), required=True)
ap.add_argument("--video", type=lambda x: Path(x).resolve(), required=True)
ap.add_argument("--out", type=lambda x: Path(x).resolve(), required=True)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
ass = args.out / "style.ass"
ass.write_text(ASS, encoding="utf-8")
results = []
for gpu in (False, True):
    images = []
    for peak in (100, 203):
        name = f"gpu{int(gpu)}-peak{peak}"
        pipe = r"\\.\pipe\ajn-hdr-probe-" + uuid.uuid4().hex
        cmd = [str(args.mpv), "--no-config", "--idle=yes", "--keep-open=yes", "--pause=yes",
               "--force-window=yes", "--window-minimized=yes", "--vo=gpu-next", "--gpu-api=d3d11",
               "--gpu-context=d3d11", "--audio=no", "--osd-level=0", "--sub-ass-override=force",
               "--sub-color=#FFFFFF", "--sub-font-size=35", "--sub-border-size=0",
               "--sub-shadow-offset=0",
               "--sub-render-ahead-frames=0", f"--sub-gpu-raster={'yes' if gpu else 'no'}",
               f"--sub-gpu-composite={'yes' if gpu else 'no'}", f"--sub-hdr-peak={peak}",
               "--target-colorspace-hint=no", "--target-trc=pq", "--target-prim=bt.2020",
               "--target-peak=1000", "--screenshot-format=png", "--screenshot-high-bit-depth=yes",
               "--screenshot-tag-colorspace=yes",
               "--input-ipc-server=" + pipe, "--log-file=" + str(args.out / (name + ".log")),
               "--sub-file=" + str(ass), str(args.video)]
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        channel = None
        try:
            deadline = time.monotonic() + 15
            while channel is None:
                if process.poll() is not None:
                    raise RuntimeError(f"mpv exited: {name}")
                try:
                    channel = open(pipe, "r+b", buffering=0)
                except OSError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(.1)
            ipc = IPC(channel)
            time.sleep(1)
            ipc.call("set_property", "pause", False)
            time.sleep(.5)
            ipc.call("set_property", "pause", True)
            time.sleep(.3)
            params = ipc.call("get_property", "video-params")
            assert params["gamma"] == "pq", params
            path = args.out / (name + ".png")
            ipc.call("screenshot-to-file", str(path), "window")
            with Image.open(path) as im:
                images.append(im.convert("RGB"))
            ipc.call("quit")
            process.wait(timeout=10)
        finally:
            if channel:
                channel.close()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    difference = ImageChops.difference(*images)
    result = {"gpu": gpu, "changed": difference.getbbox() is not None,
              "meanDifference": ImageStat.Stat(difference).mean,
              "extrema100": images[0].getextrema(), "extrema203": images[1].getextrema()}
    # A single channel of these RGB images has integer min/max extrema.
    peak100 = cast(tuple[int, int], images[0].getchannel("R").getextrema())[1]
    peak203 = cast(tuple[int, int], images[1].getchannel("R").getextrema())[1]
    result["pass"] = difference.getbbox() is not None and peak203 > peak100 + 5
    results.append(result)
    print(json.dumps(result), flush=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

raise SystemExit(0 if all(result["pass"] for result in results) else 1)
