#!/usr/bin/env python3
"""Windows D3D11 regression: live ASS override with render-ahead / GPU toggled.

Requires Pillow, an mpv executable, and a black 640x360 video of at least
20 seconds. Uses an original generated ASS fixture and bounded named-pipe IPC.
Screenshots, mpv logs and results.json are saved in a new output directory.
"""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

# Also tells type checkers that the remaining code uses Windows-only APIs.
assert sys.platform == "win32", "This D3D11/named-pipe test requires Windows."
if sys.platform == "win32":
    import msvcrt
    from PIL import Image

ASS = """[Script Info]
ScriptType: v4.00+
PlayResX: 640
PlayResY: 360
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, \
Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, \
Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,42,&H0000FF00,&H0000FF00,&H00000000,&H00000000,\
0,0,0,0,100,100,0,0,1,0,0,2,20,20,35,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:20.00,Default,,0,0,0,,Styled subtitle test
"""


class IPC:
    def __init__(self, channel):
        self.channel = channel
        self.counter = 0
        self.buffer = b""
        self.handle = ctypes.c_void_p(msvcrt.get_osfhandle(channel.fileno()))
        self.peek = ctypes.WinDLL("kernel32", use_last_error=True).PeekNamedPipe
        self.peek.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                              ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
        self.peek.restype = ctypes.c_int

    def call(self, *command):
        self.counter += 1
        request = {"command": command, "request_id": self.counter}
        self.channel.write((json.dumps(request) + "\n").encode())
        until = time.monotonic() + 10
        while time.monotonic() < until:
            if b"\n" not in self.buffer:
                available = ctypes.c_uint32()
                if not self.peek(self.handle, None, 0, None, ctypes.byref(available), None):
                    raise ctypes.WinError(ctypes.get_last_error())
                if not available.value:
                    time.sleep(.01)
                    continue
                self.buffer += self.channel.read(available.value)
                continue
            line, self.buffer = self.buffer.split(b"\n", 1)
            reply = json.loads(line)
            if reply.get("request_id") == self.counter:
                if reply.get("error") != "success":
                    raise RuntimeError(reply)
                return reply.get("data")
        raise TimeoutError(f"mpv IPC: {command}")


def run_case(args, ass, ahead, gpu):
    out = args.out / f"ahead{ahead}-gpu{int(gpu)}"
    out.mkdir()
    pipe = r"\\.\pipe\ajn-subtitle-options-" + uuid.uuid4().hex
    cmd = [str(args.mpv), "--no-config", "--idle=yes", "--keep-open=yes", "--pause=yes",
           "--force-window=yes", "--window-minimized=yes", "--vo=gpu-next", "--gpu-api=d3d11",
           "--gpu-context=d3d11", "--audio=no", "--osd-level=0", "--sub-ass-override=force",
           "--sub-color=#FFFFFF", "--sub-font-size=25", "--sub-border-size=0",
           "--sub-shadow-offset=0",
           f"--sub-render-ahead-frames={ahead}", f"--sub-gpu-raster={'yes' if gpu else 'no'}",
           f"--sub-gpu-composite={'yes' if gpu else 'no'}", "--target-colorspace-hint=no",
           "--target-trc=srgb", "--target-prim=bt.709", "--target-peak=203",
           "--screenshot-format=png", "--screenshot-high-bit-depth=no",
           "--input-ipc-server=" + pipe, "--log-file=" + str(out / "mpv.log"),
           "--sub-file=" + str(ass), str(args.video)]
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    channel = None
    result = {"case": out.name, "checks": []}
    try:
        until = time.monotonic() + 15
        while channel is None:
            if process.poll() is not None:
                raise RuntimeError(f"mpv exited {process.returncode}; see {out / 'mpv.log'}")
            try:
                channel = open(pipe, "r+b", buffering=0)
            except OSError:
                if time.monotonic() > until:
                    raise
                time.sleep(.1)
        ipc = IPC(channel)
        time.sleep(2)
        result["mpvVersion"] = ipc.call("get_property", "mpv-version")
        ipc.call("set_property", "pause", False)
        time.sleep(1)
        ipc.call("set_property", "pause", True)

        def snapshot(label, styled):
            time.sleep(.5)
            path = out / (label + ".png")
            ipc.call("screenshot-to-file", str(path), "window")
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                pixels = list(getattr(rgb, "get_flattened_data", rgb.getdata)())
            green = sum(g > 80 and g > r * 2 and g > b * 2 for r, g, b in pixels)
            white = sum(min(r, g, b) > 100 and max(r, g, b) - min(r, g, b) < 10
                        for r, g, b in pixels)
            passed = green > 100 if styled else green == 0 and white > 100
            result["checks"].append({
                "name": label, "pass": passed, "greenPixels": green, "whitePixels": white,
            })

        snapshot("force-initial", False)
        ipc.call("set_property", "sub-ass-override", "no")
        snapshot("paused-no", True)
        ipc.call("set_property", "sub-ass-override", "force")
        snapshot("paused-force", False)
        ipc.call("set_property", "pause", False)
        ipc.call("set_property", "sub-ass-override", "no")
        time.sleep(.5)
        ipc.call("set_property", "pause", True)
        snapshot("playing-no", True)
        ipc.call("seek", 8, "absolute+exact")
        snapshot("no-after-seek", True)
        ipc.call("set_property", "sub-ass-override", "force")
        snapshot("force-after-seek", False)
        ipc.call("quit")
        process.wait(timeout=10)
        result["exitCode"] = process.returncode
    except Exception as error:
        result["error"] = str(error)
    finally:
        if channel:
            channel.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
    result["pass"] = ("error" not in result and result["exitCode"] == 0
                      and all(check["pass"] for check in result["checks"]))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mpv", type=lambda p: Path(p).resolve(), required=True)
    ap.add_argument("--video", type=lambda p: Path(p).resolve(), required=True)
    ap.add_argument("--out", type=lambda p: Path(p).resolve(), required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    ass = args.out / "style-fixture.ass"
    ass.write_text(ASS, encoding="utf-8")
    results = []
    for ahead, gpu in [(0, False), (60, False), (0, True), (60, True)]:
        results.append(run_case(args, ass, ahead, gpu))
        (args.out / "results.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8",
        )
        print(json.dumps(results[-1]), flush=True)
    return 0 if all(result["pass"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
