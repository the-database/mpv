#!/usr/bin/env python3
"""D3D11 regression: updating stats must survive an overlay deadline miss.

Uses the existing delay-injection option, a continuous subtitle and the real
stats script. The VO log reports the actual retained presentation overlays;
mpv screenshots cannot test this because they rebuild overlays without a guard.
Requires Windows, Pillow (through the shared fixture) and a black video >=7s.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

from test_live_ass_options import ASS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mpv", type=lambda p: Path(p).resolve(), required=True)
    ap.add_argument("--video", type=lambda p: Path(p).resolve(), required=True)
    ap.add_argument("--out", type=lambda p: Path(p).resolve(), required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "constant.ass").write_text(ASS, encoding="utf-8")
    script = args.out / "stats.lua"
    script.write_text("""local mp = require 'mp'
mp.register_event('file-loaded', function()
    mp.commandv('script-binding', 'stats/display-stats-toggle')
    mp.add_timeout(3.2, function()
        mp.commandv('script-binding', 'stats/display-stats-toggle')
    end)
end)
""", encoding="utf-8")
    results = []
    # Both stats paths: default osd_message and persistent script overlay.
    for persistent in (False, True):
        name = "script-overlay" if persistent else "osd-message"
        log = args.out / f"{name}.log"
        cmd = [str(args.mpv), "--no-config", "--vo=gpu-next",
               "--gpu-api=d3d11", "--gpu-context=d3d11", "--audio=no",
               "--window-minimized=yes", "--sub-gpu-raster=yes",
               "--sub-gpu-composite=yes", "--sub-render-ahead-frames=60",
               "--sub-present-guard-ms=1", "--sub-debug-stall-ms=5",
               "--length=6", "--msg-level=vo/gpu-next=v",
               "--script=" + str(script), "--log-file=" + str(log),
               "--sub-file=" + str(args.out / "constant.ass"),
               "--script-opts=stats-redraw_delay=0.05,stats-persistent_overlay="
               + ("yes" if persistent else "no"), str(args.video)]
        with (args.out / f"{name}-console.log").open("wb") as console:
            process = subprocess.run(cmd, stdout=console, stderr=subprocess.STDOUT,
                                     timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
        text = log.read_text(encoding="utf-8", errors="replace")
        samples = [(float(pts), int(kept), int(total)) for pts, kept, total in re.findall(
            r"\[present-guard\] deadline exceeded at pts ([\d.]+): presenting "
            r"previous overlays \(pts [\d.]+, (\d+) of (\d+)\)", text)]
        visible = [s for s in samples if 1 <= s[0] < 3]
        hidden = [s for s in samples if 4.5 <= s[0] < 6]
        result = {"case": name, "exitCode": process.returncode,
                  "visibleSamples": visible, "hiddenSamples": hidden}
        result["pass"] = (process.returncode == 0 and len(visible) >= 5
                          and all(kept == total == 2 for _, kept, total in visible)
                          and len(hidden) >= 5
                          and all(kept == total == 1 for _, kept, total in hidden))
        results.append(result)
        (args.out / "results.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {'PASS' if result['pass'] else 'FAIL'}; "
              f"{len(visible)} visible / {len(hidden)} hidden samples", flush=True)
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
