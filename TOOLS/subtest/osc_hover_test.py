#!/usr/bin/env python3
"""osc_hover_test.py -- OSC hover-zone geometry acceptance test (WP-H8).

Regression under test: with --osd-render-res-cap engaged (window height >
cap), the OSD object's public geometry (osd-dimensions -> script layout ->
mp.set_mouse_area input zones) must still be the TRUE window size. The
pre-H8 code stored the CAPPED resolution into obj->vo_res, so the OSC laid
out its show/hide hover band for (e.g.) 2560x1440 while raw mouse
coordinates live in the true window space -- hovering the real window
bottom no longer showed the seekbar, while hovering the capped-space band
(a mid-window stripe) did.

Method: launch real mpv (gpu-next) at a large --geometry with --osc=yes,
inject mouse positions with the `mouse` command over JSON IPC, and decide
OSC visibility by diffing the window-screenshot bottom band against a
no-OSC baseline screenshot. The true window size is taken from the
screenshot dimensions (authoritative; osd-dimensions is itself under test).

Probed signals, per scenario (cap=N and cap=0):
  * true-band sweep   y = H-20        -> must show the OSC (both scenarios)
  * false-band sweep  y = 0.875*cap   -> the capped-space hover band; must
                      NOT show the OSC (if it does, the zones are displaced
                      into capped space -- the regression, or a fix that
                      merely moved them again)
  * osd-dimensions    must equal the true window size from the screenshot
  * mouse-pos.hover   after a true-band injection: cmd_mouse derives
                      enter/leave from osd_get_vo_res, so hover=false at a
                      valid in-window position is the same corruption

--expect fixed | broken asserts the full signature and sets the exit code;
without it the script just reports. Run from TOOLS/subtest/ (imports
abdiff.py for the PNG decoder and IPC client).
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abdiff import MpvIPC, read_png  # noqa: E402

# Deterministic, box-standard flags (mirrors abdiff BASE_FLAGS where it
# matters; OSC and window size are the subject here so they differ).
BASE_FLAGS = [
    "--no-config",
    "--gpu-sw=yes",
    "--vo=gpu-next",
    "--gpu-api=vulkan",
    "--no-audio",
    "--force-window=yes",
    "--keep-open=yes",
    "--no-border",
    "--osc=yes",
    # windowcontrols would add a second (top) hover zone that also triggers
    # show_osc(); disable it so the bottom band is the only show trigger.
    # A longer hidetimeout widens the show->screenshot window (zone
    # semantics under test are unchanged).
    "--script-opts=osc-windowcontrols=no,osc-hidetimeout=1500",
    "--dither=no",
    "--deband=no",
    "--no-input-default-bindings",
    "--no-osd-bar",
    "--screenshot-format=png",
    "--screenshot-png-compression=1",
    "--screenshot-png-filter=0",
    "--screenshot-high-bit-depth=no",
    "--screenshot-tag-colorspace=no",
]

DEFAULT_MEDIA = "av://lavfi:color=c=0x606060:s=1280x720:d=3600:r=24"

# A band pixel "differs" past this per-channel 8-bit delta ...
PIX_TOL = 12
# ... and the OSC counts as visible when more than this fraction of the
# sampled band pixels differ from the baseline band.
VISIBLE_FRAC = 0.02
# Bottom band sampled for OSC presence (fractions of true height).
BAND_Y0, BAND_Y1 = 0.955, 0.995


def band_diff_frac(img_a, img_b) -> float:
    """Fraction of differing pixels in the bottom band (subsampled)."""
    if (img_a.w, img_a.h) != (img_b.w, img_b.h):
        return 1.0
    y0, y1 = int(img_a.h * BAND_Y0), int(img_a.h * BAND_Y1)
    ch = img_a.channels
    total = 0
    diff = 0
    for y in range(y0, y1, 2):
        ra = img_a.row_samples(y)
        rb = img_b.row_samples(y)
        if ra == rb:
            total += img_a.w // 4
            continue
        for x in range(0, img_a.w, 4):
            base = x * ch
            total += 1
            for c in range(min(ch, 3)):
                if abs(ra[base + c] - rb[base + c]) > PIX_TOL:
                    diff += 1
                    break
    return diff / max(total, 1)


class Scenario:
    def __init__(self, mpv, media, geometry, cap, probe_cap, out, tag,
                 verbose):
        self.mpv = mpv
        self.media = media
        self.geometry = geometry
        self.cap = cap              # --osd-render-res-cap for this run
        self.probe_cap = probe_cap  # cap used to locate the FALSE band probe
        self.out = out
        self.tag = tag
        self.verbose = verbose
        self.proc = None
        self.ipc = None
        self.nshot = 0

    def log(self, msg):
        print(f"  [{self.tag}] {msg}", flush=True)

    def shot(self, name):
        self.nshot += 1
        path = os.path.join(self.out, f"{self.tag}_{self.nshot:02d}_{name}.png")
        self.ipc.command(["screenshot-to-file", path, "window"], timeout=60.0)
        return read_png(path), path

    def mouse(self, x, y):
        self.ipc.command(["mouse", int(x), int(y)])

    def sweep(self, y, w):
        """Inject a horizontal mouse sweep at row y within width w.

        Keep every position inside [0, w): leaving the showhide area (in x
        OR y) delivers MOUSE_LEAVE to the OSC, which hides it instantly --
        so a sweep that exits the zone would erase the very signal probed.
        """
        for fx in (0.15, 0.35, 0.55, 0.75, 0.85):
            self.mouse(w * fx, y)
            time.sleep(0.12)

    def run(self) -> dict:
        sock = os.path.join(self.out, f"ipc_{self.tag}.sock")
        log = os.path.join(self.out, f"mpv_{self.tag}.log")
        cmd = [self.mpv] + BASE_FLAGS + [
            f"--geometry={self.geometry}",
            f"--osd-render-res-cap={self.cap}",
            f"--input-ipc-server={sock}",
            f"--log-file={log}",
            "--msg-level=all=warn",
            self.media,
        ]
        res = {"tag": self.tag, "cap": self.cap}
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        try:
            self.ipc = MpvIPC(sock, timeout=40.0)
            self.ipc.wait_ready()
            time.sleep(2.0)

            # Baseline (no mouse has entered): OSC must be hidden. If two
            # successive shots disagree in the band, something is still
            # settling -- retry.
            base, base_path = self.shot("baseline")
            for _ in range(4):
                chk, _ = self.shot("baseline_chk")
                if band_diff_frac(base, chk) < VISIBLE_FRAC:
                    break
                time.sleep(1.5)
                base, base_path = self.shot("baseline")
            w, h = base.w, base.h
            res["window"] = [w, h]
            self.log(f"true window size (from screenshot): {w}x{h}")
            if self.cap > 0 and h <= self.cap:
                raise RuntimeError(
                    f"window height {h} <= cap {self.cap}: cap never engages "
                    f"(WM refused the geometry?); use a smaller --cap")

            dim = self.ipc.command(["get_property", "osd-dimensions"])
            res["osd_dim"] = [dim.get("w"), dim.get("h")]
            res["osd_dim_true"] = (dim.get("w") == w and dim.get("h") == h)
            self.log(f"osd-dimensions: {dim.get('w')}x{dim.get('h')} "
                     f"(true={res['osd_dim_true']})")

            # --- TRUE band: hover the real window bottom -------------------
            y_true = h - 20
            self.sweep(y_true, w)
            hover = self.ipc.command(["get_property", "mouse-pos"])
            res["hover_at_true_band"] = bool(hover.get("hover"))
            time.sleep(0.35)
            img, p = self.shot("true_band")
            frac = band_diff_frac(base, img)
            if frac < VISIBLE_FRAC:      # one retry to dodge fade races
                self.sweep(y_true, w)
                time.sleep(0.35)
                img, p = self.shot("true_band_retry")
                frac = band_diff_frac(base, img)
            res["true_band_visible"] = frac >= VISIBLE_FRAC
            res["true_band_frac"] = round(frac, 4)
            self.log(f"true-band hover (y={y_true}): band diff {frac:.2%} -> "
                     f"OSC {'VISIBLE' if res['true_band_visible'] else 'hidden'}"
                     f"  (mouse-pos.hover={res['hover_at_true_band']})")

            # --- reset: park the mouse between both candidate zones --------
            self.mouse(w // 2, int(h * 0.40))
            hidden = False
            for _ in range(14):
                time.sleep(0.5)
                img, _ = self.shot("reset")
                if band_diff_frac(base, img) < VISIBLE_FRAC:
                    hidden = True
                    break
            res["hidden_after_reset"] = hidden
            self.log(f"hidden after reset: {hidden}")

            # --- FALSE band: the capped-space hover zone --------------------
            # Pre-fix, mouse areas are computed against a cap-high osd space,
            # so in raw window pixels the show zone is roughly
            # [0.69*cap, cap] x [0, w*cap/h]; probe its middle. It must NOT
            # show the OSC once the fix pins zones to true geometry
            # (0.875*cap < 0.69*h needs h > 1.27*cap -- holds for 1440 vs
            # 2160). The x sweep stays inside the capped-space WIDTH, both
            # because that's where the displaced zone lives (repro) and
            # because exiting it mid-sweep insta-hides the OSC (see sweep()).
            y_false = int(self.probe_cap * 0.875)
            w_false = int(round(w * self.probe_cap / h))
            self.sweep(y_false, w_false)
            time.sleep(0.35)
            img, p = self.shot("false_band")
            frac = band_diff_frac(base, img)
            res["false_band_visible"] = frac >= VISIBLE_FRAC
            res["false_band_frac"] = round(frac, 4)
            res["false_band_y"] = y_false
            self.log(f"false-band hover (y={y_false}, x<{w_false}): band diff "
                     f"{frac:.2%} -> "
                     f"OSC {'VISIBLE' if res['false_band_visible'] else 'hidden'}")
            return res
        finally:
            try:
                if self.ipc:
                    self.ipc.command_raw(["quit"], timeout=5.0)
                    self.ipc.close()
            except Exception:
                pass
            if self.proc:
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mpv", required=True)
    ap.add_argument("--media", default=DEFAULT_MEDIA)
    ap.add_argument("--geometry", default="3840x2160")
    ap.add_argument("--cap", type=int, default=1440)
    ap.add_argument("--out", default="out_osc")
    ap.add_argument("--expect", choices=["fixed", "broken"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    results = {}
    for tag, cap in (("cap", args.cap), ("nocap", 0)):
        print(f"scenario {tag} (osd-render-res-cap={cap}):", flush=True)
        results[tag] = Scenario(args.mpv, args.media, args.geometry, cap,
                                args.cap, args.out, tag, args.verbose).run()

    with open(os.path.join(args.out, "osc_hover_result.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    cap_r, nocap_r = results["cap"], results["nocap"]
    # Detection sanity: the cap=0 run must show the OSC on the true band,
    # otherwise the harness itself is not measuring anything.
    if not nocap_r["true_band_visible"]:
        print("HARNESS ERROR: OSC not detected on bottom hover even with "
              "cap=0 -- detection is broken, no verdict.", flush=True)
        return 2

    fixed_sig = (cap_r["true_band_visible"]
                 and not cap_r["false_band_visible"]
                 and cap_r["osd_dim_true"]
                 and nocap_r["osd_dim_true"]
                 and cap_r["hover_at_true_band"])
    broken_sig = (not cap_r["true_band_visible"]
                  and cap_r["false_band_visible"]
                  and not cap_r["osd_dim_true"])

    print()
    print(f"cap={cap_r['cap']}: true-band visible={cap_r['true_band_visible']} "
          f"false-band visible={cap_r['false_band_visible']} "
          f"osd-dim true={cap_r['osd_dim_true']} "
          f"hover@bottom={cap_r['hover_at_true_band']}")
    print(f"cap=0:    true-band visible={nocap_r['true_band_visible']} "
          f"osd-dim true={nocap_r['osd_dim_true']}")
    verdict = ("FIXED" if fixed_sig else
               "BROKEN (regression reproduced)" if broken_sig else "MIXED")
    print(f"verdict: {verdict}")

    if args.expect == "fixed":
        return 0 if fixed_sig else 1
    if args.expect == "broken":
        return 0 if broken_sig else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
