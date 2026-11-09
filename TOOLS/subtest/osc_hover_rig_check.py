#!/usr/bin/env python3
"""osc_hover_rig_check.py -- adjudicate an osc_hover_rig.lua capture set.

Applies osc_hover_test.py's band detection (band_diff_frac, VISIBLE_FRAC)
to the screenshots the lua captured on the rig, and the same fixed/broken
signature. Usage:

  python3 osc_hover_rig_check.py --cap-dir D1 --nocap-dir D2 \
      --cap-log L1 --nocap-log L2 [--expect fixed]
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abdiff import read_png                      # noqa: E402
from osc_hover_test import band_diff_frac, VISIBLE_FRAC  # noqa: E402


def parse_log(path):
    """Pull the ohr: probe lines out of the mpv log."""
    out = {}
    pat = re.compile(r"ohr: probe (\w+) osd-dim=(-?\d+)x(-?\d+) hover=(\w+)")
    with open(path, errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                out[m.group(1)] = {
                    "osd_dim": [int(m.group(2)), int(m.group(3))],
                    "hover": m.group(4) == "true",
                }
    return out


def scenario(d, log_path, tag, cap):
    base = read_png(os.path.join(d, "baseline.png"))
    chk = read_png(os.path.join(d, "baseline_chk.png"))
    res = {"tag": tag, "cap": cap, "window": [base.w, base.h],
           "baseline_stable": band_diff_frac(base, chk) < VISIBLE_FRAC,
           "done": os.path.exists(os.path.join(d, "DONE"))}
    t1 = band_diff_frac(base, read_png(os.path.join(d, "true_band.png")))
    t2 = band_diff_frac(base, read_png(os.path.join(d, "true_band2.png")))
    frac = max(t1, t2)
    res["true_band_frac"] = round(frac, 4)
    res["true_band_visible"] = frac >= VISIBLE_FRAC
    r = band_diff_frac(base, read_png(os.path.join(d, "reset.png")))
    res["hidden_after_reset"] = r < VISIBLE_FRAC
    f = band_diff_frac(base, read_png(os.path.join(d, "false_band.png")))
    res["false_band_frac"] = round(f, 4)
    res["false_band_visible"] = f >= VISIBLE_FRAC
    probes = parse_log(log_path)
    if "true_band" in probes:
        res["hover_at_true_band"] = probes["true_band"]["hover"]
        res["osd_dim"] = probes["true_band"]["osd_dim"]
        res["osd_dim_true"] = probes["true_band"]["osd_dim"] == [base.w, base.h]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-dir", required=True)
    ap.add_argument("--nocap-dir", required=True)
    ap.add_argument("--cap-log", required=True)
    ap.add_argument("--nocap-log", required=True)
    ap.add_argument("--cap", type=int, default=1440)
    ap.add_argument("--expect", choices=["fixed", "broken"])
    ap.add_argument("--json-out")
    args = ap.parse_args()

    cap_r = scenario(args.cap_dir, args.cap_log, "cap", args.cap)
    nocap_r = scenario(args.nocap_dir, args.nocap_log, "nocap", 0)
    results = {"cap": cap_r, "nocap": nocap_r}
    print(json.dumps(results, indent=2))

    if not nocap_r["true_band_visible"]:
        print("HARNESS ERROR: OSC not detected on bottom hover even with "
              "cap=0 -- detection is broken, no verdict.")
        return 2

    fixed_sig = (cap_r["true_band_visible"]
                 and not cap_r["false_band_visible"]
                 and cap_r.get("osd_dim_true")
                 and nocap_r.get("osd_dim_true")
                 and cap_r.get("hover_at_true_band"))
    broken_sig = (not cap_r["true_band_visible"]
                  and cap_r["false_band_visible"]
                  and not cap_r.get("osd_dim_true"))
    verdict = ("FIXED" if fixed_sig else
               "BROKEN (regression reproduced)" if broken_sig else "MIXED")
    print(f"verdict: {verdict}")
    if args.json_out:
        results["verdict"] = verdict
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=2)
    if args.expect == "fixed":
        return 0 if fixed_sig else 1
    if args.expect == "broken":
        return 0 if broken_sig else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
