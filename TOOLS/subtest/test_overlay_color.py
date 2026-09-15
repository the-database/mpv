#!/usr/bin/env python3
"""Compile the production subtitle color helper and composed-overlay emitter.

Uses CPU-only libplacebo data substitutes, not a renderer or HDR display.
Pass --baseline COMMIT to verify the same assertions against an older revision.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def function(source, name):
    # Match a definition, excluding the helper's forward declaration.
    match = re.search(r"static [^;{}]*\b" + name + r"\([^;{}]*\)\s*\{", source)
    if not match:
        raise ValueError(f"Definition not found: {name}")
    end = match.end()
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cc", default="cc")
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline")
    args = ap.parse_args()
    source = (ROOT / "video/out/vo_gpu_next.c").read_text(encoding="utf-8-sig")
    if args.baseline:
        source = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", args.baseline + ":video/out/vo_gpu_next.c"],
            text=True, encoding="utf-8")
    helper = function(source, "libass_overlay_color")
    emitter = function(source, "emit_composed_overlays")
    # The old emitter has no ref_luma argument. Keep its implementation intact
    # and adapt only the harness call, so the baseline fails on actual metadata.
    has_ref_luma = "ref_luma" in emitter.split("{", 1)[0]
    generated = (HERE / "overlay_color_stubs.c").read_text(encoding="utf-8")
    generated = generated.replace("/* PRODUCTION_FUNCTIONS */", helper + "\n" + emitter)
    generated = generated.replace("/* REF_LUMA_ARGUMENT */", ", ref_luma" if has_ref_luma else "")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "overlay_color.c").write_text(generated, encoding="utf-8")
    compiler = [args.cc] + (["cc"] if Path(args.cc).stem == "zig" else [])
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW
    exe = out / ("overlay_color.exe" if os.name == "nt" else "overlay_color")
    command = compiler + ["-std=c11", "-O0", str(out / "overlay_color.c"), "-o", str(exe)]
    build = subprocess.run(command,
                           capture_output=True, text=True, timeout=60, creationflags=flags)
    if build.returncode:
        (out / "results.txt").write_text(build.stdout + build.stderr, encoding="utf-8")
        print(build.stdout + build.stderr, end="")
        return build.returncode
    result = subprocess.run([str(exe)], capture_output=True, text=True,
                            timeout=10, creationflags=flags)
    (out / "results.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    print(result.stdout + result.stderr, end="")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
