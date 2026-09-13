#!/usr/bin/env python3
"""Compile/run CPU regressions for current subtitle upload/allocation/timing code.

No GPU or full mpv build is used. Pass --cc clang (or --cc /path/to/zig),
--out DIR, and optionally --baseline COMMIT to check an old source revision.
"""

from pathlib import Path
import argparse
import json
import os
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--cc", default="cc")
ap.add_argument("--out", required=True)
ap.add_argument("--baseline")
args = ap.parse_args()
out = Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=True)
src = (ROOT / "video/out/vo_gpu_next.c").read_text(encoding="utf-8-sig")
if args.baseline:
    src = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", args.baseline + ":video/out/vo_gpu_next.c"],
        text=True,
        encoding="utf-8",
    )


def function(signature: str) -> str:
    a = src.index(signature)
    b = src.index("{", a) + 1
    depth = 1
    while depth:
        depth += (src[b] == "{") - (src[b] == "}")
        b += 1
    return src[a:b]


functions = "\n".join(
    function(s)
    for s in (
        "static bool gc_ensure(pl_gpu",
        "static void gc_prealloc_pools(",
        "static bool gc_staged_tex_upload(",
        "static void gc_flush_misses(",
    )
)
a = src.index("        struct pl_tex_transfer_params upload_params = {")
b = src.index('        stats_time_end(p->stats, "sub-upload");', a)
overlay = src[a:b]
a = src.index("    int64_t gdl = 0;")
b = src.index("    p->guard_deadline_ns = gdl;", a)
timing = src[a:b] + "\n    return gdl;"
template = (HERE / "stability_stubs.c").read_text(encoding="utf-8")
generated = (
    template.replace("/* PRODUCTION_FUNCTIONS */", functions)
    .replace("/* PRODUCTION_OVERLAY_UPLOAD */", overlay)
    .replace("/* PRODUCTION_DEADLINE */", timing)
)
(out / "stability.c").write_text(generated, encoding="utf-8")
cc = [args.cc]
if Path(args.cc).stem == "zig":
    cc.append("cc")
exe = out / ("stability.exe" if os.name == "nt" else "stability")
process_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
build = subprocess.run(
    cc + ["-std=c11", "-O0", str(out / "stability.c"), "-o", str(exe)],
    capture_output=True,
    text=True,
    timeout=60,
    creationflags=process_flags,
)
report = {
    "baseline": args.baseline,
    "method": "Original source functions/blocks, CPU resource and transport substitutes; no GPU.",
    "build_exit": build.returncode,
    "build_stdout": build.stdout,
    "build_stderr": build.stderr,
    "cases": [],
}
if build.returncode == 0:
    for name in (
        ["prealloc-ok", "prealloc-work-fail", "prealloc-edge-fail"]
        + [f"{kind}-{mode}" for kind in ("overlay", "staged", "glyph") for mode in range(5)]
        + ["timing"]
    ):
        p = subprocess.run(
            [str(exe), name],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=process_flags,
        )
        report["cases"].append(dict(name=name, exit=p.returncode, stdout=p.stdout, stderr=p.stderr))
(out / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
failures = [r["name"] for r in report["cases"] if r["exit"]]
print(
    json.dumps(
        {
            "build_exit": build.returncode,
            "completed_processes": len(report["cases"]),
            "failed": failures,
            "build_errors": build.stderr if build.returncode else "",
        },
    ),
)
raise SystemExit(1 if build.returncode or failures else 0)
