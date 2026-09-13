# Subtitle correctness regressions

Run from the repository root:

```sh
python3 TOOLS/subtest/test_stability.py --cc clang --out build/subtitle-stability
```

`--cc` also accepts a path to `zig`; the script invokes `zig cc` automatically.
Add `--baseline COMMIT` to test an older revision against the same assertions.
Python 3 and a C11 compiler are required. A failed assertion or compilation
returns a nonzero exit status. Details are written to `results.json` in the
output directory.

The script extracts the current production allocation and subtitle-upload
functions, the ordinary-overlay upload block, and the automatic deadline block
from `video/out/vo_gpu_next.c`. `stability_stubs.c` supplies CPU resource and
transfer substitutes. The extracted code is compiled and run in separate
processes so allocation failures in an old revision cannot stop later cases.

Nineteen processes check 25 scenarios:

- Successful pool preallocation and failure of each work/edge texture allocation.
- Ordinary overlays, staged textures and glyph batches with buffer transfers
  unavailable, existing buffers, allocation failure and successful allocation.
- Nanosecond frame duration, display-sync seconds, playback speed, approximate
  frame duration, fallback, disabled and explicit presentation deadlines.

These tests exercise control flow and units without a GPU. They do not replace
a full mpv build or playback tests on D3D11 and Vulkan, including subtitle
appearance and performance.
