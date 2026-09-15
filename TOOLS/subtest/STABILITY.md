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

## Live options and GPU subtitle color

The color regression compiles the production `libass_overlay_color()` and
`emit_composed_overlays()` functions with CPU data substitutes:

```sh
python3 TOOLS/subtest/test_overlay_color.py --cc clang --out build/subtitle-color
```

`--cc /path/to/zig` and `--baseline COMMIT` are supported. It checks explicit
100/203 subtitle luminance, PQ/HLG sources, automatic reference white, SDR and
missing-source defaults. Each case checks the main composed overlay and both
spill overlays. The live-203 case keeps the same textures and parts, as the
cached-compose path does. This verifies the metadata passed to the renderer;
it does not measure final HDR display brightness or intermediate GPU blending.

`test_hdr_subtitle_output.py` also tests the actual D3D11 renderer. It forces
PQ/BT.2020/1000-nit output for window-mode screenshots and compares separate
player runs with subtitle peaks 100 and 203, with GPU subtitles off and on.
It works on an SDR monitor because it checks the encoded output files; it
does not change Windows display settings or measure physical display nits.
For an original test source (requires FFmpeg with libx264):

```sh
ffmpeg -f lavfi -i color=black:s=640x360:r=24:d=20 -vf format=yuv420p10le -c:v libx264 -preset ultrafast -color_primaries bt2020 -color_trc smpte2084 -colorspace bt2020nc pq-black.mkv
python TOOLS/subtest/test_hdr_subtitle_output.py --mpv /path/to/mpv.exe --video pq-black.mkv --out build/subtitle-hdr
```

The test requires the same Windows/Pillow setup as the live-options check
below and rejects a source that the player does not report as PQ. It asserts
that 203 produces brighter subtitle pixels than 100 in both rendering paths.

The Windows D3D11 playback regression requires Python, Pillow and an mpv build
with the experimental subtitle options. Generate an original black video, then
run it against each executable being compared, using a new output directory:

```sh
ffmpeg -f lavfi -i color=black:s=640x360:r=24:d=20 -c:v ffv1 black.mkv
python TOOLS/subtest/test_live_ass_options.py --mpv /path/to/mpv.exe --video black.mkv --out build/subtitle-live
```

It generates its own ASS file, with a large green style and a small white forced
override. Four player runs combine render-ahead 0/60 with GPU raster/composition
off/on. Screenshot pixel checks verify both override states, changes while
paused and playing, and the selected state after a seek. Only the test's own
player processes are controlled and closed; IPC calls have timeouts. PNGs,
logs and `results.json` are retained. Any failed check returns a nonzero exit.

This is a focused correctness test, not a RIFE performance benchmark or coverage
of every ASS effect, output API, subtitle format and display configuration.
