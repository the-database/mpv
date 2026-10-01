# Subtitle correctness regressions

## Statistics overlay under deadline pressure

`test_osd_guard.py` checks the real D3D11 presentation path with GPU subtitles
and render-ahead enabled. It toggles the built-in stats display, refreshes it
frequently, then hides it. A 5 ms injected delay exceeds a 1 ms overlay budget.
Both default OSD messages and persistent script overlays must retain their
previous visible image during a missed deadline and disappear after being hidden.
The test reads the VO's retained-overlay counts: screenshot commands rebuild
overlays without a deadline and cannot detect this presentation-only failure.

Generate an original 24/48 fps VFR source, then run against each player build:

```sh
ffmpeg -f lavfi -i color=black:s=640x360:r=96:d=7 -vf "select='if(lt(mod(t,2),1),not(mod(n,4)),not(mod(n,2)))'" -fps_mode vfr -c:v ffv1 vfr.mkv
python TOOLS/subtest/test_osd_guard.py --mpv /path/to/mpv.exe --video vfr.mkv --out build/osd-guard
```

This Windows test requires Pillow through the shared ASS fixture module. It
does not change the physical display refresh rate or establish whether a
particular VFR file naturally misses deadlines at 23.976 Hz. Logs and results
are retained in the output directory. Use a new output directory for each run.

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

## ASS reinitialization with retained GPU outline frames

`test_ass_toggle_lifetime.py` covers a lifetime that the single-event style test
does not exercise reliably: an option change destroys the producing renderer
while render-ahead frames still pin its glyph data. Image refs keep the glyph
cache entries alive, but those entries still need the renderer's FreeType
library and font lock when their final reference is released.

```sh
python TOOLS/subtest/test_ass_toggle_lifetime.py --mpv /path/to/mpv.exe --video black.mkv --out build/subtitle-toggle-lifetime
```

Use the original 20-second 640x360 black video generated above. The test creates
its own ASS fixtures and runs six minimized D3D11 player processes: a single
event control, 320 positioned overlapping events with GPU raster/composition
and render-ahead enabled, the same dense fixture with either or both features
disabled, and the failing combination with only one libass rendering thread.
Each case requests 80 live force/no toggles and requires a clean exit. IPC,
shutdown and crash waits are bounded; only test-owned processes are terminated.
Logs, generated subtitles, exact exit codes and JSON results are retained.

The existing live-options/color tests should also pass: keeping retired
renderers alive must not prevent the new style from appearing. These synthetic
tests do not establish compatibility with every subtitle script or replace a
check of the original reported episode.
