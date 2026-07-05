# TOOLS/subtest — subtitle-performance stats harness

Offline tooling for the "8K subtitles at zero drops" effort: parse mpv
`--dump-stats` captures, adjudicate them against a per-frame budget, and drive
an acceptance matrix of mpv configs.

Python 3.12, **stdlib only** — no third-party packages.

Four tools plus self-tests:

| file                | role                                                         |
|---------------------|--------------------------------------------------------------|
| `parse_stats.py`    | parse / report / assert on `--dump-stats` and mpv text logs  |
| `run_matrix.py`     | run (or emit for Windows) an acceptance config matrix        |
| `check_reference.py`| self-test: reproduce the postmortem reference table          |
| `abdiff.py`         | screenshot A/B **pixel** diff: prove two configs render identically |

`parse_stats.py`/`run_matrix.py`/`check_reference.py` measure *performance*.
`abdiff.py` verifies *pixel output*: it screenshots the actual gpu-next window
under two configs and diffs the PNGs, so "the GPU path is bit-identical to the
CPU path" becomes a mechanical, gateable check.

---

## parse_stats.py

Parses mpv `--dump-stats=FILE` output. Timestamps in the target corpus are in
**nanoseconds**; a `start`/`end` region's duration is `(end - start) / 1e6` ms.

Recognised line shapes (everything else is skipped):

```
<ts> start <name>            start of a timed region
<ts> end   <name>            end of a timed region
<ts> value <float> <name>    a scalar sample
<ts> event <name>            a singular/counter event   (tolerated)
<ts> signal <name>           a singular/counter event   (mpv's word)
<ts> <name>                  a singular/counter event   (bare, e.g. drop-vo)
```

### Reporting

```sh
# default: video-draw @ 41.7 ms budget
python3 parse_stats.py stats_off.txt

# one line: frames, mean, p50, p95, p99, max (ms), count+% over budget
#   video-draw: frames=664  mean=20.1  p50=14.9  p95=69.6  p99=78.3
#               max=189.4 ms  over(41.7ms)=93 (14.01%)

python3 parse_stats.py stats_off.txt --events video-draw,osd-update --table
python3 parse_stats.py stats_off.txt --all-events --anomalies
python3 parse_stats.py stats_off.txt --fps 60          # budget = 1000/60
python3 parse_stats.py stats_off.txt --budget 33.3
```

`--anomalies` reports unpaired starts/ends (pairing is a per-name LIFO stack;
leftover starts and orphan ends are counted, never silently dropped).

### Percentile convention (locked)

The postmortem reference table is reproduced **exactly** by the numpy
`nearest` method:

```
idx = round((n - 1) * q)        # 0-based index into the sorted durations
```

This is the *only* convention that simultaneously reproduces
`stats_off` p99 (sorted idx 656 = 78.3) and `stats_composite_on` p99
(sorted idx 638 = 95.1); floor/ceil/nearest-rank variants each miss one.
See the comment in `parse_stats.py` and don't change it without re-running
`check_reference.py`.

### Assertions (CI gating) — nonzero exit + message on failure

```sh
# fail if ANY video-draw pair exceeds 41.7 ms
python3 parse_stats.py stats_off.txt --assert-under video-draw=41.7

# counter (event/signal/bare) markers must be <= N
python3 parse_stats.py stats_off.txt --assert-count "drop-vo<=0"

# last value sample of a name must equal V (--value-tol to loosen)
python3 parse_stats.py stats_off.txt --assert-value "frame-duration==0.0417" --value-tol 0.001
```

All three flags are repeatable and can be combined; exit code is 1 if any
assertion fails, 0 otherwise.

### mpv text-log parsing

```sh
python3 parse_stats.py --log mpv-breakdown.log     # slowframe + render-ahead
```

* `[slowframe] osd-update=6.9 (subrender=4.3 capcomp=0.0 blur=0.0 other=2.6)
  render-submit=15.9 gpu-passes=0.0 ms` → count + worst offender per phase.
* `[render-ahead] served=804 empty=35 miss=2` → last line + max `miss`.

`--log` can be combined with a stats file argument (both are reported).

---

## run_matrix.py

Runs an acceptance matrix: `name → list of extra mpv options`.

### Config format

JSON (preferred) — options are a list or a whitespace string; an optional
`base` key is prepended to every config:

```json
{
  "base":          ["--sub-ass=yes"],
  "subs_off":      ["--sub-visibility=no"],
  "composite_on":  ["--vo=gpu-next", "--gpu-composite=yes"],
  "composite_off": "--vo=gpu-next --gpu-composite=no"
}
```

ini-ish fallback (one config per line, `name = opts` or `name : opts`, `#`
comments) is also accepted.

### Local mode (real mpv)

```sh
python3 run_matrix.py matrix.json \
    --video /path/to/torture.mkv --mpv /usr/bin/mpv \
    --frames 9000 --seek-at 120 --seek-to 240 \
    --out matrix-out --event video-draw --budget 41.7
```

Each config runs mpv with `--no-audio --dump-stats=matrix-out/stats_<name>.txt`
plus its options; a background thread fires an absolute seek (`--seek-at`
seconds in, to `--seek-to`) over `--input-ipc-server` to force a mid-run
subtitle/render discontinuity. Afterwards every capture is adjudicated with
`parse_stats` and printed as one combined table.

### Windows emit mode (no local mpv)

```sh
python3 run_matrix.py matrix.json --emit-windows winout \
    --mpv mpv.exe --video "D:\torture.mkv" --frames 9000
```

Writes `winout/run_tests.bat` + one `<name>.conf` per config. A human edits
`MPV`/`VIDEO` at the top of the `.bat`, runs it, and copies the resulting
`stats_*.txt` back for offline `parse_stats.py` adjudication. Because driving
mpv's IPC socket from a `.bat` is painful, the **seek is manual**: the `.bat`
header tells the operator to press RIGHT ARROW (or drag the seek bar) around
the halfway mark — or to set `SEEKSTART=--start=SECONDS` to sample the tail of
the file with no manual step.

---

## abdiff.py — screenshot A/B pixel-diff harness

Proves that two mpv configs produce **identical on-screen subtitle rendering**,
or quantifies the difference. This is the §2.B acceptance mechanism: with
`--blend-subtitles=no`, the CPU-bitmap and GPU-deferred paths share the same
final float blend, so a final-frame pixel diff of exactly **0** is a valid gate.

### How it works

For each config it launches mpv once over `--input-ipc-server=<sock>` (JSON
IPC), waits until the file is loaded and `seekable`, then for each requested PTS
does `seek <pts> absolute+exact`, waits for the `playback-restart` event (+ a
small settle), and `screenshot-to-file <out.png> window`. **Window** mode
captures the real gpu-next output — including GPU-composited subs — and the PNG
is lossless. The two screenshots are then diffed pixel-for-pixel.

### PNG decode (why it's built in)

ffmpeg/ffprobe are **not installed** on this box and the Python stdlib has no
PNG decoder, so `abdiff.py` ships a minimal one built on `zlib`. mpv's window
screenshots are **16-bit RGBA (rgba64), non-interlaced**; the decoder handles
8- and 16-bit RGB/RGBA and all five scanline filters. We pin
`--screenshot-png-filter=0` (filter None) so decode is effectively a memcpy.
Diffs are computed in native sample units (0..65535 at 16-bit) and also
reported normalised to an 8-bit (`/255`) scale so "1 LSB" reads intuitively
(1 LSB @8-bit ≈ 257 @16-bit).

### Determinism controls (baked into `BASE_FLAGS`)

Every run pins: `--geometry=1280x720 --window-scale=1` (window size ==
composite resolution), `--no-border --no-osc --no-osd-bar --osd-level=0
--no-input-default-bindings --force-window=yes --keep-open=yes --pause`,
`--dither=no --deband=no --icc-profile-auto=no --hr-seek=yes --sub-auto=no
--blend-subtitles=no`, `--screenshot-format=png --screenshot-png-compression=1
--screenshot-png-filter=0 --screenshot-high-bit-depth=yes
--screenshot-tag-colorspace=no`, plus the box requirements `--no-config
--gpu-sw=yes --vo=gpu-next --gpu-api=vulkan --no-audio` and a fixed `--sid`.
With these pinned, determinism is exact — same config twice gives maxdiff 0 on
both synthetic lavfi media and the real kobayashi mkv (verified, see below). No
extra pinning was needed.

> **NB:** give synthetic media as an `av://lavfi:...` URL. The `av://` protocol
> selects the demuxer itself; do **not** add a global `--demuxer-lavf-format=lavfi`
> — it also routes the external `.ass` sub file through lavfi and fails to open it.

### Single pair

```sh
# from TOOLS/subtest/ so the samples/ paths resolve
python3 abdiff.py --mpv ../../build/mpv \
    --media "av://lavfi:color=c=0x606060:s=1280x720:d=120:r=24" \
    --sub samples/blur.ass --pts 3,9,15 \
    --config-a="" --config-b="--sub-gpu-blur=yes" \
    --out out --tag blur_gpu
```

Output: per-PTS `maxdiff` (16-bit + `~/255`), differing-pixel count and `%px`,
and an overall `IDENTICAL`/`DIFFERS` verdict. Exit code is 0 iff identical
(so tests 1–3 are gateable). On any mismatch it saves `<tag>_a_<pts>.png`,
`<tag>_b_<pts>.png`, and an amplified `<tag>_diff_<pts>.png` into `--out`, plus
a machine-readable `<tag>_result.json`.

> **argparse gotcha:** a `--config-*` value that starts with `-` must use the
> `=` form (`--config-b="--sub-gpu-blur=yes"`), otherwise argparse treats it as
> an option. The JSON batch mode below has no such restriction.

### JSON batch mode (config matrix)

```sh
python3 abdiff.py --mpv ../../build/mpv --batch samples/selftest_batch.json --out out
```

The batch spec is an object with optional top-level defaults (`media`, `pts`,
`sub`, `amp`) and a `pairs` list; each pair may override any default and sets
`tag`, `config_a`, `config_b`:

```json
{
  "media": "av://lavfi:color=c=0x606060:s=1280x720:d=120:r=24",
  "pts": [3, 9, 15],
  "pairs": [
    {"tag": "det",       "sub": "samples/dialogue.ass", "config_a": "", "config_b": ""},
    {"tag": "gpu_blur",  "sub": "samples/blur.ass",     "config_a": "", "config_b": "--sub-gpu-blur=yes"}
  ]
}
```

Exit code is 1 if any pair is not `IDENTICAL`. A `batch_summary.json` is written
to `--out`. `samples/selftest_batch.json` is the committed self-test matrix.

### samples/

Ten hand-written `.ass` files (font: **DejaVu Sans**, resolved by fontconfig;
nothing embedded), each with a few events spanning 0–22 s so PTS 1–18 hit
visible content:

| sample               | exercises                                            |
|----------------------|------------------------------------------------------|
| `dialogue.ass`       | plain styled text (baseline)                         |
| `blur.ass`           | `\blur0.6`, `\blur3`, `\blur12`                       |
| `be.ass`             | `\be1`, `\be4`                                        |
| `clip_rect.ass`      | rectangular `\clip(x1,y1,x2,y2)`                      |
| `clip_vect.ass`      | vector-drawing `\clip(m … l …)`                       |
| `iclip.ass`          | inverse `\iclip`                                      |
| `kf.ass`             | `\kf` karaoke sweep (mid-line at each PTS)            |
| `shadow_layers.ass`  | `\shad` + overlapping layered events                 |
| `rotate.ass`         | `\frz` via `\t` animation                             |
| `fs300.ass`          | fontsize 300 + heavy `\blur8` glyphs                 |

### Self-tests (verification of this harness)

Run: `python3 abdiff.py --mpv ../../build/mpv --batch samples/selftest_batch.json
--out out` (llvmpipe/`--gpu-sw`; the real-mkv pair needs `/mnt/y/Video/kobayashi`).
Baseline results captured on this box — maxdiffs are in **16-bit** sample units:

| test | what                         | sample(s)                        | maxdiff | verdict   |
|------|------------------------------|----------------------------------|---------|-----------|
| 1    | same config ×2 (determinism) | dialogue.ass                     | 0       | IDENTICAL |
| 1    | same config ×2               | kobayashi mkv (pts 30/90/150)    | 0       | IDENTICAL |
| 2    | threads =1 vs =8             | dialogue / blur / shadow_layers  | 0       | IDENTICAL |
| 3    | `--sub-gpu-composite=yes`    | dialogue.ass                     | 0       | IDENTICAL |
| 3    | `--sub-gpu-composite=yes`    | shadow_layers.ass                | 0       | IDENTICAL |
| 3    | `--sub-gpu-composite=yes`    | clip_rect.ass                    | 0       | IDENTICAL |
| 4    | `--sub-gpu-blur=yes`         | blur.ass                         | 635 (~2.47/255)   | DIFFERS |
| 5    | `--sub-gpu-blur=yes`         | be.ass                           | 15584 (~60.64/255)| DIFFERS |

* Tests 1–3 are **0** (bit-exact): determinism holds with no extra pinning;
  `--sub-ass-render-threads` is bit-exact by design; and `--sub-gpu-composite`
  reproduces the CPU blend exactly on these samples.
* Test 4 documents the GPU box-blur **approximation** (small but nonzero) that
  this effort will replace.
* Test 5 is large because `\be` is silently dropped in blur-only deferral (the
  known bug B2 will fix); the amplified diff shows the difference concentrated
  on the un-softened glyph edges.

Note: `--sub-ass-render-threads=0` means *auto*, not serial — use `=1` for the
serial baseline in test 2.

---

## check_reference.py — the verification gate

```sh
python3 check_reference.py                 # corpus: /mnt/y/Video/kobayashi
python3 check_reference.py --corpus /some/dir
```

Runs `parse_stats` over the 13 reference captures and compares `video-draw`
stats against the postmortem table: **frames exact**, ms within ±0.15,
%over within ±0.1pp. Also spot-checks two worst stalls
(`stats_composite_async` max ≈ 8614.8 ms, `ep09stats3` max ≈ 972 ms). Prints a
PASS/FAIL row per file; exit 0 iff all pass. This is what pins the percentile
convention — run it after any change to `parse_stats.py`.

---

## Acceptance-test procedure (sketch)

1. **Scenes** — pick ≥3 torture scenes (dense typeset signs, karaoke with
   `\blur`/`\be`, full-screen composited sign) and cut/point mpv at ≥5 minutes
   of continuous playback that includes them.
2. **Seek** — every run performs ≥1 mid-run seek (local: `--seek-at`; Windows:
   the documented manual seek) to exercise the sub-render-ahead cache flush and
   re-fill path.
3. **Matrix** — at minimum: subtitles off (floor), current renderer on,
   candidate GPU/composite path. Capture `--dump-stats` for each.
4. **Gate** — a config passes only if, on every scene + the post-seek window:
   * `parse_stats.py stats_<cfg>.txt --assert-under video-draw=41.7`
     (no frame over the 24fps budget; use `--fps` for other rates), **and**
   * drop counters are zero, e.g.
     `parse_stats.py stats_<cfg>.txt --assert-count "drop-vo<=0"`.
5. Cross-check the text log with `parse_stats.py --log mpv.log` — slowframe
   count should trend to 0 and render-ahead `miss` should stay low.

---

## WP-D3 / Milestone-5 full-matrix accuracy gate

The Milestone-5 gate certifies the GPU subtitle pipeline (`--sub-gpu-raster=yes`,
`--sub-gpu-composite=yes`, render-ahead) is **visually lossless** vs the CPU
baseline on the lavapipe/`--gpu-sw` box:

* per-pixel `maxdiff <= 772/65535` (3/255) for **blur-carrying** content,
* **exactly 0** for everything else (coverage / karaoke / clips / shadows /
  drawings / `\be`),
* zero content errors, no temporal shimmer.

### One-command re-run

`samples/m5_matrix.json` is the whole abdiff-able gate (tests 1,2,3,5,6 + the
kobayashi dense sweep) in one batch. Run from `TOOLS/subtest/`:

```sh
python3 abdiff.py --mpv ~/d3bin/mpv --batch samples/m5_matrix.json --out out
```

abdiff's own exit code is 1 whenever *any* pair is not bit-identical, so it is
**not** the gate verdict on its own (blur pairs legitimately differ within the
772 bar). Adjudicate `out/batch_summary.json` against the two-tier bar with:

```sh
python3 - out/batch_summary.json <<'PY'
import json, sys
BLUR = ("blur","fs300","c3_draw_blur","combo_blur","draw_blur","bord4_blur8","_blur1","koba_sweep")
data = json.load(open(sys.argv[1]))
bad = []
for r in data:
    md = r["overall_maxdiff"]; tag = r["tag"]
    isblur = any(k in tag for k in BLUR)
    limit = 772 if isblur else 0
    ok = md <= limit
    # t3_* (fractional shadow) is a measured FINDING, not a pass/fail cell
    if tag.startswith("t3_"): ok = True   # reported separately, see FINDING F1
    print(f"{'ok ' if ok else 'BAD'} {tag:<28} max={md:<6} limit={limit}")
    if not ok: bad.append(tag)
print("GATE PASS" if not bad else f"GATE FAIL: {bad}")
PY
```

The committed batches `sweep_batch.json` (40 blur pairs) and
`regression_batch.json` (raster + composite) are also part of test 1; run them
the same way (`--batch sweep_batch.json` / `--batch regression_batch.json`).

### Counter health (test 7)

One full-length dump-stats play-through must keep the 9 integrity counters at 0:

```sh
~/d3bin/mpv "$KOBA" --no-config --gpu-sw=yes --vo=gpu-next --gpu-api=vulkan \
  --no-audio --sub-gpu-raster=yes --sub-render-ahead-frames=24 \
  --force-window=yes --end=120 --dump-stats=stats.txt
python3 parse_stats.py stats.txt \
  --assert-value gcache-flush==0 --assert-value atlas-overflow==0 \
  --assert-value staging-grow==0 --assert-value overlay-buf-grow==0 \
  --assert-value tex-realloc==0 --assert-value vo-alloc-after-first-frame==0 \
  --assert-value ra-miss==0 --assert-value ra-inline==0 --assert-value ra-stale==0
```

### Samples added by this package

* `samples/combo_*.ass` — 6 pairwise-combo torture cases (test 2).
* `samples/frac_shadow/fs{40,100,300}_{shad15,shad07,xyshad}_{plain,blur1}.ass`
  — 18 fractional-shadow cases (test 3). These are **static**, so the batch
  points them at a single PTS.

### Known FINDING (F1): fractional \shad is structurally displaced

The deferred/raster path rounds sub-pixel shadow offsets to whole pixels while
the CPU path does a sub-pixel bitmap shift. Every non-integer `\shad` /
`\xshad` / `\yshad` case (all 18 `frac_shadow` samples) differs from the CPU
baseline far above the 772 bar (plain up to ~27000/65535, i.e. the shadow is
offset by a whole pixel). Integer `\shad` at scale 1 remains exact. This is a
fix package, not an M5 pass. See the WP-D3 verdict for the full table.

Mechanism proven mechanically (2026-07-05, mpv 3250589251): raster `\shad1.5`
at fs100 is **bit-identical (maxdiff 0)** to CPU `\shad2` — the raster path
renders the round-to-nearest whole-pixel offset. The measured `raster1.5 vs
CPU1.5` diff (25724) is the expected fraction of a full 1-px shift
(`CPU1 vs CPU2` = 40998).

### Runtime caveats for kobayashi comparisons

* abdiff's default 0.3 s post-seek settle can be too short on this box when the
  two configs render at different speeds and the seek crosses a scene cut: the
  slower config's frame may not have presented yet and the screenshot captures
  the *previous* seek target (symptom: a full-frame diff, maxdiff ~65000 with
  >90 % of pixels differing, that vanishes on re-run). Same-config pairs are
  immune (verified maxdiff 0), and re-measuring pts 58 with a 2 s settle
  reproduced the sweep value (267) exactly, so the committed sweep numbers are
  clean — but treat any isolated full-frame diff as a capture race and re-run
  before calling it a rendering bug.
* Seeking *directly* into the middle of a subtitle line can leave the line
  unfed by the demuxer (both configs identically), so a first-seek PTS can
  legitimately read maxdiff 0 where a sequential sweep reads ~160: compare
  like-for-like seek sequences.
