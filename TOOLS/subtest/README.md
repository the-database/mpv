# TOOLS/subtest — subtitle-performance stats harness

Offline tooling for the "8K subtitles at zero drops" effort: parse mpv
`--dump-stats` captures, adjudicate them against a per-frame budget, and drive
an acceptance matrix of mpv configs.

Python 3.12, **stdlib only** — no third-party packages.

Three tools plus a self-test:

| file                | role                                                         |
|---------------------|--------------------------------------------------------------|
| `parse_stats.py`    | parse / report / assert on `--dump-stats` and mpv text logs  |
| `run_matrix.py`     | run (or emit for Windows) an acceptance config matrix        |
| `check_reference.py`| self-test: reproduce the postmortem reference table          |

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
