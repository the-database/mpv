#!/usr/bin/env python3
"""
parse_stats.py -- parser / adjudicator for mpv --dump-stats output.

Part of the subtitle-performance acceptance harness (TOOLS/subtest/).
Python 3.12 stdlib only. No third-party deps.

--------------------------------------------------------------------------
INPUT FORMAT (mpv --dump-stats=FILE)
--------------------------------------------------------------------------
Each line is  "<timestamp> <text>"  where <text> is what MP_STATS() wrote.
The timestamps in the corpus this tool targets are in NANOSECONDS; a
start/end duration is reported as (end - start) / 1e6 milliseconds.

Recognised line shapes (anything else is skipped gracefully):

    <ts> start <name>              start of a timed region
    <ts> end   <name>              end of a timed region
    <ts> value <float> <name>      a scalar sample
    <ts> event <name>              a singular/counter event  (tolerated form)
    <ts> signal <name>             a singular/counter event  (mpv's own word)
    <ts> <name>                    a singular/counter event  (bare form; e.g. "drop-vo")

Trailing "#comment" fields (added by mpv's msg.c) are ignored.

--------------------------------------------------------------------------
PERCENTILE CONVENTION  (locked empirically against the project corpus)
--------------------------------------------------------------------------
The reference table in the project postmortem is reproduced *exactly* (frame
counts exact, ms within +/-0.15, %over within +/-0.1pp) by the numpy
"nearest" method:

        idx = round((n - 1) * q)          # 0-based index into sorted data

i.e. linear position on the [0, n-1] rank scale, then round-half-to-even
(Python's built-in round()).  This is the ONLY convention that
simultaneously reproduces:
    stats_off  p99 -> sorted idx 656 (=78.3), where round(663*0.99)=656
    composite_on p99 -> sorted idx 638 (=95.1), where round(644*0.99)=638
Floor/ceil ("lower"/"higher"/nearest-rank) methods disagree on one or the
other, so they are ruled out.  Do not change this without re-running
check_reference.py.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# core parsing
# --------------------------------------------------------------------------

@dataclass
class Stats:
    """Everything extracted from one --dump-stats file."""
    # event name -> list of durations in ms (one per matched start/end pair)
    durations: dict[str, list[float]] = field(default_factory=dict)
    # event name -> count of singular ("event"/"signal"/bare) markers
    signals: dict[str, int] = field(default_factory=dict)
    # value name -> list of (ts_ns, value) samples, in file order
    values: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    # event name -> [unpaired_starts, unpaired_ends]
    anomalies: dict[str, list[int]] = field(default_factory=dict)
    lines_total: int = 0
    lines_skipped: int = 0


_KEYWORDS = {"start", "end", "value", "event", "signal"}


def _is_int(tok: str) -> bool:
    return tok.isdigit() or (tok[:1] == "-" and tok[1:].isdigit())


def parse_stats_file(path: str) -> Stats:
    """Parse a --dump-stats file into a Stats object.

    Pairing rule: a per-name LIFO stack.  On 'start' push the timestamp; on
    'end' pop the most recent 'start' of the same name and record the
    duration.  For well-formed (non-nested) data this is identical to FIFO
    ordering; nesting -- which is not expected -- resolves innermost-first.
    Leftover starts at EOF and ends with an empty stack are counted as
    anomalies rather than silently dropped.
    """
    st = Stats()
    open_stacks: dict[str, list[int]] = {}

    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            st.lines_total += 1
            # drop trailing comment then split on whitespace
            line = raw.split("#", 1)[0].split()
            if len(line) < 2 or not _is_int(line[0]):
                st.lines_skipped += 1
                continue
            ts = int(line[0])
            kind = line[1]

            if kind == "start" and len(line) >= 3:
                open_stacks.setdefault(line[2], []).append(ts)

            elif kind == "end" and len(line) >= 3:
                name = line[2]
                stack = open_stacks.get(name)
                if stack:
                    start_ts = stack.pop()
                    st.durations.setdefault(name, []).append((ts - start_ts) / 1e6)
                else:
                    st.anomalies.setdefault(name, [0, 0])[1] += 1

            elif kind == "value" and len(line) >= 4:
                try:
                    v = float(line[2])
                except ValueError:
                    st.lines_skipped += 1
                    continue
                st.values.setdefault(line[3], []).append((ts, v))

            elif kind in ("event", "signal") and len(line) >= 3:
                st.signals[line[2]] = st.signals.get(line[2], 0) + 1

            elif len(line) == 2:
                # bare "<ts> <name>" singular marker, e.g. "drop-vo"
                st.signals[line[1]] = st.signals.get(line[1], 0) + 1

            else:
                st.lines_skipped += 1

    # leftover starts == unpaired starts
    for name, stack in open_stacks.items():
        if stack:
            st.anomalies.setdefault(name, [0, 0])[0] += len(stack)

    return st


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def percentile(sorted_vals: list[float], q: float) -> float:
    """q in [0,1].  numpy 'nearest' method -- see module docstring."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    idx = round((n - 1) * q)
    return sorted_vals[idx]


@dataclass
class Report:
    event: str
    frames: int
    mean: float
    p50: float
    p95: float
    p99: float
    maximum: float
    over: int
    pct_over: float
    budget: float


def summarize(event: str, durations: list[float], budget: float) -> Report:
    n = len(durations)
    if n == 0:
        return Report(event, 0, 0, 0, 0, 0, 0, 0, 0.0, budget)
    s = sorted(durations)
    over = sum(1 for x in s if x > budget)
    return Report(
        event=event,
        frames=n,
        mean=sum(s) / n,
        p50=percentile(s, 0.50),
        p95=percentile(s, 0.95),
        p99=percentile(s, 0.99),
        maximum=s[-1],
        over=over,
        pct_over=100.0 * over / n,
        budget=budget,
    )


# --------------------------------------------------------------------------
# mpv text-log parsing (--log)
# --------------------------------------------------------------------------

_SLOW_RE = re.compile(
    r"\[slowframe\]\s+"
    r"osd-update=(?P<osd>[\d.]+)\s+"
    r"\(subrender=(?P<subrender>[\d.]+)\s+"
    r"capcomp=(?P<capcomp>[\d.]+)\s+"
    r"blur=(?P<blur>[\d.]+)\s+"
    r"other=(?P<other>[\d.]+)\)\s+"
    r"render-submit=(?P<submit>[\d.]+)\s+"
    r"gpu-passes=(?P<gpu>[\d.]+)\s+ms"
)

_SLOW_PHASES = [
    ("osd-update", "osd"),
    ("subrender", "subrender"),
    ("capcomp", "capcomp"),
    ("blur", "blur"),
    ("other", "other"),
    ("render-submit", "submit"),
    ("gpu-passes", "gpu"),
]

_RA_RE = re.compile(
    r"\[render-ahead\]\s+served=(?P<served>\d+)\s+empty=(?P<empty>\d+)\s+miss=(?P<miss>\d+)"
)


def parse_log_file(path: str) -> dict:
    """Extract [slowframe] and [render-ahead] telemetry from an mpv text log."""
    slow_count = 0
    # phase label -> (worst value, raw line)
    worst: dict[str, tuple[float, str]] = {}
    ra_count = 0
    ra_last: tuple[int, int, int] | None = None
    ra_max_miss = 0
    ra_worst_line = ""

    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            m = _SLOW_RE.search(raw)
            if m:
                slow_count += 1
                for label, grp in _SLOW_PHASES:
                    v = float(m.group(grp))
                    if label not in worst or v > worst[label][0]:
                        worst[label] = (v, raw.rstrip("\n"))
                continue
            m = _RA_RE.search(raw)
            if m:
                ra_count += 1
                served = int(m.group("served"))
                empty = int(m.group("empty"))
                miss = int(m.group("miss"))
                ra_last = (served, empty, miss)
                if miss > ra_max_miss:
                    ra_max_miss = miss
                    ra_worst_line = raw.rstrip("\n")

    return {
        "slow_count": slow_count,
        "slow_worst": worst,
        "ra_count": ra_count,
        "ra_last": ra_last,
        "ra_max_miss": ra_max_miss,
        "ra_worst_line": ra_worst_line,
    }


# --------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------

def print_report(rep: Report, file=sys.stdout) -> None:
    if rep.frames == 0:
        print(f"{rep.event}: no start/end pairs found", file=file)
        return
    print(
        f"{rep.event}: frames={rep.frames}  "
        f"mean={rep.mean:.1f}  p50={rep.p50:.1f}  p95={rep.p95:.1f}  "
        f"p99={rep.p99:.1f}  max={rep.maximum:.1f} ms  "
        f"over({rep.budget:.1f}ms)={rep.over} ({rep.pct_over:.2f}%)",
        file=file,
    )


def print_table(reports: list[Report], file=sys.stdout) -> None:
    hdr = f"{'event':<20} {'frames':>6} {'mean':>7} {'p50':>7} {'p95':>7} {'p99':>7} {'max':>9} {'over':>6} {'%over':>7}"
    print(hdr, file=file)
    print("-" * len(hdr), file=file)
    for r in reports:
        if r.frames == 0:
            print(f"{r.event:<20} {'--':>6}", file=file)
            continue
        print(
            f"{r.event:<20} {r.frames:>6} {r.mean:>7.1f} {r.p50:>7.1f} "
            f"{r.p95:>7.1f} {r.p99:>7.1f} {r.maximum:>9.1f} {r.over:>6} {r.pct_over:>6.2f}%",
            file=file,
        )


def print_log_summary(info: dict, file=sys.stdout) -> None:
    print(f"[slowframe] lines: {info['slow_count']}", file=file)
    if info["slow_worst"]:
        print("  worst by phase (ms):", file=file)
        for label, _grp in _SLOW_PHASES:
            if label in info["slow_worst"]:
                v, line = info["slow_worst"][label]
                print(f"    {label:<14} {v:>7.1f}   {line}", file=file)
    print(f"[render-ahead] lines: {info['ra_count']}", file=file)
    if info["ra_last"] is not None:
        s, e, m = info["ra_last"]
        print(f"  last: served={s} empty={e} miss={m}", file=file)
        print(f"  max miss={info['ra_max_miss']}   {info['ra_worst_line']}", file=file)


# --------------------------------------------------------------------------
# assertions (CI-style gating)
# --------------------------------------------------------------------------

def _fail(msg: str, failures: list[str]) -> None:
    failures.append(msg)
    print(f"ASSERT FAIL: {msg}", file=sys.stderr)


def run_assertions(st: Stats, args) -> int:
    """Return process exit code (0 == all passed)."""
    failures: list[str] = []

    for spec in args.assert_under or []:
        # EVENT=MS  -- fail if ANY pair exceeds MS
        if "=" not in spec:
            _fail(f"--assert-under bad spec {spec!r} (want EVENT=MS)", failures)
            continue
        ev, ms = spec.split("=", 1)
        ev = ev.strip()
        try:
            limit = float(ms)
        except ValueError:
            _fail(f"--assert-under bad number in {spec!r}", failures)
            continue
        durs = st.durations.get(ev, [])
        if not durs:
            _fail(f"--assert-under {spec}: no pairs for event {ev!r}", failures)
            continue
        mx = max(durs)
        if mx > limit:
            n_over = sum(1 for d in durs if d > limit)
            _fail(
                f"{ev}: max {mx:.1f} ms exceeds {limit:.1f} ms "
                f"({n_over}/{len(durs)} pairs over)",
                failures,
            )
        else:
            print(f"assert-under OK: {ev} max {mx:.1f} <= {limit:.1f} ms")

    for spec in args.assert_count or []:
        # EVENT<=N  -- counter (event/signal/bare) markers must be <= N
        m = re.match(r"^\s*(.+?)\s*<=\s*(\d+)\s*$", spec)
        if not m:
            _fail(f"--assert-count bad spec {spec!r} (want EVENT<=N)", failures)
            continue
        ev, n = m.group(1), int(m.group(2))
        got = st.signals.get(ev, 0)
        if got > n:
            _fail(f"{ev}: count {got} > {n}", failures)
        else:
            print(f"assert-count OK: {ev} {got} <= {n}")

    for spec in args.assert_value or []:
        # NAME==V  -- last value sample of NAME must equal V
        if "==" not in spec:
            _fail(f"--assert-value bad spec {spec!r} (want NAME==V)", failures)
            continue
        name, v = spec.split("==", 1)
        name = name.strip()
        try:
            want = float(v)
        except ValueError:
            _fail(f"--assert-value bad number in {spec!r}", failures)
            continue
        samples = st.values.get(name, [])
        if not samples:
            _fail(f"--assert-value {spec}: no value samples named {name!r}", failures)
            continue
        got = samples[-1][1]
        if abs(got - want) > args.value_tol:
            _fail(f"{name}: last value {got} != {want} (tol {args.value_tol})", failures)
        else:
            print(f"assert-value OK: {name} last {got} == {want}")

    if failures:
        print(f"\n{len(failures)} assertion(s) failed.", file=sys.stderr)
        return 1
    print("\nAll assertions passed.")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parse mpv --dump-stats output; report timings or gate on assertions.",
    )
    p.add_argument("stats", nargs="?", help="path to a --dump-stats file")
    p.add_argument("--event", default="video-draw",
                   help="event to report (default: video-draw)")
    p.add_argument("--events",
                   help="comma-separated list of events to report")
    p.add_argument("--all-events", action="store_true",
                   help="report every start/end paired event")
    p.add_argument("--budget", type=float, default=41.7,
                   help="over-budget threshold in ms (default 41.7)")
    p.add_argument("--fps", type=float,
                   help="set budget to 1000/FPS (overrides --budget)")
    p.add_argument("--table", action="store_true",
                   help="print an aligned table instead of one line per event")
    p.add_argument("--anomalies", action="store_true",
                   help="also print unpaired start/end counts")
    # log parsing
    p.add_argument("--log", help="parse an mpv text log for slowframe/render-ahead lines")
    # assertions
    p.add_argument("--assert-under", action="append", metavar="EVENT=MS",
                   help="fail if any pair of EVENT exceeds MS (repeatable)")
    p.add_argument("--assert-count", action="append", metavar="EVENT<=N",
                   help="fail if counter EVENT occurs more than N times (repeatable)")
    p.add_argument("--assert-value", action="append", metavar="NAME==V",
                   help="fail if last value sample of NAME != V (repeatable)")
    p.add_argument("--value-tol", type=float, default=1e-6,
                   help="tolerance for --assert-value (default 1e-6)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fps:
        args.budget = 1000.0 / args.fps

    if args.log:
        info = parse_log_file(args.log)
        print_log_summary(info)
        if not args.stats:
            return 0

    if not args.stats:
        print("error: no stats file given (and no --log)", file=sys.stderr)
        return 2

    st = parse_stats_file(args.stats)

    # assertion mode short-circuits normal reporting
    if args.assert_under or args.assert_count or args.assert_value:
        return run_assertions(st, args)

    # choose events
    if args.all_events:
        events = sorted(st.durations.keys())
    elif args.events:
        events = [e.strip() for e in args.events.split(",") if e.strip()]
    else:
        events = [args.event]

    reports = [summarize(ev, st.durations.get(ev, []), args.budget) for ev in events]

    if args.table or len(reports) > 1 or args.all_events:
        print_table(reports)
    else:
        for r in reports:
            print_report(r)

    if args.anomalies and st.anomalies:
        print("\nanomalies (unpaired start/end):")
        for name, (us, ue) in sorted(st.anomalies.items()):
            print(f"  {name}: {us} unpaired start(s), {ue} unpaired end(s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
