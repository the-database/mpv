#!/usr/bin/env python3
"""
abdiff.py -- screenshot-based A/B pixel-diff harness for subtitle rendering.

Purpose
-------
Prove that two mpv configurations produce *identical* on-screen subtitle
rendering (or quantify how they differ) by driving mpv over its JSON IPC
socket, seeking to fixed presentation timestamps, capturing lossless
window-mode PNG screenshots, and comparing them pixel-for-pixel.

This is the acceptance mechanism for the deferred-GPU subtitle effort:
with `--blend-subtitles=no`, the CPU-bitmap and GPU-deferred paths share the
same final float blend, so a final-frame diff of exactly 0 is a valid gate.

Determinism controls
---------------------
Every run pins a fixed window size (=> composite resolution), disables
dithering/debanding/ICC, forces exact hr-seek, and writes lossless PNG with
filter type 0 (None) so the decode below is trivial and exact.  The two configs
under test are appended verbatim to this common base.

PNG decode
----------
ffmpeg/ffprobe are not present on this box, and the Python stdlib has no PNG
decoder, so a minimal one is implemented here on top of `zlib`.  mpv's
`image_writer` (png encoder) emits non-interlaced RGB/RGBA at 8 or 16 bit; in
window mode with high-bit-depth screenshots it is 16-bit RGBA (rgba64).  All of
those cases are handled.  We pin `--screenshot-png-filter=0`, so every scanline
uses filter None and reconstruction is a memcpy; the general per-row unfilter
(Sub/Up/Average/Paeth) is implemented too for robustness.

Python 3.12, stdlib only.
"""

from __future__ import annotations

import argparse
import array
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import zlib


# ==========================================================================
# minimal PNG reader (zlib-based)
# ==========================================================================

_PNG_SIG = b"\x89PNG\r\n\x1a\n"

# colour type -> channels
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


class Image:
    """Decoded image: row-major, no filter bytes, native sample order (ints)."""

    __slots__ = ("w", "h", "channels", "bitdepth", "maxval", "raw")

    def __init__(self, w, h, channels, bitdepth, raw):
        self.w = w
        self.h = h
        self.channels = channels
        self.bitdepth = bitdepth
        self.maxval = (1 << bitdepth) - 1
        self.raw = raw  # bytes: h * w * channels * (bitdepth//8)

    @property
    def row_bytes(self) -> int:
        return self.w * self.channels * (self.bitdepth // 8)

    def row_samples(self, y: int) -> array.array:
        """Return the samples of scanline y as an array of ints."""
        rb = self.row_bytes
        chunk = self.raw[y * rb:(y + 1) * rb]
        if self.bitdepth == 8:
            return array.array("B", chunk)
        a = array.array("H")
        a.frombytes(chunk)
        if sys.byteorder == "little":  # PNG samples are big-endian
            a.byteswap()
        return a


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def read_png(path: str) -> Image:
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:8] != _PNG_SIG:
        raise ValueError(f"{path}: not a PNG (bad signature)")

    pos = 8
    width = height = bitdepth = colortype = interlace = None
    idat = bytearray()
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length  # 4 len + 4 type + body + 4 crc
        if ctype == b"IHDR":
            (width, height, bitdepth, colortype, _comp, _filt, interlace) = \
                struct.unpack(">IIBBBBB", body)
        elif ctype == b"IDAT":
            idat += body
        elif ctype == b"IEND":
            break

    if width is None:
        raise ValueError(f"{path}: no IHDR")
    if interlace != 0:
        raise ValueError(f"{path}: interlaced PNG not supported")
    if colortype not in _CHANNELS:
        raise ValueError(f"{path}: unsupported colour type {colortype}")
    channels = _CHANNELS[colortype]
    if colortype == 3:
        raise ValueError(f"{path}: palette PNG not supported")
    if bitdepth not in (8, 16):
        raise ValueError(f"{path}: unsupported bit depth {bitdepth}")

    raw = zlib.decompress(bytes(idat))
    bpp = channels * (bitdepth // 8)
    stride = width * bpp
    out = bytearray(height * stride)

    src = 0
    prev = bytearray(stride)  # scanline above (starts all zero)
    for y in range(height):
        ftype = raw[src]
        src += 1
        line = bytearray(raw[src:src + stride])
        src += stride
        if ftype == 0:
            pass  # None -- fast path (what we pin mpv to emit)
        elif ftype == 1:  # Sub
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 0xFF
        elif ftype == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ftype == 3:  # Average
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:  # Paeth
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                c = prev[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + _paeth(a, prev[i], c)) & 0xFF
        else:
            raise ValueError(f"{path}: bad filter type {ftype} on row {y}")
        out[y * stride:(y + 1) * stride] = line
        prev = line

    return Image(width, height, channels, bitdepth, bytes(out))


# ==========================================================================
# minimal PNG writer (8-bit RGB, filter None) -- for amplified diff output
# ==========================================================================

def write_png_rgb(path: str, w: int, h: int, rgb: bytes) -> None:
    def chunk(tag: bytes, body: bytes) -> bytes:
        return (struct.pack(">I", len(body)) + tag + body
                + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit, RGB, none
    stride = w * 3
    raw = bytearray()
    for y in range(h):
        raw.append(0)  # filter None
        raw += rgb[y * stride:(y + 1) * stride]
    comp = zlib.compress(bytes(raw), 6)
    with open(path, "wb") as fh:
        fh.write(_PNG_SIG)
        fh.write(chunk(b"IHDR", ihdr))
        fh.write(chunk(b"IDAT", comp))
        fh.write(chunk(b"IEND", b""))


# ==========================================================================
# diff engine
# ==========================================================================

class DiffResult:
    __slots__ = ("maxdiff", "ndiff", "total_px", "maxval", "channels", "w", "h")

    def __init__(self, maxdiff, ndiff, total_px, maxval, channels, w, h):
        self.maxdiff = maxdiff        # native sample units
        self.ndiff = ndiff            # differing pixels
        self.total_px = total_px
        self.maxval = maxval
        self.channels = channels
        self.w = w
        self.h = h

    @property
    def maxdiff8(self) -> float:
        """maxdiff normalised to an 8-bit (0..255) scale."""
        return self.maxdiff * 255.0 / self.maxval

    @property
    def pct(self) -> float:
        return 100.0 * self.ndiff / self.total_px if self.total_px else 0.0

    @property
    def identical(self) -> bool:
        return self.maxdiff == 0


def diff_images(a: Image, b: Image) -> DiffResult:
    if (a.w, a.h, a.channels, a.bitdepth) != (b.w, b.h, b.channels, b.bitdepth):
        raise ValueError(
            f"image geometry mismatch: {a.w}x{a.h}c{a.channels}b{a.bitdepth} "
            f"vs {b.w}x{b.h}c{b.channels}b{b.bitdepth}")

    total_px = a.w * a.h
    # fast path: byte-identical (the common case for the ==0 gate)
    if a.raw == b.raw:
        return DiffResult(0, 0, total_px, a.maxval, a.channels, a.w, a.h)

    ch = a.channels
    rb = a.row_bytes
    maxdiff = 0
    ndiff = 0
    for y in range(a.h):
        ar = a.raw[y * rb:(y + 1) * rb]
        br = b.raw[y * rb:(y + 1) * rb]
        if ar == br:
            continue
        sa = a.row_samples(y)
        sb = b.row_samples(y)
        for px in range(a.w):
            base = px * ch
            pd = 0
            for k in range(ch):
                d = sa[base + k] - sb[base + k]
                if d < 0:
                    d = -d
                if d > pd:
                    pd = d
            if pd:
                ndiff += 1
                if pd > maxdiff:
                    maxdiff = pd
    return DiffResult(maxdiff, ndiff, total_px, a.maxval, ch, a.w, a.h)


def write_diff_png(path: str, a: Image, b: Image, amp: int) -> None:
    """Amplified grayscale diff: v = min(255, d8 * amp) per pixel."""
    ch = a.channels
    rb = a.row_bytes
    scale = 255.0 / a.maxval
    rgb = bytearray(a.w * a.h * 3)
    for y in range(a.h):
        ar = a.raw[y * rb:(y + 1) * rb]
        br = b.raw[y * rb:(y + 1) * rb]
        if ar == br:
            continue
        sa = a.row_samples(y)
        sb = b.row_samples(y)
        for px in range(a.w):
            base = px * ch
            pd = 0
            for k in range(ch):
                d = sa[base + k] - sb[base + k]
                if d < 0:
                    d = -d
                if d > pd:
                    pd = d
            if pd:
                v = int(pd * scale * amp)
                if v > 255:
                    v = 255
                o = (y * a.w + px) * 3
                rgb[o] = v
                rgb[o + 1] = v
                rgb[o + 2] = v
    write_png_rgb(path, a.w, a.h, bytes(rgb))


# ==========================================================================
# mpv IPC driver
# ==========================================================================

# pinned determinism / window controls, shared by every run
BASE_FLAGS = [
    "--no-config",
    "--gpu-sw=yes",
    "--vo=gpu-next",
    "--gpu-api=vulkan",
    "--no-audio",
    "--geometry=1280x720",
    "--window-scale=1",
    "--no-border",
    "--no-osc",
    "--no-osd-bar",
    "--osd-level=0",
    "--no-input-default-bindings",
    "--force-window=yes",
    "--keep-open=yes",
    "--pause",
    "--dither=no",
    "--deband=no",
    "--icc-profile-auto=no",
    "--hr-seek=yes",
    "--sub-auto=no",
    # §2.B premise: with blend-subtitles=no the CPU-bitmap and GPU-deferred
    # paths share the same final float blend (also gpu-next's default).
    "--blend-subtitles=no",
    "--screenshot-format=png",
    "--screenshot-png-compression=1",
    "--screenshot-png-filter=0",
    "--screenshot-high-bit-depth=yes",
    "--screenshot-tag-colorspace=no",
]


class MpvIPC:
    def __init__(self, path: str, timeout: float = 30.0):
        deadline = time.time() + timeout
        self.sock = None
        while time.time() < deadline:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(path)
                self.sock = s
                break
            except OSError:
                time.sleep(0.1)
        if self.sock is None:
            raise RuntimeError(f"could not connect to mpv IPC socket {path}")
        self.sock.settimeout(1.0)
        self.buf = b""
        self.events: list[dict] = []
        self._rid = 0

    def _read_line(self, deadline: float):
        while b"\n" not in self.buf:
            if time.time() > deadline:
                return None
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                return None
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        try:
            return json.loads(line.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            return {}

    def command(self, cmd: list, timeout: float = 30.0):
        ok, data = self.command_raw(cmd, timeout)
        if not ok:
            raise RuntimeError(f"mpv command {cmd!r} -> {data}")
        return data

    def command_raw(self, cmd: list, timeout: float = 30.0):
        """Like command() but returns (ok, data_or_error) instead of raising."""
        self._rid += 1
        rid = self._rid
        payload = json.dumps({"command": cmd, "request_id": rid}) + "\n"
        self.sock.sendall(payload.encode())
        deadline = time.time() + timeout
        while True:
            msg = self._read_line(deadline)
            if msg is None:
                return False, "timeout"
            if "event" in msg:
                self.events.append(msg)
                continue
            if msg.get("request_id") == rid:
                if msg.get("error") != "success":
                    return False, msg.get("error")
                return True, msg.get("data")

    def wait_ready(self, timeout: float = 40.0) -> None:
        """Wait until a file is loaded and seekable (race-free vs file-loaded)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            ok, val = self.command_raw(["get_property", "seekable"], timeout=5.0)
            if ok and val:
                return
            time.sleep(0.1)
        raise RuntimeError("mpv never became seekable (see log)")

    def wait_event(self, name: str, timeout: float = 30.0) -> bool:
        for i, ev in enumerate(self.events):
            if ev.get("event") == name:
                del self.events[i]
                return True
        deadline = time.time() + timeout
        while True:
            msg = self._read_line(deadline)
            if msg is None:
                return False
            if msg.get("event") == name:
                return True
            if "event" in msg:
                self.events.append(msg)

    def drain_event(self, name: str) -> None:
        self.events = [e for e in self.events if e.get("event") != name]

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def run_config(mpv: str, media: str, sub: str | None, extra: list[str],
               pts_list: list[float], out_shots: list[str], logpath: str,
               sid: int = 1, settle: float = 0.30) -> None:
    """Launch mpv once, screenshot each pts into out_shots[i]."""
    sock_path = os.path.join(
        tempfile.gettempdir(), f"abdiff-{os.getpid()}-{time.time_ns()}.sock")
    # NB: media should be given as an av:// URL (e.g. av://lavfi:color=...); the
    # av:// protocol selects the demuxer itself.  We must NOT pass a global
    # --demuxer-lavf-format=lavfi: it also routes the external .ass sub file
    # through the lavfi demuxer, which then fails to open it as a filtergraph.
    cmd = [mpv, media] + BASE_FLAGS + [
        f"--input-ipc-server={sock_path}",
        f"--sid={sid}",
    ]
    if sub:
        cmd.append(f"--sub-file={sub}")
    cmd += extra

    env = dict(os.environ)
    env.setdefault("DISPLAY", ":0")

    logf = open(logpath, "wb")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    ipc = None
    try:
        ipc = MpvIPC(sock_path, timeout=40.0)
        ipc.wait_ready(timeout=40.0)
        ipc.command(["set_property", "pause", True])
        for pts, shot in zip(pts_list, out_shots):
            ipc.drain_event("playback-restart")
            ipc.command(["seek", pts, "absolute+exact"])
            if not ipc.wait_event("playback-restart", timeout=30.0):
                raise RuntimeError(f"no playback-restart after seek to {pts}")
            time.sleep(settle)
            ipc.command(["screenshot-to-file", shot, "window"], timeout=60.0)
        ipc.command(["quit"])
    finally:
        if ipc is not None:
            ipc.close()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        logf.close()
        try:
            os.unlink(sock_path)
        except OSError:
            pass


# ==========================================================================
# pair comparison
# ==========================================================================

def fmt_pts(p: float) -> str:
    return f"{p:g}".replace(".", "_")


def compare_pair(mpv: str, media: str, sub: str | None, pts_list: list[float],
                 config_a: list[str], config_b: list[str], out: str,
                 tag: str, amp: int) -> dict:
    os.makedirs(out, exist_ok=True)
    work = tempfile.mkdtemp(prefix=f"abdiff-{tag}-")

    a_shots = [os.path.join(work, f"a_{fmt_pts(p)}.png") for p in pts_list]
    b_shots = [os.path.join(work, f"b_{fmt_pts(p)}.png") for p in pts_list]
    log_a = os.path.join(out, f"{tag}_a.log")
    log_b = os.path.join(out, f"{tag}_b.log")

    run_config(mpv, media, sub, config_a, pts_list, a_shots, log_a)
    run_config(mpv, media, sub, config_b, pts_list, b_shots, log_b)

    rows = []
    overall_max = 0
    any_diff = False
    for p, pa, pb in zip(pts_list, a_shots, b_shots):
        if not (os.path.exists(pa) and os.path.exists(pb)):
            rows.append({"pts": p, "error": "missing screenshot"})
            any_diff = True
            continue
        ia = read_png(pa)
        ib = read_png(pb)
        d = diff_images(ia, ib)
        row = {
            "pts": p,
            "maxdiff": d.maxdiff,
            "maxdiff8": round(d.maxdiff8, 3),
            "ndiff": d.ndiff,
            "total_px": d.total_px,
            "pct": round(d.pct, 4),
            "bitdepth": ia.bitdepth,
            "identical": d.identical,
        }
        if not d.identical:
            any_diff = True
            overall_max = max(overall_max, d.maxdiff)
            tp = fmt_pts(p)
            sa = os.path.join(out, f"{tag}_a_{tp}.png")
            sb = os.path.join(out, f"{tag}_b_{tp}.png")
            sd = os.path.join(out, f"{tag}_diff_{tp}.png")
            shutil.copyfile(pa, sa)
            shutil.copyfile(pb, sb)
            write_diff_png(sd, ia, ib, amp)
            row["saved"] = {"a": sa, "b": sb, "diff": sd}
        rows.append(row)

    shutil.rmtree(work, ignore_errors=True)

    verdict = "DIFFERS" if any_diff else "IDENTICAL"
    result = {
        "tag": tag,
        "media": media,
        "sub": sub,
        "config_a": " ".join(config_a),
        "config_b": " ".join(config_b),
        "verdict": verdict,
        "overall_maxdiff": overall_max,
        "rows": rows,
    }
    with open(os.path.join(out, f"{tag}_result.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    return result


def print_result(r: dict) -> None:
    print(f"\n=== [{r['tag']}] {r['verdict']} ===")
    print(f"  media : {r['media']}")
    if r["sub"]:
        print(f"  sub   : {r['sub']}")
    print(f"  A: {r['config_a'] or '(none)'}")
    print(f"  B: {r['config_b'] or '(none)'}")
    print(f"  {'pts':>7}  {'maxdiff':>9}  {'~/255':>7}  {'diffpx':>9}  "
          f"{'%px':>8}  verdict")
    for row in r["rows"]:
        if "error" in row:
            print(f"  {row['pts']:>7}  {'ERR':>9}  {row['error']}")
            continue
        verd = "IDENTICAL" if row["identical"] else "DIFFERS"
        print(f"  {row['pts']:>7.3f}  {row['maxdiff']:>9}  "
              f"{row['maxdiff8']:>7.2f}  {row['ndiff']:>9}  "
              f"{row['pct']:>8.4f}  {verd}")
    print(f"  -> {r['verdict']}  (overall maxdiff={r['overall_maxdiff']}, "
          f"~{r['overall_maxdiff'] * 255.0 / 65535:.2f}/255 @16bit)")


# ==========================================================================
# CLI
# ==========================================================================

def parse_pts(s) -> list[float]:
    if isinstance(s, list):
        return [float(x) for x in s]
    return [float(x) for x in str(s).split(",") if x.strip()]


def split_cfg(s) -> list[str]:
    if isinstance(s, list):
        return [str(x) for x in s]
    return str(s).split()


def run_batch(path: str, args) -> int:
    with open(path) as fh:
        spec = json.load(fh)
    d_media = spec.get("media")
    d_pts = spec.get("pts")
    d_sub = spec.get("sub")
    d_amp = spec.get("amp", args.amp)
    results = []
    exit_code = 0
    for i, pair in enumerate(spec["pairs"]):
        tag = pair.get("tag", f"pair{i}")
        media = pair.get("media", d_media)
        pts = parse_pts(pair.get("pts", d_pts))
        sub = pair.get("sub", d_sub)
        r = compare_pair(
            args.mpv, media, sub, pts,
            split_cfg(pair.get("config_a", "")),
            split_cfg(pair.get("config_b", "")),
            args.out, tag, pair.get("amp", d_amp))
        print_result(r)
        results.append(r)
        if r["verdict"] != "IDENTICAL":
            exit_code = 1
    with open(os.path.join(args.out, "batch_summary.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("\n=== batch summary ===")
    for r in results:
        print(f"  {r['tag']:<24} {r['verdict']:<10} "
              f"maxdiff={r['overall_maxdiff']}")
    return exit_code


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Screenshot A/B pixel-diff harness for subtitle rendering.")
    p.add_argument("--mpv", required=True, help="path to mpv binary")
    p.add_argument("--out", default="abdiff-out", help="output directory")
    p.add_argument("--amp", type=int, default=32,
                   help="amplification factor for saved diff PNGs (default 32)")

    p.add_argument("--batch", help="JSON batch spec (list of A/B pairs)")

    p.add_argument("--media", help="media file or lavfi URL")
    p.add_argument("--sub", help="external .ass subtitle file")
    p.add_argument("--pts", help="comma-separated presentation timestamps")
    p.add_argument("--config-a", default="", help="extra mpv flags for A")
    p.add_argument("--config-b", default="", help="extra mpv flags for B")
    p.add_argument("--tag", default="ab", help="label for this pair")

    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    if args.batch:
        return run_batch(args.batch, args)

    if not (args.media and args.pts):
        p.error("--media and --pts are required (or use --batch)")
    r = compare_pair(
        args.mpv, args.media, args.sub, parse_pts(args.pts),
        split_cfg(args.config_a), split_cfg(args.config_b),
        args.out, args.tag, args.amp)
    print_result(r)
    return 0 if r["verdict"] == "IDENTICAL" else 1


if __name__ == "__main__":
    sys.exit(main())
