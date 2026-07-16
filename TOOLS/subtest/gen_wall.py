#!/usr/bin/env python3
"""WP-H10 synthetic: ep09-shaped dense-typeset wall for the transient-store
chain gates.

Demand shape mirrors the round-5 rig evidence (h10_repro):
  - giant glyphs per frame, each > the gc_cacheable cap -> per-frame
    transient raster (rig: 84 x ~3346x1542);
  - many bordered+blurred text runs whose region-layer area far exceeds the
    result atlas -> sustained result_tex spill (rig: 229/250 layers);
  - everything animated (\\move) so change_id churns every frame and compose
    reuse can never mask the demand (rig: compose-reuse delta == 0 in-storm).

Presets:
  big    5328x3000 window, auto atlas 16384 (link0 = 16384x16384 on llvmpipe).
         Giants fs620 fscx260 (~2.8kx1.9k = 5.4 Mpx > 4.19 Mpx cap); demand
         ~2-3 links. The "max feasible geometry" gate; ~15-25 s/frame llvmpipe.
  small  1920x1080 window, --sub-glyph-atlas-size=4096 (link0 = 4096x4096).
         Giants fs520 (~0.45 Mpx > 0.26 Mpx cap); 120 runs fs120 so region
         demand also beats the display-derived result_tex. Fast (~1 s/frame):
         the x3 seek/ramp/retire/sanitizer workhorse.

Timeline (media time):
   0.0-16.0   light dialogue (cheap frames; prefill/estimator territory)
  20.0-26.0   THE WALL (dense typeset)
  26.0-40.0   light again (recovery / retire-path territory)
"""
import sys

PRESETS = {
    "big":   dict(n_giant=26, n_runs=90,  giant_fs=620, giant_fscx=260,
                  run_fs=144, run_blur=6, giant_blur=10),
    "small": dict(n_giant=26, n_runs=120, giant_fs=520, giant_fscx=240,
                  run_fs=120, run_blur=4, giant_blur=6),
}
WALL_T0, WALL_T1 = 20.0, 26.0

HDR = """[Script Info]
Title: h10 dense wall ({preset})
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Dlg,DejaVu Sans,48,&H00FFFFFF,&H000000FF,&H00101010,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,40,40,30,1
Style: Wall,DejaVu Sans,{run_fs},&H00E0E0E0,&H000000FF,&H00202020,&H00000000,0,0,0,0,100,100,0,0,1,4,0,7,0,0,0,1
Style: Giant,DejaVu Sans,{giant_fs},&H60FFC080,&H000000FF,&H00301010,&H00000000,0,0,0,0,{giant_fscx},100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

def ts(t):
    h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def main(preset, out):
    P = PRESETS[preset]
    ev = []
    for i in range(8):
        t0 = i * 2.0
        ev.append(f"Dialogue: 0,{ts(t0)},{ts(t0+1.8)},Dlg,,0,0,0,,Quiet line {i} before the wall.")
    for i in range(7):
        t0 = 26.0 + i * 2.0
        ev.append(f"Dialogue: 0,{ts(t0)},{ts(t0+1.8)},Dlg,,0,0,0,,Quiet line {i} after the wall.")

    t0s, t1s = ts(WALL_T0), ts(WALL_T1)
    blocks = "█▓▒░"
    for i in range(P["n_giant"]):
        gx = (i % 8) * 230 + 10
        gy = (i // 8) * 200 + 20
        ch = blocks[i % len(blocks)]
        ev.append(
            f"Dialogue: 1,{t0s},{t1s},Giant,,0,0,0,,"
            f"{{\\move({gx},{gy},{gx+8},{gy})\\blur{P['giant_blur']}\\alpha&H90&}}{ch}")
    words = ["DENSETYPESET", "WALLOFSIGNSX", "KANJIKANJIKA", "SUBTITLEWALL"]
    for i in range(P["n_runs"]):
        rx = (i % 10) * 190 + 5
        ry = (i * 37) % 900 + 30
        w = words[i % len(words)]
        c = ["&H00FFD0D0&", "&H00D0FFD0&", "&H00D0D0FF&", "&H00FFFFD0&"][i % 4]
        ev.append(
            f"Dialogue: 2,{t0s},{t1s},Wall,,0,0,0,,"
            f"{{\\move({rx},{ry},{rx+6},{ry+2})\\blur{P['run_blur']}\\bord4\\1c{c}}}{w}")
    with open(out, "w") as f:
        f.write(HDR.format(preset=preset, **P))
        f.write("\n".join(ev) + "\n")
    print(f"wrote {out} [{preset}]: {len(ev)} events "
          f"({P['n_giant']} giants + {P['n_runs']} runs in [{WALL_T0},{WALL_T1}))")

if __name__ == "__main__":
    preset = sys.argv[1] if len(sys.argv) > 1 else "big"
    out = sys.argv[2] if len(sys.argv) > 2 else f"wall_{preset}.ass"
    main(preset, out)
