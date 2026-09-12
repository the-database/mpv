# Building the fork locally

For CI and the shipped builds see [`animejanai-build-ci.md`](animejanai-build-ci.md). For what the fork changes see
[`../CLAUDE.md`](../CLAUDE.md).

> **Paths in this document.** `<mpv-checkout>` is your clone of this repository,
> `<libass-src>` / `<libass-prefix>` are where you put the libass fork's source and install
> prefix, and `%AJI_WIN%` / `$AJI_WIN` is the separate engine-testing working area described at
> the end. Substitute your own locations.

Two environments are in use: **MSYS2 UCRT64** on Windows and **WSL (Ubuntu)**. MSYS2 is the faster loop for compile-checking the fork's code; WSL is what the
sanitizer builds were done in.

## The one thing to get right: which libass you link

The five `HAVE_ASS_*` features are auto-detected from whatever libass `pkg-config` finds. Link
stock libass and every deferred-subtitle code path is **compiled out** — the build succeeds and
the feature silently does nothing. So keep two build directories:

| Build dir | libass | Purpose |
|---|---|---|
| `build-stock` | MSYS2's `libass` (0.17.4 here) | plain fork compile-check, matches upstream CI |
| `build-fork` | `the-database/libass` built into a prefix | the deferred/threaded subtitle paths |

Confirm which you have — this is the whole test:

```bash
grep -E 'HAVE_ASS_(BLUR|COMPOSITE|OUTLINE)_DEFERRED|HAVE_ASS_RENDER_THREAD_COUNT|HAVE_ASS_SHADOW_SHIFT' build-fork/config.h
```

Expected in `build-fork` (all five `1`); in `build-stock` all five are `0`:

```c
#define HAVE_ASS_BLUR_DEFERRED 1
#define HAVE_ASS_COMPOSITE_DEFERRED 1
#define HAVE_ASS_OUTLINE_DEFERRED 1
#define HAVE_ASS_RENDER_THREAD_COUNT 1
#define HAVE_ASS_SHADOW_SHIFT 1
```

`config.h`'s `FULLCONFIG` string is the other tell: the fork build's begins
`"ass-blur-deferred ass-composite-deferred ass-outline-deferred ass-render-thread-count ass-shadow-shift build-date ..."`
where the stock build's begins `"build-date ..."`.

## Windows: MSYS2 UCRT64

### Invocation shape

Everything runs through a UCRT64 login shell. From PowerShell:

```powershell
$env:MSYSTEM="UCRT64"; $env:CHERE_INVOKING="1"
& C:\msys64\usr\bin\bash.exe -lc '<commands>'
```

`CHERE_INVOKING=1` keeps the current directory instead of jumping to `$HOME`.

### What MSYS2 UCRT64 has, and what it does not

Verified present: `meson` 1.11.1, `ninja` 1.13.2, `gcc` 16.1.0, `pkgconf` 2.5.1, and stock
`libass` 0.17.4, plus libplacebo and FFmpeg.

**Missing from this MSYS2 install:** `autoconf`, `automake`, `libtool`, `make`, `nasm`, `git`,
`cmake`.

Two consequences:

- Use **Windows git from PowerShell**, not from inside the MSYS2 shell.
- **The libass fork's autotools build cannot be run here** (no autoconf/automake/libtool/make).
  On Windows, build libass with meson instead — see below. The autotools path is only used by
  the Linux CI bundle, for the `libass.sym` visibility reason documented in
  [`animejanai-build-ci.md`](animejanai-build-ci.md).

### Build A — stock libass

```bash
cd <mpv-checkout>
meson setup build-stock -Dlibmpv=true -Dtests=true
ninja -C build-stock
```

The existing `build-stock` here was configured with `-Dwerror=true` as well
(`build-stock/meson-private/cmd_line.txt`).

> **`--werror` has broken before** on a pre-existing warning in `video/filter/refqueue.h`. If a
> `-Dwerror=true` build fails on a file the current change does not touch, drop `-Dwerror`
> rather than "fixing" upstream code.

### Build B — fork libass

Build the libass fork into a **stable prefix** and point `PKG_CONFIG_PATH` at it:

```bash
# choose a path that will still exist tomorrow
PREFIX=<libass-prefix>

git clone https://github.com/the-database/libass.git <libass-src>   # from PowerShell; MSYS2 has no git
meson setup libass-build libass-src --prefix="$PREFIX" -Ddefault_library=shared -Dasm=disabled
ninja -C libass-build install
# installs libass-9.dll, libass.dll.a, ass.h, ass_types.h, libass.pc

cd <mpv-checkout>
PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig" meson setup build-fork -Dlibmpv=true -Dtests=true
ninja -C build-fork
```

`-Dasm=disabled` is needed because MSYS2 has no `nasm`. Then verify the five defines as above.

> **Do not reuse the `build-fork` directory that is currently in this tree.** Its
> `PKG_CONFIG_PATH` was baked to a Claude Code **session scratchpad** path
> (`.../Temp/claude/.../scratchpad/libass-prefix`), which is ephemeral. Reconfiguring it after
> that directory is gone will silently fall back to stock libass and flip all five defines to
> `0`. Delete it and re-setup against a stable prefix.

To run the resulting binaries, put the prefix's `bin` on `PATH`.

### Smoke runs

```bash
./build-fork/mpv.exe --gpu-api=vulkan --sub-ass-render-threads=4 \
  --sub-file=TOOLS/subtest/samples/<sample>.ass 'av://lavfi:testsrc2=size=1920x1080:rate=24'
```

> **`--gpu-api=vulkan` is required** for MSYS2-built mpv here. The default d3d11 backend fails
> every OSD texture upload (`Failed uploading OSD texture!`). This reproduces on the stock
> baseline too, so it is an environment issue, not a fork regression.

`mpv -v | grep 'libass render threads:'` confirms at runtime that you are on a
deferred/threaded libass build.

### `meson test` baseline

```bash
meson test -C build-stock
```

**Five failures are pre-existing** with the local FFmpeg 8.1.1 and are not regressions:
the four FFmpeg-dependent tests (`img-format`, `scale-sws`, `repack`, `scale-zimg`) and
`libmpv-lifetime`. Compare against this baseline before investigating a test failure.

## WSL (Ubuntu)

WSL 2.6.1.0, one distro (`Ubuntu`, 24.04, gcc 13.3.0), clone at `~/src/mpv`.

Recorded configurations, recovered from the build directories' own metadata:

```bash
cd ~/src/mpv
meson setup build
meson setup build-fork -Db_sanitize=address -Db_lundef=false -Dlua=enabled
```

`-Db_lundef=false` is what lets a sanitizer build link. The `build-fork` there is the
AddressSanitizer build; use it when chasing a crash or UAF in the subtitle pipeline (the libass
fork's history includes a `render_and_combine_glyphs` use-after-free fix found this way).

The libass fork is cloned separately at `~/src/libass`, with its own meson build dirs
including `build-asan` and `build-tsan` — see that repo's `CLAUDE.md`.

> **Only one WSL dependency is provable**: `sudo apt-get install -y libluajit-5.1-dev` appears
> in `~/.bash_history`. The rest of the package set is **unverified** — the builds were driven
> from inside Claude Code sessions, so `.bash_history` contains no `meson`/`ninja` lines, and
> the `/tmp/aji-bootstrap.sh` referenced there no longer exists. Expect to install mpv's normal
> Debian build-deps (`apt build-dep mpv` or the list in upstream's docs) on a fresh distro.

> The `~/src/mpv` checkout is from **2026-06-28** and is behind the Windows clone.
> `git pull` before using it as a reference for current code.

## Which clone is which

| Path | Repo | Use |
|---|---|---|
| `<mpv-checkout>` | `the-database/mpv` (+ `upstream` remote) | the working clone |
| a sibling `mpv upstream` clone | `mpv-player/mpv` | read-only upstream reference |
| `~/src/mpv` (WSL) | `the-database/mpv` (+ `upstream`) | sanitizer builds; stale |

`%AJI_WIN%\mpv` is a **separate** mpv source tree with its own meson `build/`,
whose `origin` is `a local path`. It is the tree the `aji-win` playback and
benchmark scripts run (`%AJI_WIN%\mpv\build\mpv.exe`), rebuilt with
`%AJI_WIN%\build-mpv.cmd` / `build-mpv-incr.cmd`, which wrap:

```bash
cd $AJI_WIN/mpv
meson setup build --buildtype=release   # build-mpv.sh (full, rm -rf build first)
ninja -C build                          # build-mpv-incr.sh (incremental)
```

Treat it as an engine-testing rig, not as this repo's working clone.
