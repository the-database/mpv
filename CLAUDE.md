# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this fork is

`the-database/mpv` — a fork of [`mpv-player/mpv`](https://github.com/mpv-player/mpv) carrying
two bodies of work for the AnimeJaNai project:

1. **`vf_animejanai`** — a video filter that hands the decoder's GPU frames to the `aji`
   inference engine (real-time anime upscaling + RIFE interpolation) and gets them back,
   without ever leaving the GPU.
2. **A GPU/threaded subtitle pipeline** — multithreaded libass rendering, render-ahead, and
   deferred GPU primitives (blur / composite / outline raster) built on new APIs in the
   companion `the-database/libass` fork.

Remotes: `origin` = `the-database/mpv`, `upstream` = `mpv-player/mpv`. As of 2026-09-12
`origin/master` is **227 commits ahead of and 0 behind** `upstream/master` (160 of them
non-merge). That number moves with every nightly sync — recompute rather than trusting it:

```bash
git fetch upstream && git rev-list --count upstream/master..origin/master
```

Nothing in the AnimeJaNai package is built from this repo locally — the shipped binaries come
from CI. This is where the source changes live.

## Documentation map

- **[`DOCS/animejanai-build-local.md`](DOCS/animejanai-build-local.md)** — build and smoke-test the fork on this
  machine (MSYS2 UCRT64 on Windows, or WSL), including how to get a fork-libass build so the
  deferred-subtitle code paths actually compile in.
- **[`DOCS/animejanai-build-ci.md`](DOCS/animejanai-build-ci.md)** — the two shipped builds (Linux here, Windows in
  `the-database/mpv-winbuild`), the upstream-sync workflow, and which CI failures are
  pre-existing rather than regressions.

Upstream's own build docs are still accurate for everything not mentioned above.

## Upstream sync policy: merge only

`.github/workflows/sync-upstream.yml` runs nightly (`cron: '0 6 * * *'`) plus on dispatch. It
does `git merge upstream/master` and pushes `master`; on conflict it aborts, collects
`git diff --name-only --diff-filter=U`, and files/comments on an issue titled
`Upstream sync conflict`.

**Never rebase or squash a sync.** The workflow's own header explains why a merge is required
(the `merge-upstream` REST API only fast-forwards, which a diverged fork cannot use), and merge
topology is what keeps the nightly job fast-forwardable. Land a conflicting sync by resolving
the merge and fast-forwarding `master` to the merge commit.

To preview a sync's conflicts without touching the tree:

```bash
git merge-tree --write-tree --messages origin/master upstream/master
```

> The local `master` branch in this clone has been a stale divergent branch before. Branch from
> `origin/master`, not from local `master`.

## What the fork adds and changes

Added files (`git diff --name-status upstream/master...HEAD`, excluding `TOOLS/subtest/samples/`):

| Path | What |
|---|---|
| `video/filter/vf_animejanai.c` | the filter (~80 KB) |
| `video/filter/aji.h` | the `aji` C ABI it loads — **must match the engine's `include/aji.h`** |
| `sub/sub_ahead.c`, `sub/sub_ahead.h` | subtitle render-ahead worker |
| `sub/sub_phase.h` | per-phase subtitle timing instrumentation |
| `ci/build-linux-portable.sh` | the portable Linux bundle build |
| `.github/workflows/build-linux.yml` | dispatches the above |
| `.github/workflows/sync-upstream.yml` | the nightly merge |
| `TOOLS/subtest/` | offline subtitle-performance harness (19 scripts + `samples/`) |

Modified areas, by weight: `sub/` (15 files), `filters/` (8), `video/filter/` (5), plus
`player/`, `options/`, `video/out/`, `video/decode/`, `meson.build`, and `DOCS/man/options.rst`.

**No workflow file is modified** — only the two above are added. The fork adds **no meson
options**: `meson.options` is untouched (note mpv uses `meson.options`, not
`meson_options.txt`).

### `vf_animejanai` is gated on CUDA, not on an option of its own

`meson.build:1436-1439`:

```meson
features += {'cuda-hwaccel': cuda_hwaccel.allowed()}
if features['cuda-hwaccel']
    sources += files('video/filter/vf_animejanai.c',
                     'video/out/hwdec/hwdec_cuda.c')
endif
```

So `-Dcuda-hwaccel=enabled` (or auto-detected CUDA) is what compiles the filter in. There is no
`-Danimejanai` flag to look for.

### The libass feature probes

`meson.build:479-509` adds five auto-detected features. They are **compile-only declaration
checks**, deliberately not `has_function()` — the comment in the file explains that linking a
probe can spuriously fail against a static libass whose transitive deps are not all on the
probe link line (e.g. mingw cross builds), silently compiling out the call.

| Feature | `config.h` define | Probe |
|---|---|---|
| `ass-render-thread-count` | `HAVE_ASS_RENDER_THREAD_COUNT` | `ass_set_render_thread_count` |
| `ass-blur-deferred` | `HAVE_ASS_BLUR_DEFERRED` | `ass_set_blur_deferred` |
| `ass-composite-deferred` | `HAVE_ASS_COMPOSITE_DEFERRED` | `ass_set_composite_deferred` |
| `ass-outline-deferred` | `HAVE_ASS_OUTLINE_DEFERRED` | `ass_set_outline_deferred` |
| `ass-shadow-shift` | `HAVE_ASS_SHADOW_SHIFT` | `ASS_Image.shift_x64` (a **struct member**, not a symbol) |

Every consumer is guarded by its own `HAVE_` define and degrades to the CPU path with a one-shot
`MP_WARN` when the symbol is absent, because the libass fork may ship any subset and stock
libass ships none. **Build against stock libass and all five are 0** — the deferred code is
compiled out, so a "no effect" result usually means the wrong libass, not a logic bug. See
`DOCS/animejanai-build-local.md` for how to confirm.

### Fork-added runtime options

Registered in `options/options.c:357-369`, documented in `DOCS/man/options.rst`:

```
--sub-ass-render-threads=<0-64>        --sub-gpu-blur=<yes|no>
--sub-render-ahead-frames=<0-240>      --sub-gpu-composite=<yes|no>
--sub-render-ahead-threads=<0-64>      --sub-gpu-raster=<yes|no>
--sub-render-ahead-miss-wait=<-1|0|ms>
--sub-render-ahead-max-frames=<0-960>
```

plus `--sub-glyph-atlas-size`, `--sub-glyph-atlas-height`, `--sub-present-guard-ms`,
`--sub-debug-stall-ms`, `--sub-prefill-budget-ms`, `--sub-render-res-limit`, and
`--osd-render-res-cap` in the man page.

> **`0` and `-1` are not always "off".** In the AnimeJaNai package's managed profile,
> `sub-ass-render-threads=1` and `sub-present-guard-ms=0` are the *off* values, while `0` and
> `-1` mean auto/armed. Check the option's documented semantics before assuming a default
> disables a feature.

## Where this fork is consumed

| Consumer | How |
|---|---|
| `the-database/mpv-winbuild` | GitHub Actions cross-compiles Windows `libmpv-2.dll` + `mpv.exe` from this repo's `master` |
| this repo's `build-linux.yml` | builds the portable Linux bundle `mpv-linux-x64-<tag>.tar.zst` |
| `the-database/mpv-AnimeJaNai` | the assembler downloads both, pinned by `MpvForkVersion` / `MpvForkLinuxVersion` |

**The filter↔engine ABI couples this repo to `the-database/animejanai-inference`.** When
`aji.h`'s `AJI_API_VERSION` changes, rebuild both and bump both pins in the assembler. The
filter lives on `master` (aji ABI v8); the old standalone `vf-animejanai` branch is stale
(ABI v4) and must not be used.

## Conventions

- Commits: author `the-database`, short imperative subject, no co-author trailers.
- Keep fork changes minimal and upstream-mergeable where possible; the nightly sync has to keep
  working. `sub/osd*` and `video/out/vo_gpu_next.c` are the usual conflict sites when upstream
  refactors.
- New subtitle options belong in `options/options.c` + `DOCS/man/options.rst`, not in a
  downstream config file.
