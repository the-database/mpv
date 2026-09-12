# CI and the shipped builds

Local builds: [`animejanai-build-local.md`](animejanai-build-local.md). What the fork changes: [`../CLAUDE.md`](../CLAUDE.md).

The AnimeJaNai package ships **two** mpv builds from this fork, produced by two different
repos. Neither is built on a dev machine.

| Platform | Built by | Artifact |
|---|---|---|
| Linux | `build-linux.yml` **in this repo** | `mpv-linux-x64-<tag>.tar.zst` |
| Windows | `the-database/mpv-winbuild`, workflow `MPV` | `mpv-x86_64-<date>-git-<hash>.7z` + `mpv-dev-...7z` |

Both are consumed by `the-database/mpv-AnimeJaNai`'s assembler, pinned by
`MpvForkLinuxVersion` and `MpvForkVersion`/`MpvForkBuildDate`/`MpvForkGitHash`.

## Linux: `Build Linux (portable mpv + vf_animejanai)`

`.github/workflows/build-linux.yml`. Builds a portable bundle — `mpv` + `libmpv.so.2` with
`vf_animejanai`, plus its non-system shared-lib deps — and publishes it as a release asset.

### Dispatching

```bash
gh workflow run "Build Linux (portable mpv + vf_animejanai)" -R the-database/mpv \
  --ref master -f release_tag=2026-07-23-7fc08d9
```

`release_tag` is the only input and is **required** on dispatch. It becomes the release tag,
the release name, and part of the asset filename, and must match `MpvForkLinuxVersion` in the
assembler. A push-triggered run instead falls back to `env.DEFAULT_TAG` (currently
`2026-06-18-9bb5fe9680`).

> **Dispatch from `master`.** The workflow also has
> `push: branches: [linux-support]` (path-filtered to `ci/build-linux-portable.sh` and the
> workflow file), and its checkout step is commented "linux-support = master + the build
> script". **That is no longer true**: `origin/linux-support` is **442 commits behind
> `origin/master` and 0 ahead** — a strict ancestor — and `ci/build-linux-portable.sh` is
> present on `master`. A push-triggered run would build June code. Use `--ref master`.

### What runs

`ubuntu-latest` inside the container `ghcr.io/the-database/animejanai-linux-build:ubuntu2204`
(GHCR login with `github.actor` + `GITHUB_TOKEN`; that image is built by
`build-image.yml` in `the-database/mpv-AnimeJaNai`). Then:

```bash
chmod +x ci/build-linux-portable.sh
MPV_SRC="$PWD" ci/build-linux-portable.sh "<tag>" "$PWD/out"
```

and `ncipollo/release-action@v1` uploads
`out/mpv-linux-x64-<tag>.tar.zst` with `draft: false`, `allowUpdates: true`.

### `ci/build-linux-portable.sh`

Installs into `PREFIX` (default `/opt/animejanai`), with
`PKG_CONFIG_PATH`/`LD_LIBRARY_PATH` pointed there first so the from-source builds win over the
image's system packages. Sources are shallow-cloned at pinned tags:

```bash
WAYLAND_TAG=1.23.1; WAYLAND_PROTOCOLS_TAG=1.41
FFNVCODEC_TAG=n13.0.19.0; FFMPEG_TAG=n7.1; SHADERC_TAG=v2024.0
VULKAN_TAG=vulkan-sdk-1.3.296.0  # jammy ships 1.3.204; mpv needs vulkan >= 1.3.238
```

plus libplacebo `v7.360.1`. FFmpeg is configured
`--enable-gpl --enable-version3 --enable-shared --disable-static --enable-ffnvcodec
--enable-nvdec --enable-cuvid --enable-nvenc`.

**The libass step is the load-bearing one.** It builds the fork with **autotools, not meson**:

```bash
clone https://github.com/the-database/libass.git "$DEPS/libass" master
( cd "$DEPS/libass" && ./autogen.sh \
  && ./configure --prefix="$PREFIX" --disable-static --enable-shared \
  && make -j"$JOBS" && make install )
```

The script's comment gives the reason: libass's meson build cannot restrict symbol visibility
on a shared library ("not suitable for distribution"), while the autotools build applies the
`libass.sym` version script. Defaults pick up fontconfig / libunibreak / asm / threads from the
image's `-dev` packages, and `require-system-font-provider` (on by default) fails the build
rather than silently shipping a degraded libass.

Note there is **no `--enable-threads`** on that configure line: the libass fork's autotools
option defaults to *check*, so threading comes on by itself when a thread implementation is
found. Contrast the Windows path, which must pass `-Dthreads=enabled` explicitly to meson.

Then mpv itself:

```bash
meson setup "$MPV_SRC/b" "$MPV_SRC" --prefix="$PREFIX" --buildtype=release \
  -Dlibmpv=true -Dcuda-hwaccel=enabled -Dvulkan=enabled -Dlua=enabled
ninja -C "$MPV_SRC/b" install
```

Bundling copies `$PREFIX/bin/mpv` plus its non-system `.so` deps into
`mpv-linux-x64-<tag>/`, skipping anything matched by `EXCLUDE` (the C runtime, libstdc++,
libgcc_s, GL/EGL/GLX, drm/gbm, libva, and `libcuda`/`libnvidia`, which must come from the
host driver), then `tar --zstd` from **inside** the bundle dir so the archive is flat and
extracts straight into the package's `mpv/`.

> **CUDA path discrepancy.** The script exports `PATH="/usr/local/cuda-13.2/bin:$PATH"`, but
> `build/Dockerfile.ubuntu2204` in `the-database/mpv-AnimeJaNai` installs CUDA **13.3** and sets
> `CUDACXX=/usr/local/cuda-13.3/bin/nvcc`. The `animejanai-inference` workflow's comment also
> says "CUDA 13.2 from the image". Unverified which is currently correct; if a build fails
> looking for `nvcc`, check this line first.

## Windows: `the-database/mpv-winbuild`

Cross-compiled on Ubuntu in an Arch container by a fork of `zhongfly/mpv-winbuild`. It resolves
`the-database/mpv` and `the-database/libass` refs to SHAs, `sed`s the
`shinchiro/mpv-winbuild-cmake` toolchain's `packages/{mpv,libass}.cmake` to point at them
(adding `-Dthreads=enabled` for libass), and drops the cached source trees so the pinned
`GIT_TAG` is honoured.

```bash
gh workflow run MPV -R the-database/mpv-winbuild \
  -f mpv_ref=master -f libass_ref=master \
  -f build_target=64bit -f compiler=clang -f release=true
```

Full details, including why the cache-drop lines exist, are in that repo's `CLAUDE.md`.

> The in-repo `ci/build-mingw64.sh` (upstream's mingw CI path) builds **stock libass** via its
> `build_if_missing ... libass` loop. It is a CI compile-check, **not** how the shipped Windows
> build is produced — do not use it to test the deferred-subtitle paths.

## Upstream sync

`sync-upstream.yml`, nightly at `cron: '0 6 * * *'` plus dispatch. Merges `upstream/master`
into `master` and pushes; on conflict it aborts and files/comments on an issue titled
`Upstream sync conflict`. Commits as
`the-database <25811902+the-database@users.noreply.github.com>`.

See [`../CLAUDE.md`](../CLAUDE.md) for the merge-only policy and the conflict dry-run command.

## Pre-existing CI failures — do not "fix" these

A red check on this fork is not automatically a regression. Known-red, verified against a
baseline:

- **The six `lint.yml` jobs.** They fail on fork-only content: `TOOLS/subtest` Python/Lua style,
  missing final newlines in two JSON files, and the commit-message linter rejecting the fork's
  merge-commit history.
- **The two `win32` MSVC jobs.** They fail in `meson.build` at the libass feature probes —
  `cc.has_header_symbol(..., dependencies: libass)` errors with "Dependencies must be external
  dependencies" because libass is a meson **subproject** in that configuration. The fix would
  be to guard the probes on `libass.type_name() == 'internal'`; it has not been done.
- **`Fuzzing (undefined)`** can fail on FFmpeg-internal undefined behaviour (e.g. a
  div-by-zero in `libavformat/vpk.c`), unrelated to mpv.

Everything else in `build.yml` passes: Linux gcc/clang, macOS, FreeBSD/OpenBSD, all msys2
variants, and the mingw cross builds.

`gh run rerun --failed` is appropriate only for genuinely transient failures (e.g. a 502
downloading freetype from savannah), never for the items above.

> **`build.yml`'s `publish` job can never run here.** It is guarded on
> `github.repository == 'mpv-player/mpv'`, so it is skipped in every fork run by design.
