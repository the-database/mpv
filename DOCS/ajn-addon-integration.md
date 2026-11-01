# AJN addon native integration: reviewer entry point

This PR implements trusted native providers for the Windows AJN addon host. WebAssembly addons run outside mpv and never receive raw handles, frame pointers, arbitrary player commands or DLL loading. The central system/creator documents are in the companion `mpv-AnimeJaNai` PR under `addons/REVIEW-GUIDE.md` and `addons/DEVELOPER-GUIDE.md`.

## Code map

| Area | Review paths | Contract |
| --- | --- | --- |
| Independent media workers | `common/ajn_probe.*`, `ajn_mux.*`, `ajn_subtitles.*`, `common/encode_lavc.c`, `video/out/vo_lavc.c` | Owned input callbacks, bounded encoded output, track/seek/subtitle handling and cancellation |
| GPU sampling | `video/filter/vf_ajn_sample.c`, `ajn_sample_gpu.*`, `ajn_sample_cuda.*`, sample shared/lifetime headers | Preserve original frames; bounded optional samples; seek epochs; release only after GPU completion |
| Existing-player observation | Filter insertion/control in `filters/` and sample providers | Lease-bound source/processed observations with shared producers and independent readers |
| RIFE decisions | `video/filter/ajn_scene.*`, `vf_animejanai.c`, `aji.h` | Optional per-player ownership, asynchronous source-pair analysis, immediate built-in fallback and revocation |
| Built-in MVTools helper | `video/filter/mvtools/` | Trusted, lazily loaded AJN helper; vendored source, license and upstream record are included |
| Feature discovery | `etc/ajn-native-capabilities.json` | Private ABI expectations to combine with producer metadata; source declarations alone are not binary proof |

The detailed original D3D11 sample contract is in [ajn-sample-worker.md](ajn-sample-worker.md). API 1.12 also adds optional CUDA source color/character-mask observation through `aji_analysis.h`. The full current mask semantics and unsupported formats are documented in the companion host's `PLAYER-ANALYSIS.md`.

## Performance and lifetime expectations

An unrequested observation does not insert a sampling filter, load the optional analysis module or perform readback/model work. Enabled observations reduce on the GPU, hold bounded pending work, and skip when busy. The source image and ordinary render path are preserved. Source samples precede display color processing, OSD and subtitles; their PTS is not a display timestamp.

An attached RIFE scene detector uses bounded source-frame lookahead and a worker-owned queue. The worker downloads the original decoded source, reduces data or runs MVTools, and waits for an addon reply. Playback only enqueues bounded references and tries to consume an already-completed result for the exact PTS/epoch. It never waits for a host-supplied response budget, GPU readback or CPU motion analysis. A missing, late or stale result uses the built-in detector. A detector that continually takes 95 ms therefore cannot hold playback at ten pairs per second; it can miss its decisions instead.

This optional analysis still consumes GPU/CPU resources and retains bounded source references. No scene worker, lookahead or readback is added with the hook absent. MVTools is a trusted, lazily loaded GPL helper located relative to libmpv. It is built only when GPL is enabled; LGPL builds omit it. Arbitrary guest DLLs are not supported.

`--vo-null-accept-hwframes` is opt-in for an independent discard worker. Default `vo=null` behavior remains unchanged. Hardware encoding imports frames from their owning hardware pool rather than requiring a display interop context. Backpressure applies to the owned worker, not another player's decoder.

The host retains its native library graph for the worker process lifetime. This PR does not add global ggml pinning to mpv's `DllMain`. Optional sample/scene workers retain their own module while retiring; filter destruction signals cancellation and releases ownership without waiting for worker initialization. Failed mapping/setup paths release their partial resources.

## Compatibility and tests

Keep the additive inference hooks synchronized with the companion inference PR. Default detection remains available with older backends; unsupported explicit overrides must not be silently accepted. Private maps validate versions, dimensions, ownership, epochs and leases before use. The host supervises worker termination if a device or native helper stops responding.

The probe, encoder, mux and subtitle entry points are Windows-only private exports declared in `common/ajn_probe.h`, `common/ajn_mux.h` and `common/ajn_subtitles.h`. They are not part of libmpv's public cross-platform API. The mux callback supports the FFmpeg 6.1 and 7+ signatures. HLS uses a private `ajn-cache:` URL with host-side atomic playlist publication, avoiding FFmpeg filesystem renames outside the supplied callbacks.

The host enforces guest permissions and restricts player commands. mpv IPC and scripts are trusted separately: a caller already allowed to issue arbitrary mpv commands can configure scene/sample filters and raw-handle options. Do not expose that IPC endpoint to untrusted addons or clients.

Native tests exercise sample lifetime/shared-state and scene ownership as well as libmpv encoding lifetime. The main repo's native addon tests exercise SDK/host/player interactions, including mpv.net, and provide media fixtures. The PR records exactly which revised paths were tested and with which binaries; older preview results do not validate the asynchronous scene rewrite.

Use the winbuild PR to build the selected source and generate a capability/file-hash record. The main assembler verifies that record before combining host, Manager and inference. Preserve original build-source pins when reporting results. Linux addon support, HDR masking, universal GPU behavior, physical lighting and Plex-client interoperability are not claimed by these Windows checks.

No smooth-camera-pan or custom display-sync implementation is introduced here. The separate cadence-scan PR remains independent.
