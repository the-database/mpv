# AJN addon native integration: reviewer entry point

This PR implements trusted native providers for the Windows AJN addon host. WebAssembly addons run outside mpv and never receive raw handles, frame pointers, arbitrary player commands or DLL loading. The central system/creator documents are in the companion `mpv-AnimeJaNai` PR under `addons/REVIEW-GUIDE.md` and `addons/DEVELOPER-GUIDE.md`.

## Code map

| Area | Review paths | Contract |
| --- | --- | --- |
| Independent media workers | `common/ajn_probe.*`, `ajn_mux.*`, `ajn_subtitles.*`, `common/encode_lavc.c`, `video/out/vo_lavc.c` | Owned input callbacks, bounded encoded output, track/seek/subtitle handling and cancellation |
| GPU sampling | `video/filter/vf_ajn_sample.c`, `ajn_sample_gpu.*`, `ajn_sample_cuda.*`, sample shared/lifetime headers | Preserve original frames; bounded optional samples; seek epochs; release only after GPU completion |
| Existing-player observation | Filter insertion/control in `filters/` and sample providers | Lease-bound source/processed observations with shared producers and independent readers |
| RIFE decisions | `video/filter/ajn_scene.*`, `vf_animejanai.c`, `aji.h` | Optional per-player ownership, bounded decision waits, explicit fallback/suspension and revocation |
| Built-in MVTools helper | `video/filter/mvtools/` | Trusted, lazily loaded AJN helper; vendored source, license and upstream record are included |
| Feature discovery | `etc/ajn-native-capabilities.json` | Private ABI expectations to combine with producer metadata; source declarations alone are not binary proof |

The detailed original D3D11 sample contract is in [ajn-sample-worker.md](ajn-sample-worker.md). API 1.12 also adds optional CUDA source color/character-mask observation through `aji_analysis.h`. The full current mask semantics and unsupported formats are documented in the companion host's `PLAYER-ANALYSIS.md`.

## Performance and lifetime expectations

An unrequested observation does not insert a sampling filter, load the optional analysis module or perform readback/model work. Enabled observations reduce on the GPU, hold bounded pending work, and skip when busy. The source image and ordinary render path are preserved. Source samples precede display color processing, OSD and subtitles; their PTS is not a display timestamp.

The MVTools option is different: its enabled scene detector downloads source data for CPU analysis. This cost belongs to the explicitly enabled detector. Its fixed native helper is shipped by AJN; arbitrary guest DLLs are not supported.

`--vo-null-accept-hwframes` is opt-in for an independent discard worker. Default `vo=null` behavior remains unchanged. Hardware encoding imports frames from their owning hardware pool rather than requiring a display interop context. Backpressure applies to the owned worker, not another player's decoder.

Repeated libmpv load/unload needs the included Windows ggml lifetime fix: already-loaded ggml modules are pinned because their process-wide terminate handler otherwise points into unloaded code. The patch does not load a new library from DllMain. This is startup lifetime handling, not per-frame processing.

## Compatibility and tests

Keep the additive inference hooks synchronized with the companion inference PR. Default detection remains available with older backends; unsupported explicit overrides must not be silently accepted. Private maps validate versions, dimensions, ownership, epochs and leases before use. The host supervises worker termination if a device or native helper stops responding.

Native tests cover sample lifetime/shared-state and scene ownership as well as libmpv encoding lifetime. The main repo's native addon tests cover real SDK/host/player interactions, including mpv.net, and provide the broader media fixtures. The frozen 3.9.0 integration used mpv `4eb81dc5e00ad1e4d208c91e384edf04d0de2d05`; the review changes here add documentation, with unchanged runtime code unless the PR explicitly records otherwise.

Use the winbuild PR to build the selected source and generate a capability/file-hash record. The main assembler verifies that record before combining host, Manager and inference. Preserve original build-source pins when reporting results. Linux addon support, HDR masking, universal GPU behavior, physical lighting and Plex-client interoperability are not claimed by these Windows checks.

No smooth-camera-pan or custom display-sync implementation is introduced here. The separate cadence-scan PR remains independent.
