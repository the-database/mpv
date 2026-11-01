# MVTools scene detector source

The analysis core is from dubhatervapoursynth/vapoursynth-mvtools,
commit `17250aa979616ac48dfb0e18abfdcf2bd4e3afc0` (2026-06-21), under GPL-2.0-or-later.
Source: https://github.com/dubhatervapoursynth/vapoursynth-mvtools/tree/17250aa979616ac48dfb0e18abfdcf2bd4e3afc0

The adapter reproduces AnimeJaNai 3.2.1's `Super(pel=1)`, `Analyse(isb=True)`
and `SCDetection(thscd1=240, thscd2=130)` defaults. It keeps the original search,
pyramid, block coverage, chroma contribution, integer threshold scaling and
strict greater-than comparisons. It consumes full-resolution planar YUV/gray.

Local adaptations: the VapourSynth plugin interface is excluded; its allocation
and row-copy helpers are supplied in `vendor/memory.h`; unused VS declarations
were removed from `MVAnalysisData.h`. DCT is fixed to the legacy default of zero,
so FFTW is not required. On x64, `AJN_MVTOOLS_SSE2` enables the original SSE2
intrinsics for the detector's 8x8 SAD/luma and subsampled chroma SAD blocks.
Other kernels retain their original C implementation; the plugin's assembly
dispatch is not required. Motion-search allocations and vector output storage
are reused, with predictors cleared for each pair to preserve fresh Analyse
semantics. No VapourSynth runtime is loaded or distributed. `detector.cpp`
supplies the frame adapter and thresholds.

The optional `ajn_mvtools_analyze_enhanced` export is AJN's scene classifier,
not part of upstream MVTools or the old `sc_mode=2` algorithm. `enhanced.h`
measures changed-region and motion-aligned appearance; temporal motion energy
and feature correspondence distinguish weak cuts from continuous animation.
It can accept cuts below the legacy block-count threshold or reject legacy
false positives. It uses the existing frame readback, adds no lookahead, and
leaves the legacy export's results unchanged. A matching host/player advertises
this separately as `privateEnhancedSceneAbi: 1`.

The source revision of the affected user's old libmvtools.dll is not yet known.
Visual equivalence and real-time performance need qualification on their clips.
