# Private AJN media-worker sample transport

Start with [ajn-addon-integration.md](ajn-addon-integration.md) for the complete API 1.12 native review map and build/qualification requirements.

`ajn-sample=mapping=<handle>` is an optional filter for the pinned, supervised AJN
media worker. The handle is an unnamed mapping duplicated by its trusted parent.
It is not exposed to addon code, scripts or network clients. The public addon API
does not depend on this native ABI or accept filter strings or Windows handles.

The version-1 layout is `video/filter/ajn_sample_shared.h`. The mapping is exactly
256 + 230400 bytes. Only the parent writes the packed configuration; only the
filter publishes metadata/pixels and increments its discontinuity epoch. A
reader must check the even publication sequence before and after copying and
reject obsolete configuration/epoch values. Reads must have bounded retries.
No consumer acknowledgment is needed and old samples are overwritten.

No GPU work occurs with configuration disabled. While enabled, the filter retains
at most one input image until its asynchronous GPU read completes. A private
video processor scales and converts matrix/range into a small BGRA8 surface,
then copies only that small surface to CPU staging. A readback thread polls an
event query and maps with DO_NOT_WAIT. Input delivery never waits for a sample
consumer; a busy tap skips a sample. Device initialization and GPU submission
still have a cost and must be measured with actual playback.

The original full-resolution `mp_image` passes through unchanged. Samples are
from the processed filter raster, with its crop applied and pixel aspect,
rotation and vertical-flip metadata preserved for consumers. They precede the
display's rotation, tone mapping, color management, subtitles and OSD. The PTS is
a processing PTS, not a measured display-presentation time. The requested sample
rectangle stretches the crop; it does not add display letterboxing.

This producer supports progressive mono SDR D3D11 images with BT.601/709/RGB
matrices, supported SDR transfer functions and BT.601/709 primaries. Matrix and
range conversion preserve the source primaries/transfer. HDR, interlaced, stereo
and CUDA images report unavailable; they are never silently tagged as SDR.
Dimensions are bounded to 320x180 and sampling to 60 Hz. These are negotiated
producer limits, not fixed limits for every future addon capability.

On seek, old pending samples retain their source until GPU completion but cannot
publish into the new epoch. Normal teardown drains the sole sample. The parent
media-process supervisor enforces a shutdown deadline if the driver stops
responding. No addon code runs inside the player or driver process.
