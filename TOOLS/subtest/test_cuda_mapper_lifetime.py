#!/usr/bin/env python3
"""Exercise the real CUDA mapper callbacks with a deferred-copy CUDA stub.

No GPU/player build is needed. For example:
  python test_cuda_mapper_lifetime.py --compiler zig cc

Only these callbacks are compiled; this does not replace a full player build
or playback validation. The stub checks source lifetime, not GPU scheduling.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "video/out/hwdec/hwdec_cuda.c"

PREAMBLE = r"""
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define assert(condition) do { if (!(condition)) { \
    fprintf(stderr, "line %d: %s\n", __LINE__, #condition); exit(1); \
} } while (0)

typedef void *CUcontext;
typedef void *CUevent;
typedef void *CUarray;
typedef uintptr_t CUdeviceptr;
typedef struct {
    int srcMemoryType, dstMemoryType;
    CUdeviceptr srcDevice;
    CUarray dstArray;
    unsigned srcPitch, srcY, WidthInBytes, Height;
} CUDA_MEMCPY2D;
enum { CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_ARRAY };

static int refs, waits, stream_waits;
static bool copy_queued, copy_finished, event_recorded;
static bool fail_signal, fail_record;
static int push(CUcontext context) { return 0; }
static int pop(CUcontext *context) { return 0; }
static int copy_async(const CUDA_MEMCPY2D *copy, void *stream) {
    assert(refs > 0);
    copy_queued = true;
    return 0;
}
static int record(CUevent event, void *stream) {
    assert(copy_queued);
    if (fail_record) return -1;
    event_recorded = true;
    return 0;
}
static int event_sync(CUevent event) {
    assert(event_recorded && refs > 0);
    waits++;
    copy_finished = true;
    return 0;
}
static int stream_sync(void *stream) {
    assert(refs > 0);
    stream_waits++;
    copy_finished = true;
    return 0;
}
typedef struct {
    int (*cuCtxPushCurrent)(CUcontext);
    int (*cuCtxPopCurrent)(CUcontext *);
    int (*cuMemcpy2DAsync)(const CUDA_MEMCPY2D *, void *);
    int (*cuEventRecord)(CUevent, void *);
    int (*cuEventSynchronize)(CUevent);
    int (*cuStreamSynchronize)(void *);
} CudaFunctions;
struct mp_image { int num_planes; void *planes[4]; unsigned stride[4]; };
struct ra_hwdec_mapper;
struct cuda_hw_priv {
    CudaFunctions *cu;
    bool do_full_sync;
    bool (*ext_wait)(const struct ra_hwdec_mapper *, int);
    bool (*ext_signal)(const struct ra_hwdec_mapper *, int);
};
struct cuda_mapper_priv {
    struct mp_image layout;
    CUarray cu_array[4];
    CUcontext display_ctx;
    CUevent copy_done;
    bool copy_pending;
};
struct format { unsigned pixel_size; };
struct texture { struct { struct format *format; } params; };
struct owner { struct cuda_hw_priv *priv; };
struct ra_hwdec_mapper {
    struct cuda_mapper_priv *priv;
    struct owner *owner;
    struct mp_image *src;
    struct texture *tex[4];
};
static unsigned mp_image_plane_w(struct mp_image *image, int n) { return 16; }
static unsigned mp_image_plane_h(struct mp_image *image, int n) { return 16; }
static void drop_ref(void) {
    assert(refs > 0);
    if (--refs == 0) assert(!copy_queued || copy_finished);
}
static void mp_image_unrefp(struct mp_image **image) {
    if (*image) drop_ref();
    *image = NULL;
}
static bool signal_copy(const struct ra_hwdec_mapper *mapper, int n) {
    return !fail_signal;
}
#define CHECK_CU(call) (call)
"""

TEST = r"""
static void run_case(bool vulkan, bool signal_error, bool event_error) {
    refs = 2; // mapper's reference and the libplacebo frame queue's reference
    waits = stream_waits = 0;
    copy_queued = copy_finished = event_recorded = false;
    fail_signal = signal_error;
    fail_record = event_error;
    CudaFunctions cu = {push, pop, copy_async, record, event_sync, stream_sync};
    struct cuda_hw_priv hw = {.cu = &cu, .do_full_sync = !vulkan,
                              .ext_signal = vulkan ? signal_copy : NULL};
    struct owner owner = {&hw};
    struct cuda_mapper_priv priv = {.layout.num_planes = 1,
                                    .copy_done = vulkan ? (CUevent)1 : NULL};
    struct mp_image image = {0};
    struct format format = {1};
    struct texture tex = {.params.format = &format};
    struct ra_hwdec_mapper mapper = {&priv, &owner, &image, {&tex}};

    int ret = mapper_map(&mapper);
    assert((ret < 0) == (signal_error || event_error));
    if (vulkan) assert(mapper.src && refs == 2);

    // Simulate a seek discarding the queued frame while the copy is deferred.
    drop_ref();
    mapper_unmap(&mapper);
    // ra_hwdec_mapper_unmap performs this after the driver's unmap callback.
    mp_image_unrefp(&mapper.src);
    assert(refs == 0 && copy_finished);
    assert(waits == (vulkan && !event_error));
    assert(stream_waits == (!vulkan || event_error));
    mapper_unmap(&mapper); // repeated cleanup must not wait again
    assert(waits == (vulkan && !event_error));
}
int main(void) {
    run_case(true, false, false);
    run_case(true, true, false); // queued copy survives a partial map failure
    run_case(true, false, true); // failed event record still drains the copy
    run_case(false, false, false); // existing OpenGL full-sync path
    puts("CUDA mapper lifetime: 4 cases passed");
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", nargs="+", required=True)
    args = parser.parse_args()
    source = SOURCE.read_text(encoding="utf-8")
    callbacks = []
    for name in ("mapper_unmap", "mapper_map"):
        match = re.search(r"^static (?:void|int) " + name + r"\([^\n]*\)\n\{.*?^\}",
                          source, re.MULTILINE | re.DOTALL)
        if not match:
            raise RuntimeError(f"Cannot locate {name} in {SOURCE}")
        callbacks.append(match.group())
    with tempfile.TemporaryDirectory(prefix="ajn-cuda-lifetime-") as temp:
        temp = Path(temp)
        cfile = temp / "mapper.c"
        exe = temp / "mapper-test.exe"
        cfile.write_text(PREAMBLE + "\n".join(callbacks) + TEST, encoding="utf-8")
        subprocess.run([*args.compiler, "-std=c11", "-Werror=implicit-function-declaration",
                        str(cfile), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
