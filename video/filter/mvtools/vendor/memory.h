// Minimal allocation/copy helpers for the standalone MVTools analysis core.
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>
#ifdef _WIN32
#include <malloc.h>
#define VS_RESTRICT __restrict
#define VSH_ALIGNED_MALLOC(p, size, align) (*(p) = _aligned_malloc(size, align))
#define VSH_ALIGNED_FREE(p) _aligned_free(p)
#else
#define VS_RESTRICT __restrict__
#define VSH_ALIGNED_MALLOC(p, size, align) posix_memalign(p, align, size)
#define VSH_ALIGNED_FREE(p) free(p)
#endif
#define VSMIN(a, b) ((a) < (b) ? (a) : (b))
#define VSMAX(a, b) ((a) > (b) ? (a) : (b))
namespace vsh {
inline void bitblt(void *dst, ptrdiff_t dstStride, const void *src,
                   ptrdiff_t srcStride, size_t bytes, size_t height) {
    for (size_t y = 0; y < height; ++y)
        std::memcpy(static_cast<char *>(dst) + y * dstStride,
                    static_cast<const char *>(src) + y * srcStride, bytes);
}
}
