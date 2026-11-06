// sc_mode=2 uses Analyse's default dct=0. No FFTW dependency is needed.
// The original search includes unreachable calls for other DCT modes.
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
struct DCTFFTW;
inline void dctBytes2D(DCTFFTW *, const uint8_t *, ptrdiff_t, uint8_t *, ptrdiff_t) {
    throw std::logic_error("The sc_mode=2 detector only supports dct=0");
}
