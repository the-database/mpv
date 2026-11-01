/* SPDX-License-Identifier: LGPL-2.1-or-later */
#include <assert.h>
#include "video/filter/ajn_sample_shared.h"

int main(void)
{
    int w, h, fps;
    int64_t generation = INT64_C(1) << 32;
    assert(ajn_sample_config(generation | (60 << 24) | (180 << 12) | 320, &w, &h, &fps));
    assert(w == 320 && h == 180 && fps == 60);
    assert(ajn_sample_config(generation | (1 << 24) | (1 << 12) | 1, &w, &h, &fps));
    assert(!ajn_sample_config(0, &w, &h, &fps));
    assert(!ajn_sample_config(generation, &w, &h, &fps));
    assert(!ajn_sample_config(generation | (61 << 24) | (36 << 12) | 64, &w, &h, &fps));
    assert(!ajn_sample_config(generation | (30 << 24) | (181 << 12) | 64, &w, &h, &fps));
    assert(!ajn_sample_config(generation | (30 << 24) | (36 << 12) | 321, &w, &h, &fps));
    assert(!ajn_sample_config((30 << 24) | (36 << 12) | 64, &w, &h, &fps));
    assert(!ajn_sample_config(INT64_MIN | generation | (30 << 24) | (36 << 12) | 64, &w, &h, &fps));
    return 0;
}
