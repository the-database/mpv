/* Exercise the built-in enhanced scene detector on Linux with real CPU frames.
 * Windows covers the same path, plus the addon bridge, in ajn_scene.c.
 * GPU download and inference are covered by separate playback qualification.
 * SPDX-License-Identifier: LGPL-2.1-or-later */
#include <dlfcn.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "common/common.h"
#include "osdep/timer.h"
#include "test_utils.h"
#include "video/mp_image.h"
#include "video/filter/ajn_scene.h"

static char detector[PATH_MAX];

static bool detector_loaded(void)
{
    void *m = dlopen(detector, RTLD_NOW | RTLD_NOLOAD);
    if (m) dlclose(m);
    return m;
}

static void await_unloaded(void)
{
    int64_t deadline = mp_time_ns() + MP_TIME_S_TO_NS(3);
    while (detector_loaded()) {
        mp_require(mp_time_ns() < deadline);
        mp_sleep_ns(MP_TIME_MS_TO_NS(1));
    }
}

static int await_decision(struct ajn_scene *s, struct mp_image *a,
                          struct mp_image *b, int expected)
{
    int64_t deadline = mp_time_ns() + MP_TIME_S_TO_NS(3);
    int decision;
    do {
        decision = ajn_scene_decide(s, a, b, true);
        if (decision == expected) return decision;
        mp_require(mp_time_ns() < deadline);
        mp_sleep_ns(MP_TIME_MS_TO_NS(1));
    } while (true);
}

int main(void)
{
    // The worker loads the detector beside the module carrying it, which is
    // this executable here (meson copies the library next to it).
    ssize_t n = readlink("/proc/self/exe", detector, sizeof(detector) - 1);
    mp_require(n > 0);
    detector[n] = 0;
    char *slash = strrchr(detector, '/');
    mp_require(slash && (size_t)(slash - detector) + 32 < sizeof(detector));
    strcpy(slash + 1, "ajn-mvtools.so");
    char unavailable[PATH_MAX + 16];
    snprintf(unavailable, sizeof(unavailable), "%s.unavailable", detector);

    struct ajn_scene *s = ajn_scene_create(NULL);
    mp_require(!detector_loaded());
    // The addon bridge is Windows-only.
    mp_require(!ajn_scene_connect(s, "Local\\AJN.Scene.00000000000000000000000000000000"));
    mp_require(!ajn_scene_active(s));
    ajn_scene_destroy(s);

    // A missing optional library must fail startup before enabling look-ahead.
    mp_require(!rename(detector, unavailable));
    s = ajn_scene_create(NULL);
    bool missing_started = ajn_scene_start_enhanced(s);
    bool missing_active = ajn_scene_active(s);
    ajn_scene_destroy(s);
    mp_require(!rename(unavailable, detector));
    mp_require(!missing_started && !missing_active);
    mp_require(!detector_loaded());

    struct mp_image *a = mp_image_alloc(IMGFMT_NV12, 64, 64);
    struct mp_image *b = mp_image_alloc(IMGFMT_NV12, 64, 64);
    mp_require(a && b);
    a->pts = 1; b->pts = 2;
    a->params.chroma_location = b->params.chroma_location = PL_CHROMA_LEFT;
    for (int y = 0; y < 64; y++) {
        memset(a->planes[0] + y * a->stride[0], 64, 64);
        memset(b->planes[0] + y * b->stride[0], 64, 64);
    }
    for (int y = 0; y < 32; y++)
        memset(a->planes[1] + y * a->stride[1], 64, 64);

    s = ajn_scene_create(NULL);
    mp_require(ajn_scene_start_enhanced(s) && ajn_scene_active(s));
    mp_require(detector_loaded());
    for (int cut = 1; cut >= 0; cut--) {
        for (int y = 0; y < 32; y++)
            memset(b->planes[1] + y * b->stride[1], cut ? 192 : 64, 64);
        ajn_scene_reset(s);
        assert_int_equal(ajn_scene_decide(s, a, b, true), -1);
        ajn_scene_submit(s, a, b);
        assert_int_equal(await_decision(s, a, b, cut), cut);
        mp_require(detector_loaded());
    }
    ajn_scene_reset(s);
    assert_int_equal(ajn_scene_decide(s, a, b, true), -1);
    // Destroying must not wait for the worker, which then retires the library.
    int64_t began = mp_time_ns();
    ajn_scene_destroy(s);
    mp_require(mp_time_ns() - began < MP_TIME_MS_TO_NS(50));
    await_unloaded();
    mp_require(!ajn_scene_active(NULL));

    talloc_free(a); talloc_free(b);
    puts("built-in enhanced mode: dependency probe, opt-in, cut/continuity, reset and library retirement passed");
    return 0;
}
