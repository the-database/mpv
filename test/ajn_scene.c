/* Exercise the production bridge with real CPU frames and Windows events.
 * GPU download and inference are covered by separate playback qualification.
 * SPDX-License-Identifier: LGPL-2.1-or-later */
#include <stdio.h>
#include <string.h>
#include <windows.h>
#include "common/common.h"
#include "test_utils.h"
#include "video/mp_image.h"
#include "video/filter/ajn_scene.h"
#include "video/filter/ajn_scene_shared.h"

struct responder {
    struct ajn_scene_shared *h;
    HANDLE ready, reply;
    int decision;
    bool wrong_id;
};
static DWORD WINAPI respond(void *opaque)
{
    struct responder *r = opaque;
    mp_require(WaitForSingleObject(r->ready, 3000) == WAIT_OBJECT_0);
    while (InterlockedCompareExchange((volatile LONG *)&r->h->lock, 1, 0))
        SwitchToThread();
    // Samples must be the previous/current original frame, in that order.
    mp_require(r->h->width == 8 && r->h->height == 4);
    mp_require(r->h->pixels[0] < 4 && r->h->pixels[32] > 250);
    r->h->decision = r->decision;
    r->h->response_id = r->h->request_id + r->wrong_id;
    InterlockedExchange((volatile LONG *)&r->h->lock, 0);
    SetEvent(r->reply);
    return 0;
}
int main(void)
{
    char name[96];
    snprintf(name, sizeof(name), "Local\\AJN.Scene.%032lx", (unsigned long)GetCurrentProcessId());
    HANDLE mapping = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
                                        0, sizeof(struct ajn_scene_shared), name);
    mp_require(mapping && GetLastError() != ERROR_ALREADY_EXISTS);
    struct ajn_scene_shared *h = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(*h));
    mp_require(h);
    *h = (struct ajn_scene_shared){ .magic = AJN_SCENE_MAGIC, .version = 1,
        .capacity = AJN_SCENE_BYTES, .header_bytes = 256, .enabled = 1,
        .generation = 1, .lease_until_ms = GetTickCount64() + 30000,
        .width = 8, .height = 4, .budget_ms = 100 };
    char event_name[112];
    snprintf(event_name, sizeof(event_name), "%s.Ready", name);
    HANDLE ready = CreateEventA(NULL, FALSE, FALSE, event_name);
    snprintf(event_name, sizeof(event_name), "%s.Reply", name);
    HANDLE reply = CreateEventA(NULL, FALSE, FALSE, event_name);
    mp_require(ready && reply);
    struct ajn_scene *s = ajn_scene_create(NULL);
    mp_require(!GetModuleHandleW(L"ajn-mvtools.dll"));
    mp_require(!ajn_scene_connect(s, "Local\\AJN.Scene.invalid"));
    mp_require(ajn_scene_connect(s, name) && ajn_scene_active(s));
    struct mp_image *a = mp_image_alloc(IMGFMT_Y8, 16, 8);
    struct mp_image *b = mp_image_alloc(IMGFMT_Y8, 16, 8);
    mp_require(a && b);
    a->params.repr.levels = b->params.repr.levels = PL_COLOR_LEVELS_FULL;
    a->params.color.transfer = b->params.color.transfer = PL_COLOR_TRC_BT_1886;
    a->pts = 1; b->pts = 2;
    for (int y = 0; y < 8; y++) {
        memset(a->planes[0] + y * a->stride[0], 0, 16);
        memset(b->planes[0] + y * b->stride[0], 255, 16);
    }
    for (int decision = -1; decision <= 1; decision++) {
        struct responder r = {h, ready, reply, decision, false};
        HANDLE thread = CreateThread(NULL, 0, respond, &r, 0, NULL);
        mp_require(thread);
        assert_int_equal(ajn_scene_decide(s, a, b, true), decision);
        mp_require(WaitForSingleObject(thread, 3000) == WAIT_OBJECT_0);
        CloseHandle(thread);
        assert_int_equal(h->accepted, decision + 2);
    }
    int64_t epoch = h->epoch;
    h->lock = 1; // Reset must also invalidate a request during host readback.
    ajn_scene_reset(s);
    mp_require(h->epoch == epoch + 1 && !h->request_id && !h->response_id);
    h->lock = 0;
    assert_int_equal(ajn_scene_decide(s, a, b, false), -1);
    assert_int_equal(h->status, 5);
    b->params.color.transfer = PL_COLOR_TRC_PQ;
    assert_int_equal(ajn_scene_decide(s, a, b, true), -1);
    assert_int_equal(h->status, 4);
    b->params.color.transfer = PL_COLOR_TRC_BT_1886;
    struct responder wrong = {h, ready, reply, 1, true};
    HANDLE thread = CreateThread(NULL, 0, respond, &wrong, 0, NULL);
    assert_int_equal(ajn_scene_decide(s, a, b, true), -1);
    mp_require(WaitForSingleObject(thread, 3000) == WAIT_OBJECT_0);
    CloseHandle(thread);
    assert_int_equal(h->timeouts, 1);
    h->budget_ms = 5;
    assert_int_equal(ajn_scene_decide(s, a, b, true), -1);
    assert_int_equal(ajn_scene_decide(s, a, b, true), -1);
    mp_require(h->timeouts == 3 && h->status == 6 && !ajn_scene_active(s));
    assert_int_equal(ajn_scene_decide(s, a, b, true), -1);
    assert_int_equal(h->timeouts, 3);
    h->consecutive_timeouts = 0; h->lease_until_ms = GetTickCount64() - 1;
    mp_require(!ajn_scene_active(s));
    h->lease_until_ms = GetTickCount64() + 3000; h->enabled = 0;
    mp_require(!ajn_scene_active(s));
    mp_require(!GetModuleHandleW(L"ajn-mvtools.dll"));
    talloc_free(a); talloc_free(b);
    a = mp_image_alloc(IMGFMT_Y8, 64, 64);
    b = mp_image_alloc(IMGFMT_Y8, 64, 64);
    mp_require(a && b);
    a->pts = 1; b->pts = 2;
    for (int y = 0; y < 64; y++) {
        memset(a->planes[0] + y * a->stride[0], 16, 64);
        memset(b->planes[0] + y * b->stride[0], 235, 64);
    }
    h->algorithm = 1; h->thscd1 = 240; h->thscd2 = 130;
    h->enabled = 1; h->generation++; h->lease_until_ms = GetTickCount64() + 10000;
    assert_int_equal(ajn_scene_decide(s, a, b, true), 1);
    mp_require(h->status == 2 && h->total_blocks == 64 && h->changed_blocks == 64 && h->cut_pairs == 1);
    mp_require(GetModuleHandleW(L"ajn-mvtools.dll"));
    for (int y = 0; y < 64; y++) memset(b->planes[0] + y * b->stride[0], 16, 64);
    ajn_scene_reset(s);
    assert_int_equal(ajn_scene_decide(s, a, b, true), 0);
    talloc_free(a); talloc_free(b);
    a = mp_image_alloc(IMGFMT_NV12, 64, 64);
    b = mp_image_alloc(IMGFMT_NV12, 64, 64);
    mp_require(a && b);
    a->pts = 1; b->pts = 2;
    a->params.chroma_location = b->params.chroma_location = PL_CHROMA_LEFT;
    for (int y = 0; y < 64; y++) {
        memset(a->planes[0] + y * a->stride[0], 64, 64);
        memset(b->planes[0] + y * b->stride[0], 64, 64);
    }
    for (int y = 0; y < 32; y++) {
        memset(a->planes[1] + y * a->stride[1], 64, 64);
        memset(b->planes[1] + y * b->stride[1], 192, 64);
    }
    ajn_scene_reset(s);
    assert_int_equal(ajn_scene_decide(s, a, b, true), 1);
    mp_require(h->source_bits == 8 && h->block_sad_threshold == 360);
    mp_require(ajn_scene_connect(s, ""));
    mp_require(!GetModuleHandleW(L"ajn-mvtools.dll"));
    ajn_scene_destroy(s);
    talloc_free(a); talloc_free(b);
    UnmapViewOfFile(h); CloseHandle(mapping); CloseHandle(ready); CloseHandle(reply);
    puts("scene bridge: decision overrides, samples, reset, HDR fallback, stale reply, bounded timeouts and revocation passed");
    return 0;
}
