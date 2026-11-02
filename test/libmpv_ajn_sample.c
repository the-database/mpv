/* Private sample initialization and worker module lifetime regression.
 * SPDX-License-Identifier: LGPL-2.1-or-later */
#include <windows.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpv/client.h>
#include "video/filter/ajn_sample_shared.h"

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "Failed: %s\n", #x); exit(1); } } while (0)
#define SYM(name) __typeof__(&mpv_##name) name = (void *)GetProcAddress(module, "mpv_" #name); REQUIRE(name)

int main(void)
{
    for (int valid = 0; valid < 2; valid++) {
        HMODULE module = LoadLibraryW(L"libmpv-2.dll");
        REQUIRE(module);
        SYM(create); SYM(initialize); SYM(set_option_string);
        SYM(command); SYM(wait_event); SYM(terminate_destroy);
        HANDLE mapping = NULL;
        struct ajn_sample_shared *shared = NULL;
        char filter[96] = "ajn-sample";
        if (valid) {
            mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
                                         0, sizeof(*shared), NULL);
            REQUIRE(mapping);
            shared = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(*shared));
            REQUIRE(shared);
            shared->magic = AJN_SAMPLE_MAGIC;
            shared->version = AJN_SAMPLE_VERSION;
            shared->header_bytes = 256;
            shared->capacity = AJN_SAMPLE_BYTES;
            snprintf(filter, sizeof(filter), "ajn-sample:mapping=%llu",
                     (unsigned long long)(uintptr_t)mapping);
        }
        mpv_handle *ctx = create();
        REQUIRE(ctx);
        REQUIRE(set_option_string(ctx, "config", "no") >= 0);
        REQUIRE(set_option_string(ctx, "vo", "null") >= 0);
        REQUIRE(set_option_string(ctx, "ao", "null") >= 0);
        REQUIRE(set_option_string(ctx, "vf", filter) >= 0);
        REQUIRE(initialize(ctx) >= 0);
        REQUIRE(command(ctx, (const char *[]){"loadfile", "av://lavfi:color=d=0.2", NULL}) >= 0);
        bool ended = false;
        ULONGLONG deadline = GetTickCount64() + 5000;
        while (!ended && GetTickCount64() < deadline) {
            mpv_event *event = wait_event(ctx, 0.02);
            ended = event->event_id == MPV_EVENT_END_FILE;
        }
        REQUIRE(ended);
        terminate_destroy(ctx);
        REQUIRE(FreeLibrary(module));
        deadline = GetTickCount64() + 5000;
        while (GetModuleHandleW(L"libmpv-2.dll") && GetTickCount64() < deadline)
            Sleep(10);
        REQUIRE(!GetModuleHandleW(L"libmpv-2.dll"));
        if (shared) REQUIRE(shared->status == AJN_SAMPLE_CLOSED);
        if (shared) UnmapViewOfFile(shared);
        if (mapping) CloseHandle(mapping);
    }
    return 0;
}
