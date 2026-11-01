/* SPDX-License-Identifier: LGPL-2.1-or-later */
#include <assert.h>
#include <process.h>
#include "video/filter/ajn_sample_lifetime.h"

struct fixture {
    struct ajn_sample_lifetime life;
    HANDLE finish;
    LONG cleanups;
};

static void release(struct fixture *p)
{
    if (ajn_sample_lifetime_release(&p->life))
        InterlockedIncrement(&p->cleanups);
}

static unsigned __stdcall reader(void *opaque)
{
    struct fixture *p = opaque;
    // Model a driver that has not completed: retain the readback reference.
    assert(WaitForSingleObject(p->finish, 5000) == WAIT_OBJECT_0);
    release(p);
    return 0;
}

int main(void)
{
    for (int i = 0; i < 100; i++) {
        struct fixture p = {0};
        ajn_sample_lifetime_init(&p.life);
        assert(!ajn_sample_lifetime_stopping(&p.life));
        ajn_sample_lifetime_acquire(&p.life);
        p.finish = CreateEventW(NULL, TRUE, FALSE, NULL);
        assert(p.finish);
        HANDLE thread = (HANDLE)_beginthreadex(NULL, 0, reader, &p, 0, NULL);
        assert(thread);
        if (i % 2) {
            SetEvent(p.finish);
            assert(WaitForSingleObject(thread, 5000) == WAIT_OBJECT_0);
            assert(p.cleanups == 0); // Owner is still touching its wake handle.
        }
        ULONGLONG before = GetTickCount64();
        ajn_sample_lifetime_stop(&p.life);
        release(&p);
        assert(GetTickCount64() - before < 1000);
        assert(ajn_sample_lifetime_stopping(&p.life));
        if (!(i % 2)) {
            assert(p.cleanups == 0); // No early decoder-pool reuse on retirement.
            SetEvent(p.finish);
        }
        assert(WaitForSingleObject(thread, 5000) == WAIT_OBJECT_0);
        assert(p.cleanups == 1);
        CloseHandle(thread);
        CloseHandle(p.finish);
    }
    return 0;
}
