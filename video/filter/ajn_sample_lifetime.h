/* Private Windows sample lifetime. SPDX-License-Identifier: LGPL-2.1-or-later */
#pragma once
#include <windows.h>

// The filter and readback thread each own one reference. Requesting retirement
// cannot wait for the readback lock or driver; the final owner cleans up only
// after both have stopped using the state. The enclosing object is detached
// from the filter's allocator before the thread starts.
struct ajn_sample_lifetime {
    LONG references, stopping;
};

static inline void ajn_sample_lifetime_init(struct ajn_sample_lifetime *p)
{
    p->references = 1;
    p->stopping = 0;
}

static inline void ajn_sample_lifetime_acquire(struct ajn_sample_lifetime *p)
{
    InterlockedIncrement(&p->references);
}

static inline int ajn_sample_lifetime_release(struct ajn_sample_lifetime *p)
{
    return InterlockedDecrement(&p->references) == 0;
}

static inline int ajn_sample_lifetime_stopping(struct ajn_sample_lifetime *p)
{
    return InterlockedCompareExchange(&p->stopping, 0, 0) != 0;
}

static inline void ajn_sample_lifetime_stop(struct ajn_sample_lifetime *p)
{
    InterlockedExchange(&p->stopping, 1);
}
