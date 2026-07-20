/*
 * This file is part of mpv.
 *
 * mpv is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * mpv is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with mpv.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

// WP-K5: per-osd_render() sub-phase accounting.
//
// The `sub-render` span (vo_gpu_next.c) brackets an ENTIRE osd_render() call,
// so it conflates at least five unrelated things: acquiring osd->lock, the
// bounded render-ahead miss-wait (pure waiting, not work), serving the
// worker's result by reference, the inline render fallback, and the non-sub
// OSD objects' async snapshot serves. These counters decompose it.
//
// Design notes (two traps this deliberately avoids):
//
//  1. update_overlays() -- and therefore osd_render() -- runs up to 3x per
//     presented frame. Emitting nested start/end SPANS would make the
//     downstream per-name histogram a blind mixture of parents and children
//     (this is exactly what already happened: sub/osd.c emits a span LITERALLY
//     NAMED "sub-render" per OSD object, inside vo_gpu_next.c's "sub-render",
//     and --dump-stats carries no context prefix, so both land in one bucket).
//     Instead every phase is accumulated in ns on the calling thread and
//     emitted as a set of `value` lines ONCE per osd_render() call. Each set
//     is internally complete and self-consistent, so three sets per frame
//     never double-count.
//
//  2. The residual must be honest. The phases form an explicit two-level tree
//     and only siblings are subtracted from a parent:
//
//       total  = osdlock + obj + other                      (top level)
//       obj    = fetch + inline + async + scale + objother  (per-object level)
//       fetch  = flock + fwait + fserve + ftail + fother    (inside the fetch)
//
//     osd_render() computes `other`/`objother`/`fother` in C, where the
//     nesting is unambiguous, rather than leaving the subtraction to the
//     parser.
//
// All timing is gated on `on`, which osd_render() sets only when
// mp_msg_test(MSGL_STATS) passes -- i.e. only under --dump-stats. With stats
// off every site below is a single predictable-branch load of a thread-local
// bool and no clock read at all.

#include <stdbool.h>
#include <stdint.h>

#include "osdep/compiler.h"  // thread_local

struct sub_phase_acc {
    bool on;              // an instrumented osd_render() is running on this thread

    // top level (siblings)
    int64_t osdlock_ns;   // osd_render(): acquiring osd->lock
    int64_t obj_ns;       // sum of all render_object() calls this osd_render()

    // inside obj_ns (siblings)
    int64_t fetch_ns;     // sub_ahead_get_bitmaps(), whole call
    int64_t inline_ns;    // dec_sub inline render fallback (render-ahead miss/off)
    int64_t async_ns;     // osd_{external,object}_render_async() snapshot serves
    int64_t scale_ns;     // capped-OSD rect rescale loop

    // inside inline_ns (siblings). The inline path is the ONLY one that can
    // put a synchronous libass render on the VO thread, so when it fires the
    // question is immediately "lock contention or actual rendering?".
    int64_t ilock_ns;     // acquiring dec_sub->lock (contends with decode)
    int64_t irender_ns;   // sd->driver->get_bitmaps(): the real render

    // inside irender_ns (siblings). WP-K4 measured libass ITSELF at 0.19 ms
    // mean / 3.2 ms max in outline-deferred mode, against a rig sd_ass render
    // of >115 ms -- so the split between libass and mpv's own post-processing
    // of libass's output is the whole question.
    int64_t assrender_ns; // ass_render_frame()
    int64_t pack_ns;      // mp_sub_packer_pack_ass()
    int64_t bcopy_ns;     // sub_bitmaps_copy() + rescale + mangle_colors()

    // inside pack_ns (siblings). MEASURED: pack_ns is 89% of the inline
    // render and libass itself is 0.33 ms, so this is where the 8K wall
    // actually is. In OUTLINES mode pack_ass() builds no atlas at all -- it
    // frees last frame's blob pool and ta_memdup()s this frame's blobs. Both
    // are per-part talloc traffic, a very different cost from the raw memcpy
    // volume it looks like on paper.
    int64_t packfree_ns;  // talloc_free(seg_ctx): previous frame's blob pool
    int64_t packcopy_ns;  // the ASS_Image walk + per-part ta_memdup
    int64_t pack_parts;   // parts emitted this pack
    int64_t pack_bytes;   // outline blob bytes copied this pack

    // inside fetch_ns (siblings)
    int64_t flock_ns;     // acquiring sub_ahead->lock
    int64_t fwait_ns;     // the bounded miss-wait (WAITING, not work)
    int64_t fserve_ns;    // payload_serve(): the serve-by-reference copy
    int64_t ftail_ns;     // post-unlock stats emit + retired-payload frees

    // Worst SINGLE occurrence, not the sum. Decisive for reading a large
    // total: N objects each waiting out a bounded miss-wait looks identical
    // to one object blocking for N intervals in the summed figures.
    int64_t objmax_ns;    // longest single render_object()
    int64_t fwaitmax_ns;  // longest single bounded miss-wait

    // counters
    int n_obj;            // objects rendered this osd_render()
    int n_fetch;          // render-ahead fetches
    int n_wait;           // fetches that entered the bounded miss-wait
    int n_inline;         // inline renders (must be 0 in forward play w/ ahead on)
    int n_async;          // async snapshot serves
};

extern thread_local struct sub_phase_acc sub_phase;
