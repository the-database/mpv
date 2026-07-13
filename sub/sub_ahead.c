/*
 * Subtitle render-ahead worker.
 *
 * Renders ASS subtitle bitmaps for upcoming frames on a background thread so the
 * VO display path serves a pre-rendered frame instead of running the ~33ms
 * libass render + atlas pack inline (which, stacked with the GPU upload, blows
 * the per-frame budget at 8K).
 *
 * Design (this is a rebuild; the first attempt was reverted for the two reasons
 * called out below):
 *
 * 1. The worker owns a SECOND, fully independent sd_ass instance (`worker_sd`)
 *    with its own ASS_Library/Renderer/Track. It renders that, so it NEVER
 *    takes the decoder lock (`sub->lock`) and never touches the decoder's track.
 *    -> fixes failure #1 (the old worker held sub->lock across the render, and
 *    sub_read_packets takes sub->lock every frame, so the decode path stalled).
 *    The own ASS_Library also means no font-cache race with the VO's fallback
 *    renderer (font_lock/fontselect are per-library).
 *
 * 2. The worker's libass thread count is capped separately
 *    (--sub-render-ahead-threads) so it does not starve the VO/upscaler.
 *    -> fixes failure #2 (drops scaled up with the old worker's thread count).
 *
 * 3. No `sub_pts <= last_pkt_pts` gate (it busy-spun on long-lived events); the
 *    worker waits on a cond when its window is full.
 *
 * Phase 2 -- the VO thread NEVER pays an inline render while the worker runs:
 *
 * 4. A miss no longer falls back to a synchronous render on the VO thread (the
 *    historical failure mode: one seek = ~440 misses = ~440 sequential ~90ms
 *    inline renders). Instead the fetch does a bounded wait
 *    (--sub-render-ahead-miss-wait, default one frame interval) for the worker
 *    to deliver, and past the deadline serves the nearest already-rendered
 *    EARLIER frame from the ring ("stale-serve": the viewer sees the previous
 *    subtitle content for a frame) or, when nothing earlier exists, no
 *    subtitles for that frame.
 *    Stale rules: the ring is the stale source, and it is dropped on
 *    flush/reset/geometry/timing changes -- so stale can never cross a seek,
 *    a track change or a resize. Right after a reset, when nothing has been
 *    rendered yet, the choice is deliberately "short wait, then EMPTY": a
 *    subtitle-less frame is a glitch, pre-seek content at a post-seek position
 *    is wrong output.
 *
 * 5. Seeks pre-warm instead of collapsing: sub_reset -> sub_ahead_flush drops
 *    everything, and the decode path immediately hints the seek-target pts
 *    (sub_ahead_hint_pts). The worker starts rendering the target window
 *    (starting AT the target, i=0) before the VO's first post-seek fetch.
 *    ra-prewarm counts renders done in this mode.
 *
 * 6. Adaptive banking: the worker tracks an EMA of its per-frame render cost.
 *    When the depth window is full and the EMA shows headroom (cheap renders,
 *    e.g. the libass front-end-only cost in --sub-gpu-raster mode), it keeps
 *    rendering past `depth` up to --sub-render-ahead-max-frames, banking slack
 *    for dense passages. Pre-warm always preempts banking: after a flush the
 *    target selection starts over at the (new) base pts, and a stale in-flight
 *    bank render is discarded by the gen check.
 *
 * Locking: `lock` protects the ring, the cursor/geometry, the gen, the timing
 * snapshot, the packet queue, and the pending video-params. It is never held
 * across a render. The decode path only ever takes `lock` briefly to enqueue a
 * packet copy. All mutation of `worker_sd` happens on the worker thread.
 * `ring_added` is broadcast by the worker after each store; a missed fetch
 * waits on it (with a deadline) instead of rendering inline.
 *
 * OUTLINES lifetime: in --sub-gpu-raster mode the parts of a sub_bitmaps carry
 * tile/segment blobs owned by the producing packer's seg_ctx, which is only
 * valid until that sd's NEXT pack. Ring entries outlive that, so
 * pin_outline_blobs() reparents private blob copies onto the sub_bitmaps
 * itself ONCE, on the worker thread, before the render is stored.
 *
 * Serve-by-reference (WP-H1a): a completed render is immutable -- the worker
 * only ever stores whole entries and frees them on eviction, it never writes
 * into a stored sub_bitmaps -- so the fetch path serves it WITHOUT deep-
 * copying the payload. Deep-copying was measured as the dominant VO-thread
 * cost on dense OUTLINES content (thousands of parts x pinned tile blobs =
 * MBs of memcpy per changed frame, all inside sub_ahead_get_bitmaps). The
 * render was already banked; the copy was the bill. Instead each stored
 * render lives in a refcounted payload: the ring holds one reference,
 * every served sub_bitmaps pins one more (released by a talloc destructor
 * when the consumer frees it). The serve itself allocates only a small
 * struct plus a private parts[] array -- consumers legitimately write to
 * the returned STRUCT (osd.c stamps render_index/change_id) and osd.c's
 * capped-OSD path may rescale part rects, so the parts array is per-serve;
 * the payload (blobs, packed image) is shared read-only. Eviction while a
 * serve is outstanding just drops the ring's reference; the memory is
 * reclaimed when the last consumer copy is freed.
 */

#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>

#include "common/msg.h"
#include "dec_sub.h"
#include "demux/packet.h"
#include "misc/mp_assert.h"
#include "options/options.h"
#include "osd.h"
#include "osdep/threads.h"
#include "osdep/timer.h"
#include "sd.h"
#include "sub_ahead.h"
#include "video/mp_image.h"

// Bank only while a render costs less than this fraction of a frame interval
// (i.e. the worker has real headroom and banking cannot starve keeping up).
#define BANK_HEADROOM 0.5

// Never stale-serve content older than this (seconds). Bounds how long an
// outdated subtitle can stay on screen when the worker is badly behind.
#define STALE_MAX_AGE 2.0

// A completed render, shared by reference between the ring and served copies
// (see "Serve-by-reference" above). bmp is a talloc child of the payload and
// is IMMUTABLE once the payload exists. The refcount is atomic because the
// last unref can happen on whatever thread frees the served copy (the VO
// thread's talloc_free of its overlay list), without taking the ring lock.
struct ahead_payload {
    _Atomic int refs;
    struct sub_bitmaps *bmp;
};

struct sub_ahead_entry {
    bool valid;
    double video_pts;            // raw (pre-conversion) video pts -- the key
    struct mp_osd_res dim;
    int format;
    uint64_t gen;
    uint64_t content_id;         // for change_id unification
    bool prefilled;              // consumed by a serve or by the VO's GPU
                                 // glyph pre-fill (sub_ahead_peek_prefill)
    struct ahead_payload *pl;    // ring's reference; NULL = "no subtitles"
};

static struct ahead_payload *payload_new(struct sub_bitmaps *bmp)
{
    if (!bmp)
        return NULL;
    struct ahead_payload *pl = talloc_zero(NULL, struct ahead_payload);
    atomic_init(&pl->refs, 1);
    pl->bmp = talloc_steal(pl, bmp);
    return pl;
}

static void payload_unref(struct ahead_payload *pl)
{
    if (pl && atomic_fetch_sub_explicit(&pl->refs, 1, memory_order_acq_rel) == 1)
        talloc_free(pl);
}

// Hidden talloc child of a served sub_bitmaps; its destructor releases the
// payload pin when the consumer talloc_free()s the serve.
struct ahead_pin {
    struct ahead_payload *pl;
};

static void ahead_pin_destroy(void *ptr)
{
    struct ahead_pin *pin = ptr;
    payload_unref(pin->pl);
}

// Serve a payload by reference: a fresh sub_bitmaps STRUCT (callers write
// render_index/change_id into it) with a private parts[] array (osd.c's
// capped-OSD path may edit part rects) whose blob/bitmap/packed pointers
// alias the pinned payload. Free with talloc_free(); NULL when the render
// had no parts (same contract as the old sub_bitmaps_copy serve).
static struct sub_bitmaps *payload_serve(struct ahead_payload *pl)
{
    struct sub_bitmaps *in = pl->bmp;
    if (!in || !in->num_parts)
        return NULL;
    struct sub_bitmaps *res = talloc(NULL, struct sub_bitmaps);
    *res = *in;
    res->parts = talloc_memdup(res, in->parts,
                               sizeof(in->parts[0]) * in->num_parts);
    struct ahead_pin *pin = talloc(res, struct ahead_pin);
    pin->pl = pl;
    atomic_fetch_add_explicit(&pl->refs, 1, memory_order_relaxed);
    talloc_set_destructor(pin, ahead_pin_destroy);
    return res;
}

struct sub_ahead {
    struct mp_log *log;
    struct sd *worker_sd;        // owned: the independent libass instance

    mp_thread thread;
    bool worker_running;
    mp_mutex lock;
    mp_cond wakeup;              // fetch/decode path -> worker
    mp_cond ring_added;          // worker -> a fetch waiting out a miss
    bool terminate;

    int depth;                   // lookahead frames (the guaranteed window)
    int max_frames;              // bank cap (>= depth); ring is sized for it
    double miss_wait_ms;         // <0 = auto (one frame interval), 0 = off
    int64_t debug_slow_ns;       // MPV_SUB_AHEAD_SLOW_MS: debug worker slowdown
    double video_fps;
    double vo_pts;               // latest raw video pts the VO asked for
    double hint_pts;             // pre-warm target from the decode path
    struct mp_osd_res cur_dim;
    int cur_format;
    uint64_t gen;                // bumped on any invalidation

    struct sub_ahead_entry *ring;
    int ring_len;

    // pts mapping snapshot (mirrors dec_sub.c pts_to_subtitle), so the worker
    // converts target video pts -> subtitle pts without the decoder lock.
    double sub_speed;
    float delay;
    int play_dir;

    // packet queue: raw event packet copies for the worker to decode.
    struct demux_packet **queue;
    int num_queue;

    // Smallest subtitle pts among packets enqueued since the worker last had
    // the queue fully drained. A render whose target is at/after it raced a
    // new event and is discarded at store time (the entry re-renders with the
    // event decoded). INFINITY = nothing pending.
    double newpkt_min_pts;

    // pending controls to apply to worker_sd on the worker thread.
    bool params_pending;
    bool params_set;             // pending_params holds a real value to compare
    struct mp_image_params pending_params;
    bool reset_pending;

    // change_id unification across worker hits (osd.c accumulates change_id).
    uint64_t served_change_id;
    uint64_t last_served_content;
    bool have_last_served;

    // worker render-cost EMA (seconds per frame), for the banking gate.
    double render_ema;

    int inline_active;           // VO is inline-rendering; worker must not race it

    // Render-ahead health counters. Cheap integer increments, all touched under
    // `lock`. Per fetch (disjoint): served = pre-rendered hit; empty =
    // rendered-but-no-subs; miss = not served fresh even after the bounded
    // wait. stale = misses resolved from an earlier rendered frame (subset of
    // miss). wait = fetches rescued by the bounded wait (subset of
    // served/empty). prewarm = worker renders in pre-warm mode. inline =
    // inline renders on the VO thread with the worker active (0 in normal
    // operation; see sub_ahead_note_inline). bank = current banked count
    // (gauge, emitted on change). Surfaced as MP_STATS ra-* value lines + a
    // periodic MSGL_V [render-ahead] line (prefix kept parse_stats.py-
    // compatible).
    long dbg_served, dbg_empty, dbg_miss;
    long dbg_stale, dbg_wait, dbg_prewarm, dbg_inline;
    int dbg_bank_last;

    // WP-H6 (item 2): fetches whose NON-wait phase (lock acquisition + ring
    // find + serve) exceeded RA_FETCH_STALL_NS on the VO thread. Round 3 had
    // an uncounted 256.7 ms video-draw whose whole cost sat between the VO
    // fetch entering sub_ahead_get_bitmaps and its ra-* stats emission with
    // miss/wait NOT incrementing -- i.e. blocked on `lock` (or an equally
    // invisible pre-serve step), not the bounded miss-wait. Gated ==0 in
    // acceptance; a fire logs a phase breakdown for one-look attribution.
    long dbg_fetch_stall;

    // WP-H6 (item 2): payloads whose last reference dropped under `lock` are
    // parked here and freed AFTER unlock. Freeing a dense OUTLINES frame
    // (thousands of pinned blob children) is a long talloc walk; doing it
    // under `lock` blocks the VO fetch path for the whole free. The array is
    // NULL-parented (standalone): it is stolen under the lock and freed
    // outside it, so it must never live in `a`'s talloc child list (which is
    // only safe to touch under `lock`).
    struct ahead_payload **retire;
    int num_retire;
};

// WP-H6 (item 2): non-wait fetch phases above this are a counted VO stall.
#define RA_FETCH_STALL_NS (20 * INT64_C(1000000))

// Drop a ring/loop reference under `lock`: the actual talloc_free of the
// payload tree is deferred to retire_flush() outside the lock.
static void payload_retire(struct sub_ahead *a, struct ahead_payload *pl)
{
    if (pl && atomic_fetch_sub_explicit(&pl->refs, 1, memory_order_acq_rel) == 1)
        MP_TARRAY_APPEND(NULL, a->retire, a->num_retire, pl);
}

// Steal the parked payloads (call with `lock` held)...
static struct ahead_payload **retire_steal(struct sub_ahead *a, int *n)
{
    struct ahead_payload **list = a->retire;
    *n = a->num_retire;
    a->retire = NULL;
    a->num_retire = 0;
    return list;
}

// ...and free them (call with `lock` NOT held).
static void retire_flush(struct ahead_payload **list, int n)
{
    for (int i = 0; i < n; i++)
        talloc_free(list[i]);
    talloc_free(list);
}

static double ahead_pts_to_subtitle(struct sub_ahead *a, double pts)
{
    if (pts == MP_NOPTS_VALUE)
        return pts;
    double speed = a->sub_speed != 0 ? a->sub_speed : 1.0;
    return (pts * a->play_dir - a->delay) / speed;
}

static double ahead_interval(struct sub_ahead *a)
{
    return a->video_fps > 0 ? 1.0 / a->video_fps : 1.0 / 24.0;
}

static void ahead_clear(struct sub_ahead *a)
{
    for (int n = 0; n < a->ring_len; n++) {
        if (a->ring[n].valid) {
            payload_retire(a, a->ring[n].pl);   // freed outside `lock`
            a->ring[n] = (struct sub_ahead_entry){0};
        }
    }
}

static void queue_flush(struct sub_ahead *a)
{
    for (int n = 0; n < a->num_queue; n++)
        talloc_free(a->queue[n]);
    a->num_queue = 0;
}

// In OUTLINES mode each part's coverage blob is owned by the packer that
// produced it and becomes invalid on that packer's next pack. A ring entry
// outlives that, so the worker reparents private blob copies onto the
// sub_bitmaps itself ONCE before storing it; serves then share those pinned
// blobs by reference (payload_serve). No-op for all other formats (their
// pixel data is a refcounted mp_image owned by the same payload).
static void pin_outline_blobs(struct sub_bitmaps *b)
{
    if (!b || b->format != SUBBITMAP_LIBASS_OUTLINES)
        return;
    for (int n = 0; n < b->num_parts; n++) {
        struct sub_bitmap *p = &b->parts[n];
        if (p->libass.outline && p->libass.n_outline > 0) {
            p->libass.outline =
                talloc_memdup(b, (void *)p->libass.outline,
                              (size_t)p->libass.n_outline * sizeof(int32_t));
        }
    }
}

// Index of a current-gen entry for raw video pts V at the given geometry, or -1.
// V is matched within half a frame interval.
static int ahead_find(struct sub_ahead *a, double V, struct mp_osd_res dim,
                      int format)
{
    double tol = ahead_interval(a) * 0.5;
    for (int n = 0; n < a->ring_len; n++) {
        struct sub_ahead_entry *e = &a->ring[n];
        if (e->valid && e->gen == a->gen && e->format == format &&
            osd_res_equals(e->dim, dim) && fabs(e->video_pts - V) < tol)
            return n;
    }
    return -1;
}

// Stale-serve source: the nearest already-rendered frame strictly BEFORE V at
// the current gen/geometry, or -1. Because the gen bumps and the ring clears on
// flush/reset/track teardown/resize/timing changes, this can never return
// pre-seek (or otherwise cross-boundary) content -- only slightly older frames
// of the current playback run. Age-capped so a badly-behind worker can't leave
// an ancient line on screen.
static int ahead_find_stale(struct sub_ahead *a, double V, struct mp_osd_res dim,
                            int format)
{
    double tol = ahead_interval(a) * 0.25;
    int best = -1;
    double best_pts = -INFINITY;
    for (int n = 0; n < a->ring_len; n++) {
        struct sub_ahead_entry *e = &a->ring[n];
        if (e->valid && e->gen == a->gen && e->format == format &&
            osd_res_equals(e->dim, dim) &&
            e->video_pts < V - tol && V - e->video_pts <= STALE_MAX_AGE &&
            e->video_pts > best_pts)
        {
            best = n;
            best_pts = e->video_pts;
        }
    }
    return best;
}

// Store a freshly rendered frame, taking ownership of bmp. Reuses a free slot,
// else evicts the oldest (smallest) video pts.
static void ahead_store(struct sub_ahead *a, double V, struct mp_osd_res dim,
                        int format, uint64_t gen, uint64_t content_id,
                        struct sub_bitmaps *bmp)
{
    int slot = -1;
    double oldest = INFINITY;
    for (int n = 0; n < a->ring_len; n++) {
        if (!a->ring[n].valid) {
            slot = n;
            break;
        }
        if (a->ring[n].video_pts < oldest) {
            oldest = a->ring[n].video_pts;
            slot = n;
        }
    }
    if (a->ring[slot].valid)
        payload_retire(a, a->ring[slot].pl);   // freed outside `lock`
    a->ring[slot] = (struct sub_ahead_entry){
        .valid = true, .video_pts = V, .dim = dim, .format = format,
        .gen = gen, .content_id = content_id, .pl = payload_new(bmp),
    };
}

// Compute the ra-bank gauge (number of rendered frames beyond the guaranteed
// depth window). Called under lock; returns true when it changed since the
// last call (the caller emits the MP_STATS line AFTER unlocking -- WP-H6
// item 2: no log/stats I/O under `lock`, a blocking write there stalls the
// VO fetch path).
static bool ahead_bank_gauge(struct sub_ahead *a, int *out_count)
{
    double base = a->vo_pts != MP_NOPTS_VALUE ? a->vo_pts : a->hint_pts;
    int count = 0;
    if (base != MP_NOPTS_VALUE) {
        double horizon = base + (a->depth + 0.5) * ahead_interval(a);
        for (int n = 0; n < a->ring_len; n++) {
            struct sub_ahead_entry *e = &a->ring[n];
            if (e->valid && e->gen == a->gen && e->video_pts > horizon)
                count++;
        }
    }
    *out_count = count;
    if (count == a->dbg_bank_last)
        return false;
    a->dbg_bank_last = count;
    return true;
}

// Snapshot of the health counters, taken under lock, emitted outside it.
struct ahead_dbg_snap {
    long served, empty, miss, stale, wait, prewarm, inline_, fetch_stall;
    int bank;
    bool bank_changed;
    bool verbose_due;
};

static struct ahead_dbg_snap ahead_dbg_snapshot(struct sub_ahead *a)
{
    struct ahead_dbg_snap s = {
        .served = a->dbg_served, .empty = a->dbg_empty, .miss = a->dbg_miss,
        .stale = a->dbg_stale, .wait = a->dbg_wait, .prewarm = a->dbg_prewarm,
        .inline_ = a->dbg_inline, .fetch_stall = a->dbg_fetch_stall,
    };
    s.bank_changed = ahead_bank_gauge(a, &s.bank);
    s.verbose_due = s.served && s.served % 250 == 0;
    return s;
}

static void ahead_dbg_emit(struct sub_ahead *a, const struct ahead_dbg_snap *s)
{
    MP_STATS(a, "value %ld ra-served", s->served);
    MP_STATS(a, "value %ld ra-empty", s->empty);
    MP_STATS(a, "value %ld ra-miss", s->miss);
    MP_STATS(a, "value %ld ra-stale", s->stale);
    MP_STATS(a, "value %ld ra-wait", s->wait);
    MP_STATS(a, "value %ld ra-inline", s->inline_);
    MP_STATS(a, "value %ld ra-fetch-stall", s->fetch_stall);
    if (s->bank_changed)
        MP_STATS(a, "value %d ra-bank", s->bank);
    if (s->verbose_due) {
        // Prefix (served= empty= miss=) is parsed by TOOLS/subtest/
        // parse_stats.py; extend only at the end.
        MP_VERBOSE(a, "[render-ahead] served=%ld empty=%ld miss=%ld stale=%ld "
                   "wait=%ld prewarm=%ld inline=%ld fstall=%ld\n",
                   s->served, s->empty, s->miss, s->stale,
                   s->wait, s->prewarm, s->inline_, s->fetch_stall);
    }
}

static MP_THREAD_VOID sub_ahead_thread(void *ptr)
{
    struct sub_ahead *a = ptr;
    mp_thread_set_name("sub/ahead");

    mp_mutex_lock(&a->lock);
    while (!a->terminate) {
        // Reset the worker's own track (seek) on this thread.
        if (a->reset_pending) {
            a->reset_pending = false;
            mp_mutex_unlock(&a->lock);
            if (a->worker_sd->driver->reset)
                a->worker_sd->driver->reset(a->worker_sd);
            mp_mutex_lock(&a->lock);
            continue;
        }

        // Apply any pending video params on this thread (worker_sd is only ever
        // touched here).
        if (a->params_pending) {
            struct mp_image_params p = a->pending_params;
            a->params_pending = false;
            mp_mutex_unlock(&a->lock);
            a->worker_sd->driver->control(a->worker_sd, SD_CTRL_SET_VIDEO_PARAMS, &p);
            mp_mutex_lock(&a->lock);
            continue;
        }

        // Drain queued packets into the worker's own track.
        if (a->num_queue) {
            struct demux_packet *pkt = a->queue[0];
            MP_TARRAY_REMOVE_AT(a->queue, a->num_queue, 0);
            mp_mutex_unlock(&a->lock);
            a->worker_sd->driver->decode(a->worker_sd, pkt);
            talloc_free(pkt);
            mp_mutex_lock(&a->lock);
            continue;
        }

        // Pause while the VO is inline-rendering (only possible in the
        // degraded reverse-playback mode), so the two heavy parallel renders
        // don't oversaturate the cores.
        if (a->inline_active) {
            mp_cond_wait(&a->wakeup, &a->lock);
            continue;
        }

        // Pre-warm mode: no fetch since the last flush (seek) or startup; work
        // from the decode path's hint so the target window is rendered before
        // the VO's first fetch. Includes i=0 -- the first post-seek fetch must
        // find its own frame, there is no inline fallback anymore.
        bool prewarm = a->vo_pts == MP_NOPTS_VALUE;
        double base = prewarm ? a->hint_pts : a->vo_pts;
        int depth = a->depth;
        if (depth <= 0 || base == MP_NOPTS_VALUE || a->cur_format == 0) {
            mp_cond_wait(&a->wakeup, &a->lock);
            continue;
        }
        struct mp_osd_res dim = a->cur_dim;
        int format = a->cur_format;
        uint64_t gen = a->gen;
        double interval = ahead_interval(a);

        // The queue is drained at this point, so every event enqueued so far
        // is in the worker's track; only packets arriving DURING the next
        // render can invalidate it (checked against newpkt_min_pts at store).
        a->newpkt_min_pts = INFINITY;

        // Nearest frame not yet in the ring, starting AT the base frame (i=0):
        // a missed fetch blocks on ring_added, so the current frame must be
        // renderable by the worker. (The historical i>=1 rule existed to avoid
        // racing the VO's inline render of the same frame; there is no inline
        // render anymore.)
        double target = MP_NOPTS_VALUE;
        for (int i = 0; i <= depth; i++) {
            double V = base + i * interval;
            if (ahead_find(a, V, dim, format) < 0) {
                target = V;
                break;
            }
        }
        // Adaptive banking: the depth window is full and renders are cheap
        // (EMA well under a frame interval; true in --sub-gpu-raster mode
        // where the worker only pays the libass front-end) -> keep going up
        // to the bank cap. Never while pre-warming: the guaranteed window
        // around a fresh seek target always comes first.
        if (target == MP_NOPTS_VALUE && !prewarm && a->max_frames > depth &&
            a->render_ema > 0 && a->render_ema < interval * BANK_HEADROOM)
        {
            for (int i = depth + 1; i <= a->max_frames; i++) {
                double V = base + i * interval;
                if (ahead_find(a, V, dim, format) < 0) {
                    target = V;
                    break;
                }
            }
        }
        if (target == MP_NOPTS_VALUE) {
            // Window full; wait for the VO to advance (or a flush).
            mp_cond_wait(&a->wakeup, &a->lock);
            continue;
        }

        double sub_pts = ahead_pts_to_subtitle(a, target);
        mp_mutex_unlock(&a->lock);

        // Render on the worker's own sd -- no decoder lock involved. draw_flags=0
        // (no OSD_DRAW_SUB_ONLY) so --sub-render-res-limit applies to the worker's
        // renders exactly as it does on the inline VO path.
        int64_t t0 = mp_time_ns();
        struct sub_bitmaps *bmp =
            a->worker_sd->driver->get_bitmaps(a->worker_sd, dim, format, sub_pts, 0);
        if (a->debug_slow_ns)     // debug: artificially slow worker (see create)
            mp_sleep_ns(a->debug_slow_ns);
        double render_time = (mp_time_ns() - t0) / 1e9;
        uint64_t content_id = bmp ? bmp->change_id : 0;
        pin_outline_blobs(bmp);   // ring entries outlive the packer's blobs

        mp_mutex_lock(&a->lock);
        a->render_ema = a->render_ema <= 0
            ? render_time : 0.8 * a->render_ema + 0.2 * render_time;
        // A packet that arrived mid-render may add events at/before this pts:
        // discard, the entry is still missing and re-renders next iteration
        // (after the queue drain decodes the packet).
        bool late_pkt = sub_pts >= a->newpkt_min_pts - 1e-9;
        struct sub_bitmaps *stale_bmp = NULL;
        long prewarm_snap = -1;
        bool bank_changed = false;
        int bank = 0;
        if (gen == a->gen && format == a->cur_format &&
            osd_res_equals(dim, a->cur_dim) && !late_pkt)
        {
            ahead_store(a, target, dim, format, gen, content_id, bmp);
            if (prewarm)
                prewarm_snap = ++a->dbg_prewarm;
            bank_changed = ahead_bank_gauge(a, &bank);
            mp_cond_broadcast(&a->ring_added);  // release a fetch waiting out a miss
        } else {
            stale_bmp = bmp;     // stale (flush/resize/late packet during render)
        }
        // WP-H6 (item 2): stats/log writes and payload frees happen with the
        // lock RELEASED. Round 3's uncounted 256.7 ms VO fetch stall was time
        // spent blocked on `lock` while the holder did invisible work; the
        // remaining under-lock work above is now O(ring) bookkeeping only.
        int nretire = 0;
        struct ahead_payload **retired = retire_steal(a, &nretire);
        mp_mutex_unlock(&a->lock);
        talloc_free(stale_bmp);
        retire_flush(retired, nretire);
        if (prewarm_snap >= 0)
            MP_STATS(a, "value %ld ra-prewarm", prewarm_snap);
        if (bank_changed)
            MP_STATS(a, "value %d ra-bank", bank);
        mp_mutex_lock(&a->lock);
    }
    mp_mutex_unlock(&a->lock);
    MP_THREAD_RETURN();
}

struct sub_ahead *sub_ahead_create(struct dec_sub *sub, struct sd *worker_sd,
                                   int depth, int order)
{
    if (depth <= 0 || !worker_sd || !worker_sd->driver->get_bitmaps)
        return NULL;

    struct mp_subtitle_opts *opts = worker_sd->opts;

    struct sub_ahead *a = talloc_zero(NULL, struct sub_ahead);
    a->log = worker_sd->log;
    a->worker_sd = worker_sd;
    a->depth = depth;
    // Bank cap: 0 = auto (4x depth). The ring is sized for the cap (+2 slack
    // so a couple of just-passed frames survive as the stale-serve source).
    int max_frames = opts->sub_render_ahead_max_frames;
    a->max_frames = max_frames > 0 ? MPMAX(max_frames, depth) : depth * 4;
    a->miss_wait_ms = opts->sub_render_ahead_miss_wait;
    a->vo_pts = MP_NOPTS_VALUE;
    a->hint_pts = MP_NOPTS_VALUE;
    a->newpkt_min_pts = INFINITY;
    a->cur_format = 0;
    a->sub_speed = 1.0;
    a->play_dir = 1;
    a->video_fps = 0;
    a->ring_len = a->max_frames + 2;
    a->ring = talloc_zero_array(a, struct sub_ahead_entry, a->ring_len);

    // Debug knob (used by the stale-serve tests): sleep this many ms after
    // every worker render, simulating a worker that cannot keep up. Not for
    // normal use; documented next to --sub-render-ahead-miss-wait.
    const char *slow = getenv("MPV_SUB_AHEAD_SLOW_MS");
    if (slow && atoi(slow) > 0) {
        a->debug_slow_ns = MP_TIME_MS_TO_NS(atoi(slow));
        MP_WARN(a, "MPV_SUB_AHEAD_SLOW_MS=%d: debug worker slowdown active.\n",
                atoi(slow));
    }

    mp_mutex_init(&a->lock);
    mp_cond_init(&a->wakeup);
    mp_cond_init(&a->ring_added);

    if (mp_thread_create(&a->thread, sub_ahead_thread, a) != 0) {
        MP_WARN(a, "Failed to start render-ahead worker.\n");
        mp_cond_destroy(&a->ring_added);
        mp_cond_destroy(&a->wakeup);
        mp_mutex_destroy(&a->lock);
        talloc_free(a);
        return NULL;
    }
    a->worker_running = true;
    return a;
}

void sub_ahead_destroy(struct sub_ahead *a)
{
    if (!a)
        return;
    if (a->worker_running) {
        mp_mutex_lock(&a->lock);
        a->terminate = true;
        mp_cond_signal(&a->wakeup);
        mp_cond_broadcast(&a->ring_added);
        mp_mutex_unlock(&a->lock);
        mp_thread_join(a->thread);
    }
    // Worker joined: single-threaded from here (no lock needed).
    struct ahead_dbg_snap snap = ahead_dbg_snapshot(a);
    snap.verbose_due = true;
    ahead_dbg_emit(a, &snap);
    queue_flush(a);
    ahead_clear(a);
    int nretire = 0;
    struct ahead_payload **retired = retire_steal(a, &nretire);
    retire_flush(retired, nretire);
    if (a->worker_sd) {
        a->worker_sd->driver->uninit(a->worker_sd);
        talloc_free(a->worker_sd);
    }
    mp_cond_destroy(&a->ring_added);
    mp_cond_destroy(&a->wakeup);
    mp_mutex_destroy(&a->lock);
    talloc_free(a);
}

void sub_ahead_enqueue(struct sub_ahead *a, struct demux_packet *pkt)
{
    if (!a || !pkt)
        return;
    struct demux_packet *copy = demux_copy_packet(NULL, pkt);
    if (!copy)
        return;
    mp_mutex_lock(&a->lock);
    MP_TARRAY_APPEND(a, a->queue, a->num_queue, copy);
    if (pkt->pts != MP_NOPTS_VALUE) {
        // Ring entries at/after this event's start were rendered without it:
        // drop them (they re-render once the worker decoded the packet), and
        // remember the floor so an in-flight render is discarded at store.
        a->newpkt_min_pts = MPMIN(a->newpkt_min_pts, pkt->pts);
        for (int n = 0; n < a->ring_len; n++) {
            struct sub_ahead_entry *e = &a->ring[n];
            if (e->valid &&
                ahead_pts_to_subtitle(a, e->video_pts) >= pkt->pts - 1e-9)
            {
                payload_retire(a, e->pl);   // freed outside `lock` below
                *e = (struct sub_ahead_entry){0};
            }
        }
    }
    mp_cond_signal(&a->wakeup);
    int nretire = 0;
    struct ahead_payload **retired = retire_steal(a, &nretire);
    mp_mutex_unlock(&a->lock);
    retire_flush(retired, nretire);
}

void sub_ahead_flush(struct sub_ahead *a)
{
    if (!a)
        return;
    mp_mutex_lock(&a->lock);
    queue_flush(a);
    ahead_clear(a);              // also drops the stale-serve source (the ring)
    a->gen++;
    a->reset_pending = true;     // flush the worker's own track (on its thread)
    a->vo_pts = MP_NOPTS_VALUE;
    a->hint_pts = MP_NOPTS_VALUE;  // pre-seek hint must not steer the pre-warm
    a->newpkt_min_pts = INFINITY;
    mp_cond_signal(&a->wakeup);
    mp_cond_broadcast(&a->ring_added);  // don't leave a fetch waiting on stale gen
    int nretire = 0;
    struct ahead_payload **retired = retire_steal(a, &nretire);
    mp_mutex_unlock(&a->lock);
    retire_flush(retired, nretire);
}

void sub_ahead_set_timing(struct sub_ahead *a, double sub_speed, float delay,
                          int play_dir, double video_fps)
{
    if (!a)
        return;
    mp_mutex_lock(&a->lock);
    bool changed = a->sub_speed != sub_speed || a->delay != delay ||
                   a->play_dir != play_dir || a->video_fps != video_fps;
    a->sub_speed = sub_speed;
    a->delay = delay;
    a->play_dir = play_dir;
    a->video_fps = video_fps;
    if (changed) {
        ahead_clear(a);          // stale-delay/speed entries must not be served
        a->gen++;
    }
    mp_cond_signal(&a->wakeup);
    int nretire = 0;
    struct ahead_payload **retired = retire_steal(a, &nretire);
    mp_mutex_unlock(&a->lock);
    retire_flush(retired, nretire);
}

void sub_ahead_set_video_params(struct sub_ahead *a,
                                const struct mp_image_params *params)
{
    if (!a || !params)
        return;
    mp_mutex_lock(&a->lock);
    // This is sent every frame; only invalidate when the params actually change
    // (otherwise the ring is wiped every frame and the worker never gets ahead).
    if (!a->params_set || !mp_image_params_equal(&a->pending_params, params)) {
        a->pending_params = *params;
        a->params_pending = true;
        a->params_set = true;
        ahead_clear(a);          // colorspace may change -> drop stale entries
        a->gen++;
        mp_cond_signal(&a->wakeup);
    }
    int nretire = 0;
    struct ahead_payload **retired = retire_steal(a, &nretire);
    mp_mutex_unlock(&a->lock);
    retire_flush(retired, nretire);
}

void sub_ahead_hint_pts(struct sub_ahead *a, double raw_video_pts)
{
    if (!a || raw_video_pts == MP_NOPTS_VALUE)
        return;
    mp_mutex_lock(&a->lock);
    a->hint_pts = raw_video_pts;
    mp_cond_signal(&a->wakeup);  // kick the pre-warm
    mp_mutex_unlock(&a->lock);
}

void sub_ahead_note_inline(struct sub_ahead *a)
{
    if (!a)
        return;
    mp_mutex_lock(&a->lock);
    long v = ++a->dbg_inline;
    mp_mutex_unlock(&a->lock);
    MP_STATS(a, "value %ld ra-inline", v);   // WP-H6: no stats I/O under lock
}

void sub_ahead_inline_begin(struct sub_ahead *a)
{
    if (!a)
        return;
    mp_mutex_lock(&a->lock);
    a->inline_active++;
    mp_mutex_unlock(&a->lock);
}

void sub_ahead_inline_end(struct sub_ahead *a)
{
    if (!a)
        return;
    mp_mutex_lock(&a->lock);
    a->inline_active--;
    mp_cond_signal(&a->wakeup);   // let the worker resume
    mp_mutex_unlock(&a->lock);
}

struct sub_bitmaps *sub_ahead_get_bitmaps(struct sub_ahead *a,
                                          struct mp_osd_res dim, int format,
                                          double raw_video_pts, bool *handled)
{
    *handled = false;
    if (!a)
        return NULL;
    // WP-H6 (item 2): phase-attribute the fetch. lock_ns is the time to
    // acquire `lock` (round 3's uncounted 256.7 ms class: the fetch blocked
    // before doing ANY counted work); wait_ns is the intentional bounded
    // miss-wait (already counted as ra-wait). Anything else above
    // RA_FETCH_STALL_NS bumps ra-fetch-stall (gated ==0) with a breakdown.
    int64_t t_enter = mp_time_ns();
    mp_mutex_lock(&a->lock);
    int64_t lock_ns = mp_time_ns() - t_enter;
    int64_t wait_ns = 0;
    if (a->play_dir < 0) {
        // Reverse playback: the worker only walks forward. Degraded mode --
        // the caller falls back to the inline render (counted via ra-inline).
        mp_mutex_unlock(&a->lock);
        return NULL;
    }
    *handled = true;
    if (raw_video_pts == MP_NOPTS_VALUE) {
        // No real frame pts (e.g. redraw before the first frame): "no
        // subtitles" is the right answer; don't wait and don't count a miss.
        mp_mutex_unlock(&a->lock);
        return NULL;
    }
    a->vo_pts = raw_video_pts;
    if (a->cur_format != format || !osd_res_equals(a->cur_dim, dim)) {
        a->cur_dim = dim;
        a->cur_format = format;
        a->gen++;
        ahead_clear(a);   // resize/format change; also drops the stale source
    }
    int idx = ahead_find(a, raw_video_pts, dim, format);
    if (idx < 0) {
        // Miss: bounded wait for the worker to deliver this frame instead of
        // rendering inline on this (the VO) thread. Default deadline is one
        // frame interval -- by then the frame is late anyway, and the stale
        // fallback below is the better failure mode.
        double wait_s = a->miss_wait_ms < 0 ? ahead_interval(a)
                                            : a->miss_wait_ms / 1000.0;
        if (wait_s > 0) {
            mp_cond_signal(&a->wakeup);  // the worker targets i=0 first
            int64_t wait_t0 = mp_time_ns();
            int64_t until = wait_t0 + (int64_t)(wait_s * 1e9);
            while (!a->terminate) {
                idx = ahead_find(a, raw_video_pts, a->cur_dim, a->cur_format);
                if (idx >= 0)
                    break;
                if (mp_cond_timedwait_until(&a->ring_added, &a->lock, until))
                    break;       // deadline passed
            }
            wait_ns = mp_time_ns() - wait_t0;
            if (idx >= 0)
                a->dbg_wait++;
        }
    }
    int src = idx;
    if (idx >= 0) {
        if (a->ring[idx].pl) {
            a->dbg_served++;
        } else {
            a->dbg_empty++;  // rendered, no subtitles at this pts (a real answer)
        }
    } else {
        // Still missing: serve the nearest EARLIER rendered frame (the viewer
        // keeps seeing the previous subtitle content for a frame -- correct
        // enough, and structurally harmless vs. a ~90ms inline render). After
        // a flush/reset the ring is empty and this deliberately serves EMPTY:
        // a subtitle-less frame beats pre-seek content at a post-seek position.
        a->dbg_miss++;
        src = ahead_find_stale(a, raw_video_pts, dim, format);
        if (src >= 0)
            a->dbg_stale++;
    }
    struct sub_bitmaps *res = NULL;
    if (src >= 0 && a->ring[src].pl) {
        // Serve by reference: pin the entry's payload instead of deep-copying
        // parts + blobs (the copy was the dominant VO-thread cost on dense
        // OUTLINES frames). The pin keeps the payload alive even if the
        // worker evicts the entry while the VO still holds the serve.
        res = payload_serve(a->ring[src].pl);
    }
    if (src >= 0) {
        // Whatever the VO draws this frame it also resolves against the GPU
        // glyph cache, so the idle pre-fill must not reprocess this entry.
        a->ring[src].prefilled = true;
    }
    if (res) {
        // Unified monotonic change_id: report a fresh id only when the content
        // differs from what we last served, else 0 (osd.c treats 0 as "no
        // change, no re-upload"). A stale re-serve of the last served frame
        // yields 0 naturally (same content id).
        uint64_t cid = a->ring[src].content_id;
        if (!a->have_last_served || cid != a->last_served_content) {
            a->last_served_content = cid;
            a->have_last_served = true;
            res->change_id = ++a->served_change_id;
        } else {
            res->change_id = 0;
        }
    }
    // WP-H6 (item 2): everything the fetch did MINUS the intentional bounded
    // wait; a stall here was invisible to every round-3 counter.
    int64_t stall_ns = (mp_time_ns() - t_enter) - wait_ns;
    bool stalled = stall_ns > RA_FETCH_STALL_NS;
    if (stalled)
        a->dbg_fetch_stall++;
    struct ahead_dbg_snap snap = ahead_dbg_snapshot(a);
    int nretire = 0;
    struct ahead_payload **retired = retire_steal(a, &nretire);
    mp_mutex_unlock(&a->lock);
    mp_cond_signal(&a->wakeup);  // nudge the worker to refill ahead of us
    // Log/stats writes and payload frees happen OFF the lock (item 2): a
    // blocking stats write or a big talloc free here can no longer stall a
    // concurrent fetch or the worker.
    if (stalled) {
        MP_WARN(a, "[render-ahead] fetch stalled %.1f ms outside the bounded "
                "wait (lock %.1f ms, wait %.1f ms)\n", stall_ns / 1e6,
                lock_ns / 1e6, wait_ns / 1e6);
    }
    ahead_dbg_emit(a, &snap);
    retire_flush(retired, nretire);
    return res;
}

struct sub_bitmaps *sub_ahead_peek_prefill(struct sub_ahead *a, double *out_pts)
{
    if (!a)
        return NULL;
    struct sub_bitmaps *res = NULL;
    mp_mutex_lock(&a->lock);
    // Lowest-pts first: after a seek the pre-warm target (i=0) is the first
    // frame the VO will fetch, so it must also be the first one pre-filled.
    int best = -1;
    for (int n = 0; n < a->ring_len; n++) {
        struct sub_ahead_entry *e = &a->ring[n];
        if (e->valid && e->gen == a->gen && !e->prefilled && e->pl &&
            (best < 0 || e->video_pts < a->ring[best].video_pts))
            best = n;
    }
    if (best >= 0) {
        res = payload_serve(a->ring[best].pl);
        if (res) {
            *out_pts = a->ring[best].video_pts;
        } else {
            // No renderable parts: nothing to pre-fill, never return it again.
            a->ring[best].prefilled = true;
        }
    }
    mp_mutex_unlock(&a->lock);
    return res;
}

void sub_ahead_prefill_done(struct sub_ahead *a, double video_pts)
{
    if (!a)
        return;
    mp_mutex_lock(&a->lock);
    for (int n = 0; n < a->ring_len; n++) {
        struct sub_ahead_entry *e = &a->ring[n];
        if (e->valid && e->video_pts == video_pts)
            e->prefilled = true;
    }
    mp_mutex_unlock(&a->lock);
}
