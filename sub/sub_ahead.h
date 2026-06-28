#ifndef MP_SUB_AHEAD_H
#define MP_SUB_AHEAD_H

#include <stdbool.h>

#include "osd.h"

struct dec_sub;
struct sd;
struct demux_packet;
struct mp_image_params;

// A subtitle render-ahead worker. It owns a second, fully independent sd_ass
// instance (its own ASS_Library/Renderer/Track) and renders upcoming frames on
// a background thread into a PTS-keyed ring, fed by a packet queue from the
// decode path. The VO fast path serves a pre-rendered frame without taking the
// decoder lock. See sub_ahead.c for the locking model and the design notes on
// why this avoids the two failure modes of the original (reverted) attempt.
struct sub_ahead;

// Create the worker. Takes ownership of worker_sd (an sd_ass sd built with a
// capped thread count). depth = lookahead frames (>0). Returns NULL on failure
// (caller must still free worker_sd in that case).
struct sub_ahead *sub_ahead_create(struct dec_sub *sub, struct sd *worker_sd,
                                   int depth, int order);

// Stop the worker thread and free everything, including worker_sd.
void sub_ahead_destroy(struct sub_ahead *a);

// Enqueue a sub packet for the worker to decode into its own track. Called from
// the decode path (under the decoder lock); copies the packet, brief queue lock,
// never blocks on a render.
void sub_ahead_enqueue(struct sub_ahead *a, struct demux_packet *pkt);

// Drop the queue + ring and reset the worker track (seek / clear-on-seek).
void sub_ahead_flush(struct sub_ahead *a);

// Snapshot the pts mapping the worker uses (raw video pts -> subtitle pts).
// Bumps the gen + flushes (stale-delay entries must not be served).
void sub_ahead_set_timing(struct sub_ahead *a, double sub_speed, float delay,
                          int play_dir, double video_fps);

// Forward video params to the worker sd (for color mangling / storage size).
void sub_ahead_set_video_params(struct sub_ahead *a,
                                const struct mp_image_params *params);

// VO fast path. Returns a fresh owned sub_bitmaps on a ring hit, or NULL to tell
// the caller to fall back to the inline render. raw_video_pts is the unconverted
// video pts (the ring is keyed on it).
struct sub_bitmaps *sub_ahead_get_bitmaps(struct sub_ahead *a,
                                          struct mp_osd_res dim, int format,
                                          double raw_video_pts);

// Bracket the VO's inline fallback render: while inline rendering is active the
// worker pauses, so the two heavy parallel renders never fight for the cores
// (which turned one miss into a multi-frame stall).
void sub_ahead_inline_begin(struct sub_ahead *a);
void sub_ahead_inline_end(struct sub_ahead *a);

#endif
