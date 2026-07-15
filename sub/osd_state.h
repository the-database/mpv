#ifndef MP_OSD_STATE_H_
#define MP_OSD_STATE_H_

#include <stdatomic.h>

#include "osd.h"
#include "osdep/threads.h"

enum mp_osdtype {
    OSDTYPE_SUB,
    OSDTYPE_SUB2, // IDs must be numerically successive

    OSDTYPE_OSD,

    OSDTYPE_EXTERNAL,
    OSDTYPE_EXTERNAL2,

    OSDTYPE_COUNT
};

struct ass_state {
    struct mp_log *log;
    struct ass_track *track;
    struct ass_renderer *render;
    struct ass_library *library;
    int res_x, res_y;
    bool changed;
    struct mp_osd_res vo_res; // last known value
};

struct osd_object {
    int type; // OSDTYPE_*
    bool is_sub;

    // OSDTYPE_OSD
    bool osd_changed;
    char *text;
    struct osd_progbar_state progbar_state;

    // OSDTYPE_SUB/OSDTYPE_SUB2
    struct dec_sub *sub;

    // OSDTYPE_EXTERNAL
    struct osd_external **externals;
    int num_externals;

    // WP-H7 (defect 2): async render state for OSDTYPE_EXTERNAL. The heavy
    // libass work for script overlays (the stats page re-shapes its whole
    // text on every ~1 Hz update; opening it is a 50+ ms layout) used to run
    // inside osd_render ON THE VO THREAD; at 8K that tips frames over the
    // budget. A dedicated worker (osd_libass.c) now owns ALL libass work for
    // this object; the VO is served the last completed snapshot (at most one
    // update stale -- invisible for OSD) under osd->lock only.
    //   osd->lock domain:
    int64_t ext_req_gen;              // bumped on every content change
    int64_t ext_done_gen;             // generation the snapshot was built from
    struct sub_bitmaps *ext_done;     // last completed packed snapshot
    struct mp_osd_res ext_done_res;   // render res it was built at
    int ext_done_format;
    struct mp_osd_res ext_want_res;   // latest res/format a consumer asked for
    int ext_want_format;
    struct sub_bitmap_copy_cache *serve_cache; // per-serve copy reuse

    // OSDTYPE_EXTERNAL2
    struct sub_bitmaps *external2;

    // VO cache state
    int vo_change_id;
    struct mp_osd_res vo_res;   // true output geometry, never capped
    // Rasterization resolution: == vo_res unless --osd-render-res-cap
    // shrinks a non-subtitle object's render (osd.c render_object). Never
    // exposed as geometry -- osd-dimensions/scripts/mouse read vo_res.
    struct mp_osd_res render_res;
    bool vo_had_output;

    // Internally used by osd_libass.c
    bool changed;
    struct ass_state ass;
    struct mp_sub_packer *sub_packer;
    struct sub_bitmap_copy_cache *copy_cache;
    struct ass_image **ass_imgs;
};

struct osd_external {
    struct osd_external_ass ov;
    struct ass_state ass;
    bool dirty;     // WP-H7: ov.data changed; the worker re-parses the track
                    // (ext_lock domain)
};

struct osd_state {
    mp_mutex lock;

    // WP-H7 (defect 2): protects the OSDTYPE_EXTERNAL entries array, the
    // entries' ov fields + ass states, and that object's packer/copy caches.
    // LOCK ORDER: ext_lock BEFORE osd->lock (the worker holds ext_lock across
    // a render and briefly takes osd->lock inside for opts/publication;
    // nothing may acquire ext_lock while holding osd->lock).
    mp_mutex ext_lock;
    mp_cond ext_wakeup;               // paired with osd->lock (request flags)
    mp_thread ext_thread;
    bool ext_thread_valid;
    bool ext_exit;
    // Completion callback (e.g. wake the player core so a fresh overlay is
    // presented promptly even while paused); set once at init.
    void (*ext_wakeup_cb)(void *ctx);
    void *ext_wakeup_ctx;

    struct osd_object *objs[MAX_OSD_PARTS];

    bool render_subs_in_filter;
    _Atomic double force_video_pts;

    // Bumped whenever a sub track is attached/detached/switched (osd_set_sub).
    // VOs use it to invalidate cached subtitle overlay snapshots across track
    // changes (see osd_sub_track_epoch()).
    _Atomic uint64_t sub_track_epoch;

    bool want_redraw;
    bool want_redraw_notification;

    struct m_config_cache *opts_cache;
    struct mp_osd_render_opts *opts;
    struct mpv_global *global;
    struct mp_log *log;
    struct stats_ctx *stats;
    // WP-H7 (defect 2): dedicated stats ctx for the external-OSD worker; the
    // worker must not share hot mp_log objects with the VO thread (the
    // level-cache reload in mp_msg_level is an unlocked read).
    struct stats_ctx *stats_ext;

    struct mp_draw_sub_cache *draw_cache;
};

// defined in osd_libass.c
// res is the rasterization resolution (obj->render_res; may be smaller than
// the true geometry in obj->vo_res under --osd-render-res-cap).
struct sub_bitmaps *osd_object_get_bitmaps(struct osd_state *osd,
                                           struct osd_object *obj,
                                           struct mp_osd_res res, int format);
// WP-H7 (defect 2): non-blocking OSDTYPE_EXTERNAL serve -- returns a copy of
// the worker's last completed snapshot (NULL if none at this geometry yet)
// and kicks the worker when the request generation or geometry moved on.
// Called under osd->lock.
struct sub_bitmaps *osd_external_render_async(struct osd_state *osd,
                                              struct osd_object *obj,
                                              struct mp_osd_res res,
                                              int format);
void osd_destroy_backend(struct osd_state *osd);

#endif
