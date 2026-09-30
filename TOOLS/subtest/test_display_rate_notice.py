#!/usr/bin/env python3
"""Check the real refresh-notice scheduling code with a fake clock and VO.

Run with --compiler zig cc (or another C compiler). This compiles only the
affected callbacks, not the player. It does not test OSD rendering or a TV.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]

PREAMBLE = r"""
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
struct mp_display_rate { double fps; bool variable, apply; };
struct vo_opts { int display_rate_match; double display_rate_match_delay; };
struct MPOpts { struct vo_opts *vo; bool pause, video_osd; int osd_level; };
struct vo { struct vo_opts *opts; };
struct MPContext {
    struct MPOpts *opts;
    struct vo *video_out;
    void *vo_chain;
    double display_rate_resume_time;
    struct mp_display_rate *display_rate_pending;
    bool display_rate_initialized, osd_force_update, paused, paused_for_cache;
    int stop_play, video_status;
};
enum { VO_TRUE = 1, VOCTRL_MATCH_DISPLAY_RATE, STATUS_READY };
#define MP_VERBOSE(...) ((void)0)
#define talloc_free free
static void *talloc_memdup(void *parent, const void *src, size_t n) {
    void *p = malloc(n); assert(p); return memcpy(p, src, n);
}
static double now, wake_at, earliest_apply;
static int applies, result;
static bool notice;
static struct MPContext *active;
static double mp_time_sec(void) { return now; }
static void mp_set_timeout(struct MPContext *m, double delay) {
    wake_at = now + delay;
}
static void update_internal_pause_state(struct MPContext *m) {
    m->paused = m->opts->pause || m->paused_for_cache || m->display_rate_resume_time;
}
static void update_osd_msg(struct MPContext *m) {
    notice = m->display_rate_resume_time && m->opts->video_osd && m->opts->osd_level >= 1;
}
static int vo_control(struct vo *vo, int request, struct mp_display_rate *rate) {
    if (!rate->apply) return VO_TRUE;
    assert(active->paused && now >= earliest_apply);
    assert(rate->fps == 24);
    applies++;
    now += 0.25; // time spent in the synchronous Windows mode change
    return result;
}
"""

TEST = r"""
static void run(bool visible, bool manual, bool cancel, int apply_result, double delay) {
    struct vo_opts vo_opts = {1, delay};
    struct MPOpts opts = {&vo_opts, false, visible, 1};
    struct vo vo = {&vo_opts};
    struct MPContext m = {.opts = &opts, .video_out = &vo, .vo_chain = &vo,
                          .display_rate_initialized = true, .video_status = STATUS_READY};
    active = &m;
    now = 100;
    earliest_apply = now + (visible ? 0.5 : 0);
    applies = 0; result = apply_result; notice = false;
    start_notice(&m);
    assert(m.paused && m.display_rate_pending && applies == 0);
    assert(notice == visible && wake_at == earliest_apply);
    if (visible) {
        now = earliest_apply - 0.01;
        handle_display_rate_pause(&m);
        assert(applies == 0 && m.paused && wake_at == earliest_apply);
    }
    if (cancel) {
        opts.vo->display_rate_match = 0;
        handle_display_rate_pause(&m);
        assert(applies == 0 && !m.paused && !m.display_rate_pending);
        assert(!m.display_rate_initialized && m.osd_force_update);
        return;
    }
    opts.pause = manual; // user pauses during the notice
    now = earliest_apply;
    handle_display_rate_pause(&m);
    assert(applies == 1 && !m.display_rate_pending);
    if (result == VO_TRUE && delay > 0) {
        assert(m.paused && wake_at == earliest_apply + 0.25 + delay);
        now = wake_at - 0.01;
        handle_display_rate_pause(&m);
        assert(m.paused && applies == 1);
        now = wake_at;
        handle_display_rate_pause(&m);
    }
    update_osd_msg(&m);
    assert(!m.display_rate_resume_time && !notice && m.osd_force_update);
    assert(m.paused == manual && opts.pause == manual && applies == 1);
}
int main(void) {
    run(true, false, false, VO_TRUE, 3);
    run(true, true, false, VO_TRUE, 3);
    run(true, false, true, VO_TRUE, 3);
    run(true, false, false, 0, 3);
    run(true, false, false, VO_TRUE, 0);
    run(false, false, false, VO_TRUE, 3);
    puts("Refresh notice scheduling: 6 cases passed");
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", nargs="+", required=True)
    args = parser.parse_args()
    video = (ROOT / "player/video.c").read_text(encoding="utf-8")
    loop = (ROOT / "player/playloop.c").read_text(encoding="utf-8")
    start = re.search(r"        if \(vo_control\(vo, VOCTRL_MATCH_DISPLAY_RATE, &rate\)"
                      r" == VO_TRUE\).*?^        \}", video, re.M | re.S)
    tick = re.search(r"^static void handle_display_rate_pause\([^\n]*\)\n\{.*?^\}",
                     loop, re.M | re.S)
    if not start or not tick:
        raise RuntimeError("Cannot locate refresh-notice scheduling code")
    wrapper = """
static void start_notice(struct MPContext *mpctx) {
    struct MPOpts *opts = mpctx->opts;
    struct vo *vo = mpctx->video_out;
    struct mp_display_rate rate = {.fps = 24};
""" + start.group() + "\n}\n"
    with tempfile.TemporaryDirectory(prefix="ajn-refresh-notice-") as temp:
        cfile = Path(temp) / "notice.c"
        exe = Path(temp) / "notice-test.exe"
        cfile.write_text(PREAMBLE + wrapper + tick.group() + TEST, encoding="utf-8")
        subprocess.run([*args.compiler, "-std=c11", "-Werror=implicit-function-declaration",
                        str(cfile), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
