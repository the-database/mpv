#!/usr/bin/env python3
"""Exercise the real Windows refresh callbacks with fake display settings.

Run with --compiler cc (or another C compiler). No display changes are made.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]

PREAMBLE = r"""
#include <assert.h>
#include <stdio.h>
#include <wchar.h>
#include "video/out/display_rate.h"
typedef unsigned DWORD;
typedef long LONG;
typedef int HMONITOR;
typedef struct {
    unsigned dmSize, dmPelsWidth, dmPelsHeight, dmBitsPerPel;
    unsigned dmDisplayOrientation, dmDisplayFlags, dmDisplayFrequency, dmFields;
} DEVMODEW;
typedef struct { unsigned cbSize; wchar_t szDevice[32]; } MONITORINFOEXW, MONITORINFO;
struct vo_opts { int display_rate_match; };
struct vo_w32_state {
    struct vo_opts *opts;
    int window;
    bool rate_owned, rate_blocked, rate_test_variable;
    DWORD rate_test_hz;
    wchar_t rate_device[32];
    DEVMODEW rate_original, rate_applied;
};
enum { VO_ERROR = -1, VO_FALSE, VO_TRUE };
enum { DM_INTERLACED = 1, DM_DISPLAYFREQUENCY, CDS_TEST, DISP_CHANGE_SUCCESSFUL = 0 };
#define ENUM_CURRENT_SETTINGS ((DWORD)-1)
#define IsIconic(window) false
#define MonitorFromWindow(window, flags) 0
#define MP_WARN(...) ((void)0)
#define MP_INFO(...) ((void)0)
static DEVMODEW desktop = {.dmPelsWidth = 7680, .dmPelsHeight = 4320,
                          .dmBitsPerPel = 32, .dmDisplayFrequency = 60};
static int changes;
static bool GetMonitorInfoW(HMONITOR monitor, MONITORINFO *info) {
    wcscpy(info->szDevice, L"test-display");
    return true;
}
static bool EnumDisplaySettingsW(const wchar_t *device, DWORD index, DEVMODEW *mode) {
    const unsigned modes[] = {23, 24, 47, 48, 60};
    if (index != ENUM_CURRENT_SETTINGS && index >= sizeof(modes) / sizeof(modes[0]))
        return false;
    *mode = desktop;
    if (index != ENUM_CURRENT_SETTINGS)
        mode->dmDisplayFrequency = modes[index];
    return true;
}
static LONG ChangeDisplaySettingsExW(const wchar_t *device, DEVMODEW *mode,
                                    void *window, DWORD flags, void *param) {
    assert(mode->dmFields == DM_DISPLAYFREQUENCY);
    if (flags != CDS_TEST) {
        desktop.dmDisplayFrequency = mode->dmDisplayFrequency;
        changes++;
    }
    return DISP_CHANGE_SUCCESSFUL;
}
static void force_update_display_info(struct vo_w32_state *w32) {}
"""

TEST = r"""
int main(void) {
    struct vo_opts opts = {.display_rate_match = 1};
    struct vo_w32_state w32 = {.opts = &opts};
    struct mp_display_rate known = {.fps = 24000.0 / 1001, .apply = true};
    struct mp_display_rate unknown = {.variable = true};

    // An unknown first file leaves the desktop alone.
    assert(match_display_rate(&w32, &unknown) == VO_FALSE);
    assert(desktop.dmDisplayFrequency == 60 && changes == 0 && !w32.rate_owned);

    // A known episode matches 23.976; an unknown next episode restores 60.
    assert(match_display_rate(&w32, &known) == VO_TRUE);
    assert(desktop.dmDisplayFrequency == 23 && w32.rate_owned && changes == 1);
    assert(match_display_rate(&w32, &unknown) == VO_FALSE);
    assert(desktop.dmDisplayFrequency == 60 && !w32.rate_owned && changes == 2);
    assert(match_display_rate(&w32, &unknown) == VO_FALSE);
    assert(changes == 2);

    // Do not undo a user's manual change between playlist entries.
    assert(match_display_rate(&w32, &known) == VO_TRUE);
    desktop.dmDisplayFrequency = 48;
    assert(match_display_rate(&w32, &unknown) == VO_FALSE);
    assert(desktop.dmDisplayFrequency == 48 && changes == 3);
    assert(!w32.rate_owned && w32.rate_blocked);
    puts("PASS unknown first file; playlist restore; manual display change");
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", nargs="+", required=True)
    args = parser.parse_args()
    source = (ROOT / "video/out/w32_common.c").read_text(encoding="utf-8")
    callbacks = []
    for name in ("same_display_mode", "restore_display_rate", "match_display_rate"):
        match = re.search(r"^static \w+ " + name + r"\([^\n]*\)\n\{.*?^\}",
                          source, re.M | re.S)
        if not match:
            raise RuntimeError(f"Cannot locate {name}")
        callbacks.append(match.group())
    with tempfile.TemporaryDirectory(prefix="ajn-refresh-restore-") as temp:
        cfile = Path(temp) / "restore.c"
        exe = Path(temp) / "restore-test.exe"
        cfile.write_text(PREAMBLE + "\n".join(callbacks) + TEST, encoding="utf-8")
        subprocess.run([*args.compiler, "-std=c11", "-Werror=implicit-function-declaration",
                        "-I", str(ROOT), str(cfile), "-lm", "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
