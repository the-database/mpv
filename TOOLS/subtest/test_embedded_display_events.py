#!/usr/bin/env python3
"""Exercise the embedded display callback with real hidden Windows windows.

Run with --compiler PATH/TO/gcc.exe. Does not change display modes or broadcast
messages to other applications. Requires Windows and a native C compiler.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
TEST = r'''
static int received;
static WPARAM depth;
static LPARAM dimensions;
static LRESULT CALLBACK child_proc(HWND w, UINT m, WPARAM a, LPARAM b) {
    if (m == WM_DISPLAYCHANGE) { received++; depth = a; dimensions = b; }
    return DefWindowProcW(w, m, a, b);
}
static BOOL CALLBACK contains(HWND w, LPARAM param) {
    HWND *list = (HWND *)param;
    if (w == list[0]) list[1] = w;
    return TRUE;
}
static bool top_level(HWND w) {
    HWND list[] = {w, NULL};
    EnumThreadWindows(GetCurrentThreadId(), contains, (LPARAM)list);
    return list[1] == w;
}
int main(void) {
    HINSTANCE instance = GetModuleHandleW(NULL);
    WNDCLASSW normal = {.hInstance = instance, .lpfnWndProc = child_proc,
                       .lpszClassName = L"ajn-display-test"};
    WNDCLASSW events = {.hInstance = instance, .lpfnWndProc = display_event_proc,
                       .lpszClassName = L"ajn-display-events-test"};
    assert(RegisterClassW(&normal) && RegisterClassW(&events));
    HWND parent = CreateWindowW(normal.lpszClassName, L"", WS_POPUP,
        0, 0, 1, 1, NULL, NULL, instance, NULL);
    HWND child = CreateWindowW(normal.lpszClassName, L"", WS_CHILD,
        0, 0, 1, 1, parent, NULL, instance, NULL);
    HWND receiver = CreateWindowExW(WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
        events.lpszClassName, L"", WS_POPUP, 0, 0, 0, 0,
        NULL, NULL, instance, child);
    assert(parent && child && receiver);
    assert(!top_level(child) && top_level(receiver) && !IsWindowVisible(receiver));
    // Emulate the broadcast's delivery to our top-level receiver only.
    LPARAM size = MAKELPARAM(7680, 4320);
    SendMessageW(receiver, WM_DISPLAYCHANGE, 32, size);
    MSG message;
    while (PeekMessageW(&message, NULL, 0, 0, PM_REMOVE))
        DispatchMessageW(&message);
    assert(received == 1 && depth == 32 && dimensions == size);
    DestroyWindow(receiver);
    DestroyWindow(child);
    DestroyWindow(parent);
    puts("PASS hidden top-level display notification reaches embedded child");
}
'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', required=True)
    args = parser.parse_args()
    source = (ROOT / 'video/out/w32_common.c').read_text(encoding='utf-8')
    callback = re.search(r'static LRESULT CALLBACK display_event_proc\(.*?^}',
                         source, re.S | re.M)
    assert callback
    with tempfile.TemporaryDirectory(prefix='ajn-display-events-') as directory:
        c = Path(directory) / 'test.c'
        exe = Path(directory) / 'test.exe'
        c.write_text('#include <windows.h>\n#include <assert.h>\n'
                     '#include <stdbool.h>\n#include <stdio.h>\n'
                     + callback.group() + '\n' + TEST, encoding='utf-8')
        subprocess.run([args.compiler, str(c), '-o', str(exe), '-luser32'], check=True)
        subprocess.run([str(exe)], check=True, timeout=10)

if __name__ == '__main__':
    main()
