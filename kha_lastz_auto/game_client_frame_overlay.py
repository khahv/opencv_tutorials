"""
game_client_frame_overlay.py
------------------------------
Draws a thin border around the game window **client area** (same region as capture) on Windows.

Used when the bot is in PC mode and Is Running is ON. The border is topmost and
mouse-transparent (WM_NCHITTEST → HTTRANSPARENT) so it does not steal clicks from the game.

Visibility / minimize detection follows Microsoft Win32 documentation:

- ``IsIconic`` — https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-isiconic
- ``GetWindowPlacement`` (``showCmd`` / ``SW_SHOWMINIMIZED``) —
  https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getwindowplacement
- ``GetWindowLong`` + ``GWL_STYLE`` + ``WS_MINIMIZE`` —
  https://learn.microsoft.com/en-us/windows/win32/winmsg/window-styles
- ``GetAncestor(..., GA_ROOT)`` — minimize checks use the **root** top-level window; capture may
  hold a child HWND while the shell minimizes the root.
- ``DwmGetWindowAttribute`` with ``DWMWA_CLOAKED`` (optional) — windows not on the current
  virtual desktop can be cloaked —
  https://learn.microsoft.com/en-us/windows/win32/api/dwmapi/ne-dwmapi-dwmwindowattribute

Note: ``IsWindowVisible`` alone is not sufficient — minimized windows can still report visible.

**Interactive visibility (Case 1 vs 2):** After minimize checks, the border is shown only when
``GetForegroundWindow()`` is either (a) a window whose ``GA_ROOT`` is the game’s root HWND
(focus inside LastZ, including child HWNDs), or (b) any HWND owned by this Python process
(bot Tk, dialogs, preview). If focus is in another application (e.g. Cursor), the border is
hidden even though LastZ may still appear behind other windows.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from ctypes import wintypes
from typing import Any, List, Optional

log = logging.getLogger("kha_lastz")

# https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getwindowplacement
_SW_SHOWMINIMIZED = 2
# https://learn.microsoft.com/en-us/windows/win32/api/dwmapi/ne-dwmapi-dwmwindowattribute
_DWMWA_CLOAKED = 14


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class _WINDOWPLACEMENT(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_uint),
        ("flags", ctypes.c_uint),
        ("showCmd", ctypes.c_uint),
        ("ptMinPosition", _POINT),
        ("ptMaxPosition", _POINT),
        ("rcNormalPosition", _RECT),
    ]


def _root_top_level_hwnd(hwnd: int) -> int:
    """Return the root (top-level) window for minimize / placement checks."""
    import win32con
    import win32gui

    try:
        r = win32gui.GetAncestor(int(hwnd), win32con.GA_ROOT)
        if r:
            return int(r)
    except Exception:
        pass
    return int(hwnd)


def _get_window_placement_showcmd(hwnd: int) -> Optional[int]:
    """Return ``WINDOWPLACEMENT.showCmd`` via ``GetWindowPlacement``, or None on failure."""
    user32 = ctypes.windll.user32
    wp = _WINDOWPLACEMENT()
    wp.length = ctypes.sizeof(_WINDOWPLACEMENT)
    if not user32.GetWindowPlacement(wintypes.HWND(hwnd), ctypes.byref(wp)):
        return None
    return int(wp.showCmd)


def _is_dwm_cloaked(hwnd: int) -> bool:
    """True if DWM reports the window as cloaked (e.g. other virtual desktop)."""
    try:
        dwmapi = ctypes.windll.dwmapi
        cloaked = ctypes.c_int(0)
        hr = dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(hwnd),
            ctypes.c_uint(_DWMWA_CLOAKED),
            ctypes.byref(cloaked),
            ctypes.sizeof(ctypes.c_int),
        )
        if hr == 0 and cloaked.value != 0:
            return True
    except Exception:
        pass
    return False


def _is_minimized_or_non_interactive_top_level(root_hwnd: int) -> bool:
    """Best-effort minimized state using documented Win32 APIs on the root HWND."""
    import win32con
    import win32gui

    try:
        if win32gui.IsIconic(root_hwnd):
            return True
    except Exception:
        pass
    try:
        style = win32gui.GetWindowLong(root_hwnd, win32con.GWL_STYLE)
        if style & win32con.WS_MINIMIZE:
            return True
    except Exception:
        pass
    sc = _get_window_placement_showcmd(root_hwnd)
    if sc is not None and sc == _SW_SHOWMINIMIZED:
        return True
    return False


def _foreground_allows_border(root_lastz: int) -> bool:
    """True if keyboard focus is in LastZ or in this bot process (not another app like Cursor).

    Uses ``GetForegroundWindow`` and ``GetAncestor(..., GA_ROOT)`` (see Win32 docs) so focus on a
    child HWND inside the game still counts as LastZ.
    """
    import win32con
    import win32gui
    import win32process

    try:
        fg = win32gui.GetForegroundWindow()
        if not fg:
            return False
        # pywin32 returns (thread_id, process_id) — do not compare the first value to os.getpid().
        _, pid_fg = win32process.GetWindowThreadProcessId(fg)
        if pid_fg == os.getpid():
            return True
        fg_root = win32gui.GetAncestor(fg, win32con.GA_ROOT)
        if fg_root and int(fg_root) == int(root_lastz):
            return True
    except Exception:
        return False
    return False


def should_show_game_frame_border(wc: Any) -> bool:
    """Return True when the game top-level window is not minimized and should show the border.

    Uses ``GA_ROOT`` + ``IsIconic`` / ``GetWindowPlacement`` / ``WS_MINIMIZE`` so minimize is
    detected even when the capture HWND is not the same handle that receives ``SW_SHOWMINIMIZED``.
    Also requires foreground to be LastZ or this bot process so the border hides when another app
    (e.g. Cursor) has focus.
    """
    if sys.platform != "win32" or wc is None:
        return False
    if getattr(wc, "_is_desktop", False):
        return False
    hwnd = getattr(wc, "hwnd", None)
    if not hwnd:
        return False
    try:
        import win32gui

        if not win32gui.IsWindow(hwnd):
            return False
        root = _root_top_level_hwnd(hwnd)
        if not win32gui.IsWindow(root):
            return False
        if _is_minimized_or_non_interactive_top_level(root):
            return False
        if not win32gui.IsWindowVisible(root):
            return False
        if _is_dwm_cloaked(root):
            return False
        if not _foreground_allows_border(root):
            return False
    except Exception:
        return False
    return True

_HTTRANSPARENT = -1
_BORDER_PX = 3
# Bright green so the frame stands out on most game UIs
_RGB_BORDER = (0x00, 0xFF, 0x00)
_CLASS_NAME = "KhaLastZGameClientFrameBorder_{:02X}{:02X}{:02X}".format(*_RGB_BORDER)


def _wnd_proc(hwnd, msg, wparam, lparam):
    """Custom WndProc: green fill + click-through (HTTRANSPARENT).

    Relying on WNDCLASS.hbrBackground alone often leaves WS_POPUP stripes white;
    we must fill explicitly in WM_ERASEBKGND / WM_PAINT.
    """
    try:
        import win32api
        import win32con
        import win32gui

        if msg == win32con.WM_NCHITTEST:
            return _HTTRANSPARENT

        if msg == win32con.WM_ERASEBKGND:
            hdc = int(wparam)
            rect = win32gui.GetClientRect(hwnd)
            brush = win32gui.CreateSolidBrush(win32api.RGB(*_RGB_BORDER))
            try:
                win32gui.FillRect(hdc, rect, brush)
            finally:
                win32gui.DeleteObject(brush)
            return 1

        if msg == win32con.WM_PAINT:
            ps = win32gui.PAINTSTRUCT()
            hdc = win32gui.BeginPaint(hwnd, ps)
            try:
                rect = win32gui.GetClientRect(hwnd)
                brush = win32gui.CreateSolidBrush(win32api.RGB(*_RGB_BORDER))
                try:
                    win32gui.FillRect(hdc, rect, brush)
                finally:
                    win32gui.DeleteObject(brush)
            finally:
                win32gui.EndPaint(hwnd, ps)
            return 0

        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)
    except Exception:
        import win32gui

        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)


class GameClientFrameOverlay:
    """Four thin Win32 popup stripes forming a rectangle around the capture (client) region."""

    def __init__(self) -> None:
        self._hwnds: List[int] = []
        self._class_registered = False
        self._hinst: Optional[int] = None

    def _ensure_class(self) -> None:
        if self._class_registered:
            return
        import win32api
        import win32con
        import win32gui

        self._hinst = win32api.GetModuleHandle(None)
        wc = win32gui.WNDCLASS()
        wc.hInstance = self._hinst
        wc.lpszClassName = _CLASS_NAME
        wc.lpfnWndProc = _wnd_proc
        # Avoid default white erase; we paint green in _wnd_proc.
        wc.hbrBackground = win32gui.GetStockObject(win32con.NULL_BRUSH)
        try:
            win32gui.RegisterClass(wc)
        except Exception:
            # Class already registered (e.g. module reload)
            pass
        self._class_registered = True

    def _ensure_windows(self) -> None:
        if len(self._hwnds) == 4:
            return
        import win32api
        import win32con
        import win32gui

        self._ensure_class()
        assert self._hinst is not None
        self._destroy_windows()
        ex = (
            win32con.WS_EX_TOPMOST
            | win32con.WS_EX_NOACTIVATE
            | win32con.WS_EX_TOOLWINDOW
        )
        for _ in range(4):
            hwnd = win32gui.CreateWindowEx(
                ex,
                _CLASS_NAME,
                "",
                win32con.WS_POPUP,
                0,
                0,
                1,
                1,
                0,
                0,
                self._hinst,
                None,
            )
            self._hwnds.append(hwnd)

    def _destroy_windows(self) -> None:
        import win32gui

        for hwnd in self._hwnds:
            try:
                win32gui.DestroyWindow(hwnd)
            except Exception:
                pass
        self._hwnds.clear()

    def hide(self) -> None:
        """Hide the border without destroying HWNDs (fast path)."""
        import win32con
        import win32gui

        for hwnd in self._hwnds:
            try:
                win32gui.ShowWindow(hwnd, win32con.SW_HIDE)
            except Exception:
                pass

    def destroy(self) -> None:
        """Destroy overlay windows (call on app shutdown)."""
        self._destroy_windows()

    def update_from_wincap(self, wincap: Any) -> None:
        """Refresh geometry from ``wincap``, then show or hide the border as appropriate.

        Call each game-loop tick when PC mode and Is Running is ON. Encapsulates
        :func:`should_show_game_frame_border` and positioning.
        """
        if wincap is None:
            self.hide()
            return
        try:
            wincap.refresh_geometry()
        except Exception:
            pass
        if not should_show_game_frame_border(wincap):
            self.hide()
            return
        w = int(getattr(wincap, "w", 0) or 0)
        h = int(getattr(wincap, "h", 0) or 0)
        if w <= 0 or h <= 0:
            self.hide()
            return
        ox = int(getattr(wincap, "offset_x", 0) or 0)
        oy = int(getattr(wincap, "offset_y", 0) or 0)
        self.sync_client_rect(ox, oy, w, h)

    def sync_client_rect(self, client_left: int, client_top: int, w: int, h: int) -> None:
        """
        Position the border around the given client-area rect (screen pixels).

        client_left/client_top: top-left of the window client area in screen coordinates.
        """
        import win32con
        import win32gui

        if w <= 0 or h <= 0:
            self.hide()
            return

        t = _BORDER_PX
        ox, oy = int(client_left), int(client_top)
        wi, hi = int(w), int(h)

        # Outer frame: inset by t so the stroke sits around the client rect
        rects = [
            (ox - t, oy - t, wi + 2 * t, t),  # top
            (ox - t, oy + hi, wi + 2 * t, t),  # bottom
            (ox - t, oy - t, t, hi + 2 * t),  # left
            (ox + wi, oy - t, t, hi + 2 * t),  # right
        ]

        self._ensure_windows()
        for hwnd, (x, y, cw, ch) in zip(self._hwnds, rects):
            try:
                win32gui.SetWindowPos(
                    hwnd,
                    win32con.HWND_TOPMOST,
                    x,
                    y,
                    max(1, cw),
                    max(1, ch),
                    win32con.SWP_NOACTIVATE | win32con.SWP_SHOWWINDOW,
                )
                win32gui.ShowWindow(hwnd, win32con.SW_SHOWNOACTIVATE)
                win32gui.InvalidateRect(hwnd, None, True)
            except Exception as e:
                log.debug("[GameFrameOverlay] SetWindowPos failed: %s", e)
