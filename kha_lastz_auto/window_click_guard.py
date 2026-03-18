"""
WindowClickGuard: blocks user hardware mouse button events that fall
outside the game window rect while still allowing all synthesized
(SendInput / pynput) clicks through.

Mechanism: installs a global WH_MOUSE_LL hook on a dedicated message-loop
thread. Every mouse button event is inspected:
  - Injected events (LLMHF_INJECTED flag set) → always pass through.
    These are produced by FastClicker / pynput and must reach the game.
  - Hardware clicks inside the guard rect → pass through.
  - Hardware clicks outside the guard rect → suppressed (return 1).

Usage:
    guard = WindowClickGuard()
    guard.start(left, top, right, bottom)   # screen coords
    ...
    guard.stop()
"""
import ctypes
import ctypes.wintypes
import logging
import threading

log = logging.getLogger("kha_lastz")

# ── Windows constants ────────────────────────────────────────────────────────
_WH_MOUSE_LL    = 14
_WM_QUIT        = 0x0012
_WM_LBUTTONDOWN = 0x0201
_WM_LBUTTONUP   = 0x0202
_WM_RBUTTONDOWN = 0x0204
_WM_RBUTTONUP   = 0x0205
_LLMHF_INJECTED = 0x1

_BUTTON_MESSAGES = frozenset({
    _WM_LBUTTONDOWN, _WM_LBUTTONUP,
    _WM_RBUTTONDOWN, _WM_RBUTTONUP,
})


class _MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("pt",          ctypes.wintypes.POINT),
        ("mouseData",   ctypes.wintypes.DWORD),
        ("flags",       ctypes.wintypes.DWORD),
        ("time",        ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_ulong),
    ]


_HOOKPROC = ctypes.WINFUNCTYPE(
    ctypes.c_long,
    ctypes.c_int,
    ctypes.wintypes.WPARAM,
    ctypes.wintypes.LPARAM,
)


class WindowClickGuard:
    """Block hardware mouse button clicks that land outside a given screen rect."""

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._thread_id: int = 0
        self._rect: tuple | None = None    # (left, top, right, bottom)
        self._hook = None
        self._started = threading.Event()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, left: int, top: int, right: int, bottom: int) -> None:
        """Install the hook and start guarding the given screen rect."""
        self.stop()
        self._rect = (left, top, right, bottom)
        self._started.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="WindowClickGuard",
        )
        self._thread.start()
        if not self._started.wait(timeout=2.0):
            log.warning("[WindowClickGuard] Hook did not install within 2s")
            return
        log.info(
            "[WindowClickGuard] started — guarding rect (%d,%d)→(%d,%d)",
            left, top, right, bottom,
        )

    def stop(self) -> None:
        """Uninstall the hook and stop the guard thread."""
        if self._thread and self._thread.is_alive():
            if self._thread_id:
                ctypes.windll.user32.PostThreadMessageW(
                    self._thread_id, _WM_QUIT, 0, 0,
                )
            self._thread.join(timeout=2.0)
        self._thread = None
        self._thread_id = 0
        self._hook = None

    @property
    def is_active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        user32   = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        self._thread_id = kernel32.GetCurrentThreadId()

        # Must set argtypes before calling so ctypes knows argument 2 is a
        # callback pointer (WinFunctionType instance), not a plain c_void_p.
        user32.SetWindowsHookExW.argtypes = [
            ctypes.c_int,
            _HOOKPROC,
            ctypes.wintypes.HINSTANCE,
            ctypes.wintypes.DWORD,
        ]
        user32.SetWindowsHookExW.restype = ctypes.wintypes.HHOOK

        user32.CallNextHookEx.argtypes = [
            ctypes.wintypes.HHOOK,
            ctypes.c_int,
            ctypes.wintypes.WPARAM,
            ctypes.wintypes.LPARAM,
        ]
        user32.CallNextHookEx.restype = ctypes.c_long

        user32.UnhookWindowsHookEx.argtypes = [ctypes.wintypes.HHOOK]
        user32.UnhookWindowsHookEx.restype  = ctypes.wintypes.BOOL

        def _hook_proc(n_code: int, w_param: int, l_param: int) -> int:
            if n_code >= 0 and w_param in _BUTTON_MESSAGES:
                info = ctypes.cast(l_param, ctypes.POINTER(_MSLLHOOKSTRUCT)).contents
                # Let injected (synthesized) events through unconditionally —
                # these are FastClicker / pynput clicks targeting the game.
                if not (info.flags & _LLMHF_INJECTED):
                    x, y = info.pt.x, info.pt.y
                    left, top, right, bottom = self._rect
                    if not (left <= x < right and top <= y < bottom):
                        return 1   # suppress: do NOT call CallNextHookEx
            return user32.CallNextHookEx(self._hook, n_code, w_param, l_param)

        cb = _HOOKPROC(_hook_proc)
        self._hook = user32.SetWindowsHookExW(
            _WH_MOUSE_LL, cb, None, 0,
        )
        if not self._hook:
            err = kernel32.GetLastError()
            log.warning("[WindowClickGuard] SetWindowsHookExW failed (err=%d)", err)
            self._started.set()
            return

        self._started.set()

        msg = ctypes.wintypes.MSG()
        while True:
            ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if ret <= 0:
                break
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

        user32.UnhookWindowsHookEx(self._hook)
        self._hook = None
        log.info("[WindowClickGuard] stopped")
