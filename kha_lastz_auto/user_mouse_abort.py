"""
user_mouse_abort.py
-------------------
While a YAML function is running, detect unusually fast **physical** mouse movement
(Windows: WH_MOUSE_LL, non-injected moves only) and signal the main loop to abort
the current function — same effect as the UI Stop button.

Synthetic moves from pynput/pyautogui/SendInput are flagged LLMHF_INJECTED and are
ignored, so normal bot clicking does not trip this detector.
"""

from __future__ import annotations

import ctypes
import logging
import math
import sys
import threading
import time
from collections import deque
log = logging.getLogger("kha_lastz")

# ── Defaults (overridden via configure()) ─────────────────────────────────────
_enabled = False
_min_instant_speed_px_s = 7500.0  # peak speed between two physical moves
_window_sec = 0.12
_window_min_path_px = 380.0  # total path length in window_sec (physical only)
_cooldown_after_abort_sec = 0.45
# If True, ignore LLMHF_INJECTED (bot moves may also trip — use only to verify hook / drivers)
_count_all_moves_as_physical = False

_abort_event = threading.Event()
_lock = threading.Lock()
_recent: deque[tuple[float, int, int]] = deque(maxlen=64)
_last_x: int | None = None
_last_y: int | None = None
_last_t: float = 0.0
_cooldown_until = 0.0

# Debug / diagnostics (hook thread)
_stat_phys = 0
_stat_inj = 0
_stat_last_log_t = 0.0
_hook_error_last_log_t = 0.0
_max_inst_seen = 0.0  # max instant speed (physical samples) since last stat log
_max_path_seen = 0.0  # max path length in window before last stat log
_only_injected_warn_last_t = 0.0

_hook_thread: threading.Thread | None = None
_shutdown = threading.Event()
_hook_id = ctypes.c_void_p(None)
_hook_proc_ref = None  # keep alive for callback

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

WH_MOUSE_LL = 14
WM_MOUSEMOVE = 0x0200
HC_ACTION = 0
LLMHF_INJECTED = 0x01
LLMHF_LOWER_IL_INJECTED = 0x02
PM_REMOVE = 0x0001
WM_QUIT = 0x0012


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("pt", POINT),
        ("mouseData", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", ctypes.c_void_p),
        ("message", ctypes.c_uint32),
        ("wParam", ctypes.c_size_t),
        ("lParam", ctypes.c_size_t),
        ("time", ctypes.c_uint32),
        ("pt", POINT),
    ]


LRESULT = ctypes.c_ssize_t
HOOKPROC = ctypes.WINFUNCTYPE(
    LRESULT, ctypes.c_int, ctypes.wintypes.WPARAM, ctypes.wintypes.LPARAM
)


def _is_injected(flags: int) -> bool:
    return bool(flags & (LLMHF_INJECTED | LLMHF_LOWER_IL_INJECTED))


def _should_count_move(flags: int) -> bool:
    if _count_all_moves_as_physical:
        return True
    return not _is_injected(flags)


def _trim_recent(now: float) -> None:
    while _recent and now - _recent[0][0] > _window_sec:
        _recent.popleft()


def _path_length_recent() -> float:
    if len(_recent) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(_recent)):
        x0, y0 = _recent[i - 1][1], _recent[i - 1][2]
        x1, y1 = _recent[i][1], _recent[i][2]
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def _on_physical_move(x: int, y: int, now: float) -> None:
    global _last_x, _last_y, _last_t, _cooldown_until, _max_inst_seen, _max_path_seen

    if not _enabled or _shutdown.is_set():
        return
    if now < _cooldown_until:
        return

    with _lock:
        _trim_recent(now)
        if _last_x is not None and _last_y is not None:
            dist = math.hypot(x - _last_x, y - _last_y)
            dt = now - _last_t
            if dt > 1e-4:
                inst = dist / dt
                if inst > _max_inst_seen:
                    _max_inst_seen = inst
                if log.isEnabledFor(logging.DEBUG) and inst > 2000.0:
                    log.debug(
                        "[UserMouse] physical segment dist=%.1f dt=%.4fs inst=%.0f px/s (need >= %.0f)",
                        dist,
                        dt,
                        inst,
                        _min_instant_speed_px_s,
                    )
                if inst >= _min_instant_speed_px_s:
                    log.info(
                        "[UserMouse] Trip: instant speed %.0f px/s >= %.0f — abort latch ON",
                        inst,
                        _min_instant_speed_px_s,
                    )
                    _abort_event.set()
                    _cooldown_until = now + _cooldown_after_abort_sec
                    _recent.clear()
                    _last_x, _last_y, _last_t = x, y, now
                    return

        _recent.append((now, x, y))
        _trim_recent(now)
        plen = _path_length_recent()
        if plen > _max_path_seen:
            _max_path_seen = plen
        if log.isEnabledFor(logging.DEBUG) and plen > 80.0:
            log.debug(
                "[UserMouse] path in %.3fs window = %.1f px (need >= %.0f)",
                _window_sec,
                plen,
                _window_min_path_px,
            )
        if plen >= _window_min_path_px:
            log.info(
                "[UserMouse] Trip: path %.1f px in %.3fs window >= %.0f — abort latch ON",
                plen,
                _window_sec,
                _window_min_path_px,
            )
            _abort_event.set()
            _cooldown_until = now + _cooldown_after_abort_sec
            _recent.clear()

        _last_x, _last_y, _last_t = x, y, now


def _low_level_proc(n_code: int, w_param: ctypes.wintypes.WPARAM, l_param: ctypes.wintypes.LPARAM) -> LRESULT:
    global _stat_phys, _stat_inj, _stat_last_log_t, _max_inst_seen, _max_path_seen
    global _hook_error_last_log_t, _only_injected_warn_last_t

    try:
        if n_code == HC_ACTION and int(w_param) == WM_MOUSEMOVE:
            st = ctypes.cast(l_param, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
            flags = int(st.flags)
            inj = _is_injected(flags)
            if inj:
                _stat_inj += 1
            else:
                _stat_phys += 1

            if _should_count_move(flags):
                _on_physical_move(int(st.pt.x), int(st.pt.y), time.perf_counter())

            now_mono = time.perf_counter()
            if (now_mono - _stat_last_log_t) >= 1.0:
                if (
                    _stat_phys == 0
                    and _stat_inj > 15
                    and not _count_all_moves_as_physical
                    and (now_mono - _only_injected_warn_last_t) >= 30.0
                ):
                    _only_injected_warn_last_t = now_mono
                    log.warning(
                        "[UserMouse] Last 1s: 0 physical / %d injected moves — trip logic never runs. "
                        "Drivers or software may mark all input as injected. "
                        "Try fast_user_mouse_count_all_moves_as_physical: true (bot moves may also trip).",
                        _stat_inj,
                    )
                if log.isEnabledFor(logging.DEBUG):
                    log.debug(
                        "[UserMouse] hook stats (1s): physical=%d injected=%d | max_inst=%.0f px/s max_path=%.0f px | "
                        "thresholds inst>=%.0f path>=%.0f in %.3fs | inj_bypass=%s",
                        _stat_phys,
                        _stat_inj,
                        _max_inst_seen,
                        _max_path_seen,
                        _min_instant_speed_px_s,
                        _window_min_path_px,
                        _window_sec,
                        _count_all_moves_as_physical,
                    )
                _stat_phys = 0
                _stat_inj = 0
                _max_inst_seen = 0.0
                _max_path_seen = 0.0
                _stat_last_log_t = now_mono
    except Exception:
        now_e = time.perf_counter()
        if now_e - _hook_error_last_log_t >= 5.0:
            _hook_error_last_log_t = now_e
            log.exception("[UserMouse] low_level_proc error (throttled to 1/5s)")

    return user32.CallNextHookEx(_hook_id, n_code, w_param, l_param)


def configure(
    enabled: bool,
    min_instant_speed_px_s: float | None = None,
    window_sec: float | None = None,
    window_min_path_px: float | None = None,
    cooldown_after_abort_sec: float | None = None,
    count_all_moves_as_physical: bool | None = None,
) -> None:
    """Update detector parameters (safe from any thread)."""
    global _enabled, _min_instant_speed_px_s, _window_sec, _window_min_path_px
    global _cooldown_after_abort_sec, _count_all_moves_as_physical
    with _lock:
        _enabled = bool(enabled)
        if min_instant_speed_px_s is not None:
            _min_instant_speed_px_s = max(500.0, float(min_instant_speed_px_s))
        if window_sec is not None:
            _window_sec = max(0.05, float(window_sec))
        if window_min_path_px is not None:
            _window_min_path_px = max(50.0, float(window_min_path_px))
        if cooldown_after_abort_sec is not None:
            _cooldown_after_abort_sec = max(0.1, float(cooldown_after_abort_sec))
        if count_all_moves_as_physical is not None:
            _count_all_moves_as_physical = bool(count_all_moves_as_physical)
        if not _enabled:
            _recent.clear()
            _abort_event.clear()
        log.info(
            "[UserMouse] configure: enabled=%s inst>=%.0f px/s path>=%.0f in %.3fs cooldown=%.2fs "
            "count_all_as_physical=%s",
            _enabled,
            _min_instant_speed_px_s,
            _window_min_path_px,
            _window_sec,
            _cooldown_after_abort_sec,
            _count_all_moves_as_physical,
        )


def consume_abort_request() -> bool:
    """If a fast user move was detected, clear the latch and return True."""
    if _abort_event.is_set():
        _abort_event.clear()
        log.debug("[UserMouse] consume_abort_request -> True (game loop will abort function)")
        return True
    return False


def is_enabled() -> bool:
    return _enabled


def _hook_thread_main() -> None:
    global _hook_id, _hook_proc_ref

    if sys.platform != "win32":
        return

    h_mod = kernel32.GetModuleHandleW(None)
    _hook_proc_ref = HOOKPROC(_low_level_proc)
    hid = user32.SetWindowsHookExW(WH_MOUSE_LL, _hook_proc_ref, h_mod, 0)
    if not hid:
        err = ctypes.get_last_error()
        log.warning("[UserMouse] SetWindowsHookExW failed (err=%s); fast-mouse abort disabled", err)
        return
    _hook_id = ctypes.c_void_p(hid)
    log.info(
        "[UserMouse] WH_MOUSE_LL installed (thread=%s). Physical-only moves count unless "
        "fast_user_mouse_count_all_moves_as_physical is true. Enable DEBUG on logger 'kha_lastz' for hook stats.",
        threading.current_thread().name,
    )

    msg = MSG()
    while not _shutdown.is_set():
        r = user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE)
        if r:
            if msg.message == WM_QUIT:
                break
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        elif _shutdown.wait(0.02):
            break

    if _hook_id.value:
        user32.UnhookWindowsHookEx(_hook_id)
        _hook_id = ctypes.c_void_p(None)
    log.debug("[UserMouse] WH_MOUSE_LL unhooked")


def start() -> None:
    """Start the low-level hook thread (Windows only). No-op if disabled, already running, or not Windows."""
    global _hook_thread

    if sys.platform != "win32":
        log.info("[UserMouse] Fast-mouse abort is only implemented on Windows; skipping")
        return
    if not _enabled:
        log.info("[UserMouse] abort_on_fast_user_mouse disabled — hook not installed")
        return
    if _hook_thread is not None and _hook_thread.is_alive():
        return

    _shutdown.clear()
    _hook_thread = threading.Thread(target=_hook_thread_main, name="UserMouseHook", daemon=True)
    _hook_thread.start()


def stop() -> None:
    """Stop hook thread (call on app shutdown)."""
    _shutdown.set()
    if _hook_thread is not None:
        _hook_thread.join(timeout=2.0)
    _hook_thread = None


def apply_settings_from_dict(general: dict | None, yaml_fallback: dict | None = None) -> None:
    """Load toggles from general_settings (or config.yaml root)."""
    g = general or {}
    y = yaml_fallback or {}

    def _get_bool(key: str, default: bool) -> bool:
        if key in g:
            return bool(g[key])
        if key in y:
            return bool(y[key])
        return default

    def _get_float(key: str, default: float) -> float:
        for src in (g, y):
            if key in src:
                try:
                    return float(src[key])
                except (TypeError, ValueError):
                    pass
        return default

    en = _get_bool("abort_on_fast_user_mouse", True)
    count_all = _get_bool("fast_user_mouse_count_all_moves_as_physical", False)
    configure(
        enabled=en,
        min_instant_speed_px_s=_get_float("fast_user_mouse_min_speed_px_s", 7500.0),
        window_sec=_get_float("fast_user_mouse_window_sec", 0.12),
        window_min_path_px=_get_float("fast_user_mouse_window_min_dist_px", 380.0),
        cooldown_after_abort_sec=_get_float("fast_user_mouse_cooldown_sec", 0.45),
        count_all_moves_as_physical=count_all,
    )
