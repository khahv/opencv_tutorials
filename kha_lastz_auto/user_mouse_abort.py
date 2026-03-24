"""
user_mouse_abort.py
-------------------
While a YAML function is running, detect a deliberate **horizontal zoom-shake**
gesture (like macOS accessibility zoom: pull left, then pull right quickly) and
signal the main loop to abort the current function (same as the UI Stop button).
Detection logic lives in ``user_mouse_shake_detect.py`` (test with
``python test_fast_user_mouse.py --synthetic-tests``).

**Default backend (smooth cursor):** A background thread polls ``GetCursorPos`` on
a fixed interval. This does **not** install ``WH_MOUSE_LL``, so it does not sit in
the low-level mouse delivery path — dragging and dense ``WM_MOUSE_MOVE`` streams
stay smooth.

**Optional backend:** ``fast_user_mouse_use_low_level_hook: true`` restores the
``WH_MOUSE_LL`` hook. That path can ignore **injected** moves (``LLMHF_INJECTED``)
so synthetic bot input does not trip the detector, but Windows still invokes the
hook before the rest of the input pipeline — it can make dragging feel stuttery.

**While a mouse button is held:** Samples are ignored and internal state is reset.

**Bot cursor teleports (polling only):** Call ``suppress_trip_for_sec()`` before
synthetic cursor moves (see ``FunctionRunner._safe_move``, ``FastClicker``, and
``zoom_helpers``).

**UI "Is Running":** ``set_ui_running_source(bot_paused)`` — sampling runs only when
``paused`` is false. When unset (e.g. standalone tests), monitoring stays always on.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import queue
import sys
import threading
import time
from collections import deque

from user_mouse_shake_detect import ZoomShakeParams, detect_zoom_shake

# When True, emit [UserMouse] logs (configure, poll/hook stats, trip, consume_abort, start/skip, etc.).
# Set to False to silence mouse shake–detector chatter in normal runs.
LOG_USER_MOUSE_SHAKE_DETECT = False

log = logging.getLogger("kha_lastz")

# ── Defaults (overridden via configure()) ─────────────────────────────────────
_enabled = False
# False = poll GetCursorPos (no global low-level hook). True = WH_MOUSE_LL + queue.
_use_low_level_hook = False
_poll_interval_sec = 0.008
# Horizontal macOS-style zoom shake: move left by >= left leg, then right by >= right leg, within window.
_gesture_left_leg_px = 75.0
_gesture_right_leg_px = 75.0
_gesture_window_sec = 0.42
# Max time from local high→trough (left stroke) and trough→recovery high (right stroke); 0 = disable timing cap.
_gesture_max_downstroke_sec = 0.22
_gesture_max_upstroke_sec = 0.22
_gesture_min_leg_balance_ratio = 0.4
_gesture_max_single_leg_px = 400.0
_cooldown_after_abort_sec = 0.45
_shake_params: ZoomShakeParams
_count_all_moves_as_physical = False

_abort_event = threading.Event()
_lock = threading.Lock()
_recent: deque[tuple[float, int, int]] = deque(maxlen=96)
_cooldown_until = 0.0
_suppress_trip_until = 0.0

# Debug / diagnostics
_stat_phys = 0
_stat_inj = 0
_stat_poll_samples = 0
_hook_error_last_log_t = 0.0
_max_gesture_score_seen = 0.0  # max (left_leg + right_leg) seen in window when shake matched
_only_injected_warn_last_t = 0.0

_MOVE_QUEUE_MAX = 128
_move_queue: queue.Queue[tuple[int, int, float]] = queue.Queue(maxsize=_MOVE_QUEUE_MAX)
_move_worker_thread: threading.Thread | None = None
_move_worker_shutdown = threading.Event()

_hook_thread: threading.Thread | None = None
_poll_thread: threading.Thread | None = None
_hook_stat_reporter_thread: threading.Thread | None = None
_hook_stat_reporter_shutdown = threading.Event()
_shutdown = threading.Event()
_hook_id = ctypes.c_void_p(None)
_hook_proc_ref = None

# Same object as main/UI: paused=True means Is Running is OFF — no mouse detect then.
_ui_paused_ref: dict | None = None

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short

WH_MOUSE_LL = 14
WM_MOUSEMOVE = 0x0200
HC_ACTION = 0
LLMHF_INJECTED = 0x01
LLMHF_LOWER_IL_INJECTED = 0x02
PM_REMOVE = 0x0001
WM_QUIT = 0x0012

# Virtual keys for mouse buttons (GetAsyncKeyState)
_VK_MOUSE_BUTTONS = (0x01, 0x02, 0x04, 0x05, 0x06)


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


# Fix GetCursorPos argtypes: POINT class is defined above; re-bind after class def
user32.GetCursorPos.argtypes = [ctypes.POINTER(POINT)]
user32.GetCursorPos.restype = ctypes.wintypes.BOOL


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
        ("lParam", ctypes.c_ssize_t),
        ("time", ctypes.c_uint32),
        ("pt", POINT),
    ]


LRESULT = ctypes.c_ssize_t
HOOKPROC = ctypes.WINFUNCTYPE(
    LRESULT, ctypes.c_int, ctypes.c_size_t, ctypes.c_ssize_t
)

user32.CallNextHookEx.restype = LRESULT
user32.CallNextHookEx.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_size_t,
    ctypes.c_ssize_t,
]


def _is_injected(flags: int) -> bool:
    return bool(flags & (LLMHF_INJECTED | LLMHF_LOWER_IL_INJECTED))


def _should_count_move(flags: int) -> bool:
    if _count_all_moves_as_physical:
        return True
    return not _is_injected(flags)


def _any_mouse_button_down() -> bool:
    """True if any standard mouse button is held (physical state)."""
    for vk in _VK_MOUSE_BUTTONS:
        if user32.GetAsyncKeyState(vk) & 0x8000:
            return True
    return False


def set_ui_running_source(bot_paused: dict | None) -> None:
    """Bind UI pause state: mouse detect runs only when ``paused`` is false."""
    global _ui_paused_ref
    _ui_paused_ref = bot_paused


def _ui_is_running() -> bool:
    """True if UI Is Running is on (bot not globally paused)."""
    ref = _ui_paused_ref
    if ref is None:
        return True
    return not bool(ref.get("paused", True))


def on_ui_paused_edge() -> None:
    """Call when UI transitions to paused (Is Running → off). Drains hook queue and resets motion state."""
    global _recent
    _drain_move_queue_safely()
    with _lock:
        _recent.clear()
    _abort_event.clear()


def suppress_trip_for_sec(seconds: float) -> None:
    """Extend a deadline during which trip detection is suppressed but cursor state stays synced.

    Call this immediately before synthetic cursor moves (polling backend cannot see
    LLMHF_INJECTED). Typical value: 0.12--0.2 s.
    """
    global _suppress_trip_until
    if seconds <= 0:
        return
    deadline = time.perf_counter() + float(seconds)
    with _lock:
        if deadline > _suppress_trip_until:
            _suppress_trip_until = deadline


def _trim_recent(now: float) -> None:
    while _recent and now - _recent[0][0] > _gesture_window_sec:
        _recent.popleft()


def _rebuild_shake_params() -> None:
    """Sync frozen ZoomShakeParams from module globals (call under ``_lock`` when updating)."""
    global _shake_params
    _shake_params = ZoomShakeParams(
        left_leg_px=_gesture_left_leg_px,
        right_leg_px=_gesture_right_leg_px,
        window_sec=_gesture_window_sec,
        max_downstroke_sec=_gesture_max_downstroke_sec,
        max_upstroke_sec=_gesture_max_upstroke_sec,
        min_leg_balance_ratio=_gesture_min_leg_balance_ratio,
        max_single_leg_px=_gesture_max_single_leg_px,
    )


_rebuild_shake_params()


def _drain_move_queue_safely() -> None:
    try:
        while True:
            _move_queue.get_nowait()
    except queue.Empty:
        pass


def _enqueue_move(x: int, y: int, t: float) -> None:
    try:
        _move_queue.put_nowait((x, y, t))
    except queue.Full:
        try:
            _move_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            _move_queue.put_nowait((x, y, t))
        except queue.Full:
            pass


def _move_worker_main() -> None:
    while not _move_worker_shutdown.is_set():
        try:
            first = _move_queue.get(timeout=0.08)
        except queue.Empty:
            continue
        _on_physical_move(first[0], first[1], first[2])
        while True:
            try:
                x2, y2, t2 = _move_queue.get_nowait()
            except queue.Empty:
                break
            _on_physical_move(x2, y2, t2)
    while True:
        try:
            x2, y2, t2 = _move_queue.get_nowait()
        except queue.Empty:
            break
        _on_physical_move(x2, y2, t2)


def _on_physical_move(x: int, y: int, now: float) -> None:
    global _cooldown_until, _max_gesture_score_seen

    if not _enabled or _shutdown.is_set() or not _ui_is_running():
        return
    if now < _cooldown_until:
        return

    if now < _suppress_trip_until:
        with _lock:
            _recent.clear()
        return

    deferred_log: str | None = None

    with _lock:
        _recent.append((now, x, y))
        _trim_recent(now)
        shake = detect_zoom_shake(list(_recent), _shake_params)
        if shake is not None:
            leg_l, leg_r, tspan = shake.left_leg_px, shake.right_leg_px, shake.span_sec
            score = leg_l + leg_r
            if score > _max_gesture_score_seen:
                _max_gesture_score_seen = score
            _abort_event.set()
            _cooldown_until = now + _cooldown_after_abort_sec
            _recent.clear()
            deferred_log = "left={:.0f}px right={:.0f}px span={:.2f}s".format(leg_l, leg_r, tspan)

    if deferred_log is not None and LOG_USER_MOUSE_SHAKE_DETECT:
        log.info(
            "[UserMouse] Trip: zoom-shake (left then right) %s — abort latch ON",
            deferred_log,
        )


def _hook_stat_reporter_main() -> None:
    global _only_injected_warn_last_t, _stat_phys, _stat_inj, _stat_poll_samples
    global _max_gesture_score_seen

    while not _hook_stat_reporter_shutdown.wait(1.0):
        if _hook_stat_reporter_shutdown.is_set():
            break
        mg = _max_gesture_score_seen
        _max_gesture_score_seen = 0.0
        now_mono = time.perf_counter()
        if _use_low_level_hook:
            sp, si = _stat_phys, _stat_inj
            _stat_phys = 0
            _stat_inj = 0
            if (
                LOG_USER_MOUSE_SHAKE_DETECT
                and sp == 0
                and si > 15
                and not _count_all_moves_as_physical
                and (now_mono - _only_injected_warn_last_t) >= 30.0
            ):
                _only_injected_warn_last_t = now_mono
                log.warning(
                    "[UserMouse] Last 1s: 0 physical / %d injected moves — trip logic never runs. "
                    "Drivers or software may mark all input as injected. "
                    "Try fast_user_mouse_count_all_moves_as_physical: true (bot moves may also trip).",
                    si,
                )
            if LOG_USER_MOUSE_SHAKE_DETECT and log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "[UserMouse] hook stats (1s): physical=%d injected=%d | max_shake_sum=%.0f px | "
                    "shake left>=%.0f right>=%.0f bal>=%.2f max_leg<=%.0f window=%.3fs | inj_bypass=%s",
                    sp,
                    si,
                    mg,
                    _gesture_left_leg_px,
                    _gesture_right_leg_px,
                    _gesture_min_leg_balance_ratio,
                    _gesture_max_single_leg_px,
                    _gesture_window_sec,
                    _count_all_moves_as_physical,
                )
        else:
            n = _stat_poll_samples
            _stat_poll_samples = 0
            if LOG_USER_MOUSE_SHAKE_DETECT and log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "[UserMouse] poll stats (1s): samples=%d | max_shake_sum=%.0f px | interval=%.4fs | "
                    "shake left>=%.0f right>=%.0f bal>=%.2f max_leg<=%.0f window=%.3fs",
                    n,
                    mg,
                    _poll_interval_sec,
                    _gesture_left_leg_px,
                    _gesture_right_leg_px,
                    _gesture_min_leg_balance_ratio,
                    _gesture_max_single_leg_px,
                    _gesture_window_sec,
                )


def _low_level_proc(n_code: int, w_param: int, l_param: int) -> LRESULT:
    global _stat_phys, _stat_inj
    global _hook_error_last_log_t

    try:
        if n_code == HC_ACTION and int(w_param) == WM_MOUSEMOVE:
            if not _ui_is_running():
                return user32.CallNextHookEx(_hook_id, n_code, w_param, l_param)
            now = time.perf_counter()
            if now < _suppress_trip_until:
                return user32.CallNextHookEx(_hook_id, n_code, w_param, l_param)
            if _any_mouse_button_down():
                return user32.CallNextHookEx(_hook_id, n_code, w_param, l_param)

            st = ctypes.cast(l_param, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
            flags = int(st.flags)
            inj = _is_injected(flags)
            if inj:
                _stat_inj += 1
            else:
                _stat_phys += 1

            if _should_count_move(flags):
                _enqueue_move(int(st.pt.x), int(st.pt.y), now)
    except Exception:
        now_e = time.perf_counter()
        if LOG_USER_MOUSE_SHAKE_DETECT and now_e - _hook_error_last_log_t >= 5.0:
            _hook_error_last_log_t = now_e
            log.exception("[UserMouse] low_level_proc error (throttled to 1/5s)")

    return user32.CallNextHookEx(_hook_id, n_code, w_param, l_param)


def _poll_thread_main() -> None:
    """Sample cursor position off the input hook path."""
    global _stat_poll_samples, _recent

    pt = POINT()
    while not _shutdown.is_set():
        # Slow down when disabled or UI paused — no GetCursorPos, low CPU.
        interval = (
            max(0.05, _poll_interval_sec)
            if (not _enabled or not _ui_is_running())
            else _poll_interval_sec
        )
        if _shutdown.wait(interval):
            break
        if not _enabled:
            continue

        if not _ui_is_running():
            with _lock:
                _recent.clear()
            _abort_event.clear()
            continue

        if _any_mouse_button_down():
            with _lock:
                _recent.clear()
            continue

        if not user32.GetCursorPos(ctypes.byref(pt)):
            continue
        now = time.perf_counter()
        _stat_poll_samples += 1
        _on_physical_move(int(pt.x), int(pt.y), now)


def configure(
    enabled: bool,
    gesture_left_leg_px: float | None = None,
    gesture_right_leg_px: float | None = None,
    gesture_window_sec: float | None = None,
    gesture_max_downstroke_sec: float | None = None,
    gesture_max_upstroke_sec: float | None = None,
    gesture_min_leg_balance_ratio: float | None = None,
    gesture_max_single_leg_px: float | None = None,
    cooldown_after_abort_sec: float | None = None,
    count_all_moves_as_physical: bool | None = None,
    use_low_level_hook: bool | None = None,
    poll_interval_sec: float | None = None,
) -> None:
    """Update detector parameters (safe from any thread)."""
    global _enabled, _gesture_left_leg_px, _gesture_right_leg_px, _gesture_window_sec
    global _gesture_max_downstroke_sec, _gesture_max_upstroke_sec
    global _gesture_min_leg_balance_ratio, _gesture_max_single_leg_px
    global _cooldown_after_abort_sec, _count_all_moves_as_physical
    global _use_low_level_hook, _poll_interval_sec, _suppress_trip_until

    with _lock:
        _enabled = bool(enabled)
        if gesture_left_leg_px is not None:
            _gesture_left_leg_px = max(20.0, float(gesture_left_leg_px))
        if gesture_right_leg_px is not None:
            _gesture_right_leg_px = max(20.0, float(gesture_right_leg_px))
        if gesture_window_sec is not None:
            _gesture_window_sec = max(0.12, float(gesture_window_sec))
        if gesture_max_downstroke_sec is not None:
            _gesture_max_downstroke_sec = max(0.0, float(gesture_max_downstroke_sec))
        if gesture_max_upstroke_sec is not None:
            _gesture_max_upstroke_sec = max(0.0, float(gesture_max_upstroke_sec))
        if gesture_min_leg_balance_ratio is not None:
            _gesture_min_leg_balance_ratio = min(1.0, max(0.05, float(gesture_min_leg_balance_ratio)))
        if gesture_max_single_leg_px is not None:
            _gesture_max_single_leg_px = max(0.0, float(gesture_max_single_leg_px))
        if cooldown_after_abort_sec is not None:
            _cooldown_after_abort_sec = max(0.1, float(cooldown_after_abort_sec))
        if count_all_moves_as_physical is not None:
            _count_all_moves_as_physical = bool(count_all_moves_as_physical)
        if use_low_level_hook is not None:
            _use_low_level_hook = bool(use_low_level_hook)
        if poll_interval_sec is not None:
            _poll_interval_sec = max(0.002, min(0.05, float(poll_interval_sec)))
        if not _enabled:
            _recent.clear()
            _abort_event.clear()
            _suppress_trip_until = 0.0
        _rebuild_shake_params()
    if not _enabled:
        _drain_move_queue_safely()

    if not LOG_USER_MOUSE_SHAKE_DETECT:
        return

    if _use_low_level_hook:
        log.info(
            "[UserMouse] configure: mode=WH_MOUSE_LL enabled=%s zoom-shake left>=%.0f right>=%.0f "
            "bal>=%.2f max_leg<=%.0f window=%.3fs down<=%.3fs up<=%.3fs cooldown=%.2fs count_all_as_physical=%s",
            _enabled,
            _gesture_left_leg_px,
            _gesture_right_leg_px,
            _gesture_min_leg_balance_ratio,
            _gesture_max_single_leg_px,
            _gesture_window_sec,
            _gesture_max_downstroke_sec,
            _gesture_max_upstroke_sec,
            _cooldown_after_abort_sec,
            _count_all_moves_as_physical,
        )
    else:
        log.info(
            "[UserMouse] configure: mode=GetCursorPos poll (%.4fs) enabled=%s zoom-shake left>=%.0f "
            "right>=%.0f bal>=%.2f max_leg<=%.0f window=%.3fs cooldown=%.2fs — no WH_MOUSE_LL",
            _poll_interval_sec,
            _enabled,
            _gesture_left_leg_px,
            _gesture_right_leg_px,
            _gesture_min_leg_balance_ratio,
            _gesture_max_single_leg_px,
            _gesture_window_sec,
            _cooldown_after_abort_sec,
        )


def consume_abort_request() -> bool:
    if _abort_event.is_set():
        _abort_event.clear()
        if LOG_USER_MOUSE_SHAKE_DETECT:
            log.debug("[UserMouse] consume_abort_request -> True (game loop will abort function)")
        return True
    return False


def is_enabled() -> bool:
    return _enabled


def _hook_thread_main() -> None:
    global _hook_id, _hook_proc_ref

    if sys.platform != "win32":
        return

    _hook_proc_ref = HOOKPROC(_low_level_proc)
    hid = user32.SetWindowsHookExW(WH_MOUSE_LL, _hook_proc_ref, None, 0)
    if not hid:
        err = ctypes.get_last_error()
        log.warning("[UserMouse] SetWindowsHookExW failed (err=%s); fast-mouse abort disabled", err)
        return
    _hook_id = ctypes.c_void_p(hid)
    if LOG_USER_MOUSE_SHAKE_DETECT:
        log.info(
            "[UserMouse] WH_MOUSE_LL installed (thread=%s). Physical-only moves count unless "
            "fast_user_mouse_count_all_moves_as_physical is true.",
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
    if LOG_USER_MOUSE_SHAKE_DETECT:
        log.debug("[UserMouse] WH_MOUSE_LL unhooked")


def start() -> None:
    """Start sampling thread (poll or hook + queue worker). Windows only."""
    global _hook_thread, _poll_thread, _hook_stat_reporter_thread, _move_worker_thread

    if sys.platform != "win32":
        if LOG_USER_MOUSE_SHAKE_DETECT:
            log.info("[UserMouse] Fast-mouse abort is only implemented on Windows; skipping")
        return
    if not _enabled:
        if LOG_USER_MOUSE_SHAKE_DETECT:
            log.info("[UserMouse] abort_on_fast_user_mouse disabled — not started")
        return
    if _use_low_level_hook:
        if _hook_thread is not None and _hook_thread.is_alive():
            return
    else:
        if _poll_thread is not None and _poll_thread.is_alive():
            return

    _shutdown.clear()
    _hook_stat_reporter_shutdown.clear()
    _drain_move_queue_safely()

    _hook_stat_reporter_thread = threading.Thread(
        target=_hook_stat_reporter_main, name="UserMouseStatLog", daemon=True
    )
    _hook_stat_reporter_thread.start()

    if _use_low_level_hook:
        _move_worker_shutdown.clear()
        _move_worker_thread = threading.Thread(
            target=_move_worker_main, name="UserMouseMoveProc", daemon=True
        )
        _move_worker_thread.start()
        _hook_thread = threading.Thread(target=_hook_thread_main, name="UserMouseHook", daemon=True)
        _hook_thread.start()
    else:
        _move_worker_thread = None
        _hook_thread = None
        _poll_thread = threading.Thread(target=_poll_thread_main, name="UserMousePoll", daemon=True)
        _poll_thread.start()


def stop() -> None:
    global _hook_thread, _hook_stat_reporter_thread, _move_worker_thread, _poll_thread

    _shutdown.set()
    if _hook_thread is not None:
        _hook_thread.join(timeout=2.0)
    _hook_thread = None
    if _poll_thread is not None:
        _poll_thread.join(timeout=2.0)
    _poll_thread = None

    _hook_stat_reporter_shutdown.set()
    if _hook_stat_reporter_thread is not None:
        _hook_stat_reporter_thread.join(timeout=2.0)
    _hook_stat_reporter_thread = None

    _move_worker_shutdown.set()
    if _move_worker_thread is not None:
        _move_worker_thread.join(timeout=2.0)
    _move_worker_thread = None
    _drain_move_queue_safely()


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

    def _get_float_optional(key: str) -> float | None:
        for src in (g, y):
            if key in src:
                try:
                    return float(src[key])
                except (TypeError, ValueError):
                    pass
        return None

    en = _get_bool("abort_on_fast_user_mouse", True)
    count_all = _get_bool("fast_user_mouse_count_all_moves_as_physical", False)
    use_hook = _get_bool("fast_user_mouse_use_low_level_hook", False)

    left = _get_float_optional("fast_user_mouse_gesture_left_leg_px")
    right = _get_float_optional("fast_user_mouse_gesture_right_leg_px")
    legacy_dist = _get_float_optional("fast_user_mouse_window_min_dist_px")
    if left is None and legacy_dist is not None:
        left = max(35.0, legacy_dist * 0.35)
    if right is None and legacy_dist is not None:
        right = max(35.0, legacy_dist * 0.35)

    gwin = _get_float_optional("fast_user_mouse_gesture_window_sec")
    if gwin is None:
        gwin = _get_float("fast_user_mouse_window_sec", 0.42)

    configure(
        enabled=en,
        gesture_left_leg_px=left,
        gesture_right_leg_px=right,
        gesture_window_sec=gwin,
        gesture_max_downstroke_sec=_get_float_optional("fast_user_mouse_gesture_max_downstroke_sec"),
        gesture_max_upstroke_sec=_get_float_optional("fast_user_mouse_gesture_max_upstroke_sec"),
        gesture_min_leg_balance_ratio=_get_float_optional("fast_user_mouse_gesture_min_leg_balance_ratio"),
        gesture_max_single_leg_px=_get_float_optional("fast_user_mouse_gesture_max_single_leg_px"),
        cooldown_after_abort_sec=_get_float("fast_user_mouse_cooldown_sec", 0.45),
        count_all_moves_as_physical=count_all,
        use_low_level_hook=use_hook,
        poll_interval_sec=_get_float("fast_user_mouse_poll_interval_sec", 0.008),
    )
