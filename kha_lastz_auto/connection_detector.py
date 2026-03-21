"""
Connection detector: every N minutes, check if the game is still connected.

Runs only when Is Running (UI) is true — same as all other detectors; the detector
loop in main skips all detectors when bot is paused.

- If LastZ process is not running → start it and refresh window handle/PID.
- If process is running → run world_zoomout (same as event), sleep 2, click middle,
  sleep 2, check BuffIcon. If BuffIcon not found → assume disconnected: kill process
  and restart LastZ.
"""

import os
import time
import subprocess
import ctypes

import pyautogui

from vision import get_global_scale
from zoom_helpers import do_world_zoomout, do_base_zoomout


# ── Timing ────────────────────────────────────────────────────────────────────
WAIT_BEFORE_START_SEC: float = 10.0       # wait before starting LastZ after process dies
WAIT_AFTER_START_SEC: float = 10.0        # wait after starting LastZ for window to appear
RECENTLY_STARTED_GUARD_SEC: float = 60.0  # skip re-start if process was started within this many seconds
KILL_WAIT_SEC: float = 2.0                # wait after killing process before starting
FOCUS_SLEEP_SEC: float = 0.2             # sleep after focusing window before check
AFTER_ZOOMOUT_SLEEP_SEC: float = 2.0     # sleep after world_zoomout before clicking middle
AFTER_CLICK_SLEEP_SEC: float = 2.0       # sleep after clicking middle before screenshot

# ── BuffIcon check ────────────────────────────────────────────────────────────
BUFF_MISS_RECHECK_SEC: float = 300    # re-check delay (seconds) after first BuffIcon miss
BUFF_MISSES_BEFORE_RESTART: int = 5      # consecutive misses required before killing and restarting

# ── Vision thresholds ─────────────────────────────────────────────────────────
THRESHOLD_DEFAULT: float = 0.75          # default template-match threshold

# ── close_ui ──────────────────────────────────────────────────────────────────
CLOSE_UI_CLICK_X: float = 0.03          # relative X position of the dismiss-tap (fraction of window width)
CLOSE_UI_CLICK_Y: float = 0.08          # relative Y position of the dismiss-tap
CLOSE_UI_MAX_TRIES: int = 10            # max dismiss attempts before giving up
CLOSE_UI_CLICK_SLEEP_SEC: float = 1.0   # sleep after each dismiss tap
CLOSE_UI_FOCUS_SLEEP_SEC: float = 0.05  # sleep after refocusing window inside close_ui loop
CLOSE_UI_POST_CLICK_SLEEP_SEC: float = 0.3  # sleep after refreshing screenshot inside close_ui loop

# ── world_zoomout ─────────────────────────────────────────────────────────────
WORLD_ZOOMOUT_SCROLL_TIMES: int = 0
WORLD_ZOOMOUT_SCROLL_INTERVAL_SEC: float = 0.1
WORLD_ZOOMOUT_ROI_CENTER_X: float = 0.93
WORLD_ZOOMOUT_ROI_CENTER_Y: float = 0.96
WORLD_ZOOMOUT_ROI_PADDING: int = 2

# ── base_zoomout (run after connection OK) ────────────────────────────────────
BASE_ZOOMOUT_SCROLL_TIMES: int = 5
BASE_ZOOMOUT_SCROLL_INTERVAL_SEC: float = 0.1

# ── taskkill ──────────────────────────────────────────────────────────────────
TASKKILL_TIMEOUT_SEC: int = 10           # timeout (seconds) for the taskkill command


# Windows: check if process exists by opening handle
def _is_process_running(pid):
    if pid is None or pid <= 0:
        return False
    try:
        # PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if h:
            ctypes.windll.kernel32.CloseHandle(h)
            return True
    except Exception:
        pass
    return False


def _kill_process(pid, log):
    """Terminate process by PID. On Windows uses taskkill."""
    if pid is None or pid <= 0:
        return
    try:
        subprocess.run(
            ["taskkill", "/F", "/PID", str(pid)],
            capture_output=True,
            timeout=TASKKILL_TIMEOUT_SEC,
        )
        log.info("[ConnectionDetector] Killed LastZ process PID=%s", pid)
    except Exception as e:
        log.warning("[ConnectionDetector] taskkill failed: %s", e)


def _start_lastz(exe_path, log):
    """Start LastZ process. Returns True on success."""
    if not exe_path or not os.path.isfile(exe_path):
        log.error("[ConnectionDetector] LastZ exe not found: %s", exe_path)
        return False
    try:
        cwd = os.path.dirname(exe_path)
        subprocess.Popen(
            [exe_path],
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info("[ConnectionDetector] Started LastZ: %s", exe_path)
        return True
    except Exception as e:
        log.error("[ConnectionDetector] Failed to start LastZ: %s", e)
        return False


def _find_window_and_pid(window_name):
    """Find window by title; return (hwnd, pid) or (None, None)."""
    try:
        import win32gui
        import win32process
        hwnd = win32gui.FindWindow(None, window_name)
        if not hwnd or not win32gui.IsWindow(hwnd):
            return None, None
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        return hwnd, pid
    except Exception:
        return None, None


def _do_close_ui(wincap, vision_cache, template_path, world_button_path, log,
                 threshold=THRESHOLD_DEFAULT, click_x=CLOSE_UI_CLICK_X,
                 click_y=CLOSE_UI_CLICK_Y, max_tries=CLOSE_UI_MAX_TRIES):
    """
    Close overlays until HQ or World is visible (same logic as close_ui step).
    Loop: if template or world_button found in screenshot, return True; else click at (click_x, click_y), sleep, refresh, retry.
    """
    vision = vision_cache.get(template_path) if template_path else None
    vision_world = vision_cache.get(world_button_path) if world_button_path else None
    if not vision and not vision_world:
        log.warning("[ConnectionDetector] close_ui: no template in cache")
        return False
    scr = wincap.get_screenshot()
    if scr is None:
        return False
    for _try in range(max_tries):
        phq = vision.find(scr, threshold=threshold) if vision else []
        pw = vision_world.find(scr, threshold=threshold) if vision_world else []
        if (phq and len(phq) > 0) or (pw and len(pw) > 0):
            log.info("[ConnectionDetector] close_ui → true (HQ/World visible after %d click(s))", _try)
            return True
        if _try < max_tries - 1:
            if hasattr(wincap, "focus_window"):
                wincap.focus_window(force=True)
                time.sleep(CLOSE_UI_FOCUS_SLEEP_SEC)
            px = int(wincap.w * click_x)
            py = int(wincap.h * click_y)
            sx, sy = wincap.get_screen_position((px, py))
            pyautogui.click(sx, sy)
            time.sleep(CLOSE_UI_CLICK_SLEEP_SEC)
            fresh = wincap.get_screenshot()
            if fresh is not None:
                scr = fresh
            time.sleep(CLOSE_UI_POST_CLICK_SLEEP_SEC)
    log.warning("[ConnectionDetector] close_ui → false (max_tries=%d)", max_tries)
    return False


class ConnectionDetector:
    """
    Every `interval_sec` seconds (e.g. 300 = 5 min):
    - If LastZ process is not running → start exe and refresh hwnd/pid.
    - Else: world_zoomout, sleep 2, click middle, sleep 2, check BuffIcon.
      If BuffIcon not found → kill process and restart LastZ.
    """

    def __init__(
        self,
        buff_icon_template_path: str,
        world_zoomout_template_path: str,
        world_zoomout_button_path: str,
        lastz_exe_path: str,
        lastz_pid_ref: dict,
        lastz_window_name: str = "LastZ",
        interval_sec: float = 300.0,
        buff_icon_threshold: float = 0.75,
    ):
        self._buff_icon_path = buff_icon_template_path
        self._world_template = world_zoomout_template_path
        self._world_button = world_zoomout_button_path
        self._lastz_exe = lastz_exe_path
        self._lastz_pid_ref = lastz_pid_ref  # mutable: {"pid": int}
        self._window_name = lastz_window_name
        self._interval_sec = interval_sec
        self._buff_threshold = buff_icon_threshold
        self._last_run_time = time.time()  # first check after interval_sec from startup
        self._last_start_time: float = 0.0  # timestamp of last _start_lastz call
        self._buff_miss_count: int = 0  # consecutive BuffIcon-not-found count; restart only at 2

    def reset(self):
        """On resume: next full check in interval_sec (e.g. 5 min), not immediately."""
        self._last_run_time = time.time()

    def should_run(self, now_ts):
        """Return True if interval has elapsed since last run."""
        if self._last_run_time <= 0:
            return True
        return (now_ts - self._last_run_time) >= self._interval_sec

    def update(self, wincap, vision_cache, log, current_screenshot=None, is_busy=None):
        """
        Call every detector tick. Process check runs every tick (restart LastZ soon after window closed).
        Full connection check (close_ui, world_zoomout, BuffIcon) runs every interval_sec (e.g. 5 min).
        When current_screenshot is provided and PasswordSlot is visible (login screen), skip full check.

        Args:
            is_busy: optional callable returning True while a bot function is actively running.
                     Only the full connection check is deferred when busy; process check and
                     window reattach always run so LastZ is recovered even during a bot function.
        """
        now = time.time()
        busy = is_busy and is_busy()

        pid = self._lastz_pid_ref.get("pid")

        # 1) Every tick: if process not running → wait 10s then start and refresh hwnd/pid (so closing window triggers restart soon)
        if not _is_process_running(pid):
            # Guard: if we started the process recently, just try to reattach the window
            # instead of starting again (prevents double-start when window hasn't appeared yet)
            elapsed_since_start = now - self._last_start_time
            if self._last_start_time > 0 and elapsed_since_start < RECENTLY_STARTED_GUARD_SEC:
                log.debug(
                    "[ConnectionDetector] Process recently started (%.0fs ago), waiting for window...",
                    elapsed_since_start,
                )
                hwnd, new_pid = _find_window_and_pid(self._window_name)
                if hwnd is not None:
                    wincap.hwnd = hwnd
                    wincap.refresh_geometry()
                    self._lastz_pid_ref["pid"] = new_pid
                    log.info("[ConnectionDetector] Window reattached, PID=%s", new_pid)
                return

            log.info("[ConnectionDetector] LastZ process not running (PID=%s), waiting %.0fs then starting...", pid, WAIT_BEFORE_START_SEC)
            time.sleep(WAIT_BEFORE_START_SEC)
            self._last_start_time = now
            if _start_lastz(self._lastz_exe, log):
                time.sleep(WAIT_AFTER_START_SEC)
            hwnd, new_pid = _find_window_and_pid(self._window_name)
            if hwnd is not None:
                wincap.hwnd = hwnd
                wincap.refresh_geometry()
                self._lastz_pid_ref["pid"] = new_pid
                log.info("[ConnectionDetector] Window reattached, PID=%s", new_pid)
            return

        # 1.5) Process running but caller had no screenshot (e.g. detector screenshot failed) — try to re-find window in case hwnd is stale
        if current_screenshot is None:
            hwnd, new_pid = _find_window_and_pid(self._window_name)
            if hwnd is not None:
                try:
                    import win32gui
                    if not win32gui.IsWindow(getattr(wincap, "hwnd", None)) or wincap.hwnd != hwnd:
                        wincap.hwnd = hwnd
                        wincap.refresh_geometry()
                        self._lastz_pid_ref["pid"] = new_pid
                        log.info("[ConnectionDetector] Window reattached (hwnd was invalid), PID=%s", new_pid)
                except Exception:
                    pass
            return  # no screenshot available, skip full check

        # Guard: skip full connection check while any bot function is actively running
        if busy:
            log.debug("[ConnectionDetector] Skipped — bot function is running")
            self._last_run_time = now  # reset interval so check runs after interval_sec of idle time
            return

        # 2) Full connection check only every interval_sec (e.g. 5 min)
        if not self.should_run(now):
            return
        # Skip connection check when on login screen (PasswordSlot visible)
        if current_screenshot is not None:
            try:
                _pv = vision_cache.get("buttons_template/PasswordSlot.png")
                if _pv and _pv.exists(current_screenshot, threshold=THRESHOLD_DEFAULT):
                    log.info("[ConnectionDetector] On login screen, skipping check")
                    self._last_run_time = now
                    return
            except Exception:
                pass
        self._last_run_time = now

        # 3) Process is running: run connection check (close_ui → world_zoomout → click → BuffIcon)
        if hasattr(wincap, "focus_window"):
            wincap.focus_window(force=True)
            time.sleep(FOCUS_SLEEP_SEC)
        _do_close_ui(
            wincap, vision_cache,
            self._world_template, self._world_button, log,
        )
        if not do_world_zoomout(
            wincap, vision_cache, log,
            self._world_template, self._world_button,
            screenshot=None,
            threshold=THRESHOLD_DEFAULT,
            scroll_times=WORLD_ZOOMOUT_SCROLL_TIMES,
            scroll_interval_sec=WORLD_ZOOMOUT_SCROLL_INTERVAL_SEC,
            roi_center_x=WORLD_ZOOMOUT_ROI_CENTER_X,
            roi_center_y=WORLD_ZOOMOUT_ROI_CENTER_Y,
            roi_padding=WORLD_ZOOMOUT_ROI_PADDING,
            log_prefix="[ConnectionDetector] ",
        ):
            log.warning("[ConnectionDetector] world_zoomout failed (not on world?), continuing check")
        time.sleep(AFTER_ZOOMOUT_SLEEP_SEC)
        # Click middle of screen
        cx = wincap.offset_x + wincap.w // 2
        cy = wincap.offset_y + wincap.h // 2
        pyautogui.click(cx, cy)
        time.sleep(AFTER_CLICK_SLEEP_SEC)
        screenshot = wincap.get_screenshot()
        if screenshot is None:
            log.warning("[ConnectionDetector] Screenshot failed after click")
            return
        buff_vision = vision_cache.get(self._buff_icon_path)
        if not buff_vision:
            log.warning("[ConnectionDetector] BuffIcon template not in cache: %s", self._buff_icon_path)
            return
        if buff_vision.exists(screenshot, threshold=self._buff_threshold):
            log.info("[ConnectionDetector] BuffIcon found → connection OK")
            self._buff_miss_count = 0
            # Run base_zoomout after connection OK (zoom out to world view)
            do_base_zoomout(
                wincap, vision_cache, log,
                self._world_template, self._world_button,
                screenshot=None,
                threshold=THRESHOLD_DEFAULT,
                scroll_times=BASE_ZOOMOUT_SCROLL_TIMES,
                scroll_interval_sec=BASE_ZOOMOUT_SCROLL_INTERVAL_SEC,
                log_prefix="[ConnectionDetector] ",
            )
            return

        # BuffIcon not found — require BUFF_MISSES_BEFORE_RESTART consecutive misses before restarting
        self._buff_miss_count += 1
        if self._buff_miss_count < BUFF_MISSES_BEFORE_RESTART:
            log.warning(
                "[ConnectionDetector] BuffIcon not found (miss #%d/%d), re-checking in %.0fs...",
                self._buff_miss_count, BUFF_MISSES_BEFORE_RESTART, BUFF_MISS_RECHECK_SEC,
            )
            self._last_run_time = now - self._interval_sec + BUFF_MISS_RECHECK_SEC  # schedule re-check
            return

        # Two consecutive misses → assume disconnected, kill and restart
        self._buff_miss_count = 0
        log.warning("[ConnectionDetector] BuffIcon not found 2 times in a row → disconnection, killing and restarting LastZ")
        _kill_process(pid, log)
        time.sleep(KILL_WAIT_SEC)
        self._last_start_time = time.time()
        if _start_lastz(self._lastz_exe, log):
            time.sleep(WAIT_AFTER_START_SEC)
        hwnd, new_pid = _find_window_and_pid(self._window_name)
        if hwnd is not None:
            wincap.hwnd = hwnd
            wincap.refresh_geometry()
            self._lastz_pid_ref["pid"] = new_pid
            log.info("[ConnectionDetector] LastZ restarted, window reattached, PID=%s", new_pid)
