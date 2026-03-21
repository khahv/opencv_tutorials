"""
adb_input.py
------------
ADB-based input (tap / swipe) for LDPlayer emulator mode.

Coordinates are always in **device (client) pixels** — the same coordinate
space as the game screenshot — NOT Windows screen pixels.

Commands issued:
    tap:   adb shell input tap   <x> <y>
    swipe: adb shell input swipe <x1> <y1> <x2> <y2> [duration_ms]

Touch coordinates for ``[MOUSE-LOG]`` (LDPlayer ROI tuning) come from
``adb shell getevent -lt`` only — no Win32 / pynput on the host.

Auto-detect flow (called by main.py on startup):
    1. Run ``adb devices`` — use the first online device found.
    2. If none, try ``adb connect 127.0.0.1:{port}`` for LDPlayer's known
       default ports (5555, 5557, 5559 … one per emulator instance).
    3. If a manual ``ldplayer_device_serial`` is set in config.yaml,
       skip auto-detect entirely and use that serial directly.

Usage (main.py sets up the singleton; event handlers read it):

    # main.py
    import adb_input
    inst = adb_input.AdbInput(adb_path=...)
    inst.detect_and_connect()          # auto-detect LDPlayer
    adb_input.set_adb_input(inst)

    # event handler
    import adb_input
    _adb = adb_input.get_adb_input()
    if _adb is not None:
        _adb.tap(cx, cy)
    else:
        # fall back to win32 / pyautogui
"""

import logging
import re
import subprocess
import threading
import time
from typing import List, Optional, Tuple

log = logging.getLogger("kha_lastz")

_DEFAULT_ADB_PATH = r"C:\LDPlayer\LDPlayer9\adb.exe"
_ADB_TIMEOUT_SEC = 5

# LDPlayer assigns one TCP port per emulator instance starting at 5555,
# incrementing by 2 (5555, 5557, 5559 …).  We probe the first 8 slots.
_LDPLAYER_ADB_PORTS: List[int] = [5555, 5557, 5559, 5561, 5563, 5565, 5567, 5569]

# Parsed from ``adb shell getevent -lt`` (multitouch or single-touch ABS).
_RE_GETEVENT_MT_X = re.compile(r"ABS_MT_POSITION_X\s+([0-9a-fA-F]+)")
_RE_GETEVENT_MT_Y = re.compile(r"ABS_MT_POSITION_Y\s+([0-9a-fA-F]+)")
_RE_GETEVENT_ABS_X = re.compile(r"\bABS_X\s+([0-9a-fA-F]+)")
_RE_GETEVENT_ABS_Y = re.compile(r"\bABS_Y\s+([0-9a-fA-F]+)")


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _run_adb_text(adb_path: str, args: List[str], timeout: int = 5) -> Optional[str]:
    """Run an adb command and return decoded stdout, or None on any error."""
    try:
        result = subprocess.run(
            [adb_path] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout
    except Exception:
        return None


def _parse_adb_devices(output: str) -> List[str]:
    """
    Parse ``adb devices`` output and return serials of online (ready) devices.

    Skips devices that are ``offline``, ``unauthorized``, or ``no permissions``.
    """
    serials: List[str] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("List of") or line.startswith("*"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].strip() == "device":
            serials.append(parts[0].strip())
    return serials


# ── Public auto-detect function ────────────────────────────────────────────────

def detect_ldplayer_device(adb_path: Optional[str] = None) -> Optional[str]:
    """
    Auto-detect a running LDPlayer ADB device and return its serial.

    Detection steps
    ---------------
    1. ``adb devices`` — return the first online serial immediately.
    2. If none found, iterate through ``_LDPLAYER_ADB_PORTS`` and try
       ``adb connect 127.0.0.1:{port}``.  Return the first successful address.

    Returns
    -------
    str  Device serial such as ``"emulator-5554"`` or ``"127.0.0.1:5555"``,
         or ``None`` when nothing is found.
    """
    path = adb_path or _DEFAULT_ADB_PATH

    # ── Step 1: already-connected devices ─────────────────────────────────────
    out = _run_adb_text(path, ["devices"])
    if out:
        serials = _parse_adb_devices(out)
        if serials:
            log.info(
                "[ADB] Auto-detect: found connected device(s): %s — using '%s'",
                serials, serials[0],
            )
            return serials[0]

    # ── Step 2: probe LDPlayer TCP ports ──────────────────────────────────────
    log.info(
        "[ADB] Auto-detect: no device in 'adb devices', "
        "probing LDPlayer ports %s …",
        _LDPLAYER_ADB_PORTS,
    )
    for port in _LDPLAYER_ADB_PORTS:
        addr = "127.0.0.1:{}".format(port)
        out = _run_adb_text(path, ["connect", addr], timeout=3)
        if out and ("connected" in out.lower() or "already connected" in out.lower()):
            log.info("[ADB] Auto-detect: connected to LDPlayer at %s", addr)
            return addr
        log.debug("[ADB] Auto-detect: port %d → %s", port, (out or "").strip())

    log.warning("[ADB] Auto-detect: no LDPlayer device found on any known port.")
    return None


def parse_wm_size_output(stdout: str) -> Optional[Tuple[int, int]]:
    """
    Parse ``adb shell wm size`` text and return (width, height) in device pixels.

    Prefer ``Override size`` when present (logical resolution used by input and
    screencap); otherwise use ``Physical size``.
    """
    if not stdout:
        return None
    override_wh: Optional[Tuple[int, int]] = None
    physical_wh: Optional[Tuple[int, int]] = None
    for line in stdout.splitlines():
        line_l = line.strip()
        m = re.search(r"(\d+)\s*x\s*(\d+)", line_l)
        if not m:
            continue
        w, h = int(m.group(1)), int(m.group(2))
        if "override" in line_l.lower():
            override_wh = (w, h)
        elif "physical" in line_l.lower():
            physical_wh = (w, h)
    if override_wh:
        return override_wh
    if physical_wh:
        return physical_wh
    m = re.search(r"(\d+)\s*x\s*(\d+)", stdout)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def map_ldplayer_client_to_device(
    client_x: float,
    client_y: float,
    client_w: int,
    client_h: int,
    device_w: int,
    device_h: int,
) -> Optional[Tuple[int, int]]:
    """
    Map LDPlayer **Win32 client** coordinates to ADB device pixels.

    The emulator draws the Android framebuffer with uniform scaling inside the
    client; extra space from aspect mismatch is assumed at the **top** (toolbar /
    letterbox), matching typical LDPlayer layout — viewport is bottom-aligned.

    Returns
    -------
    (dx, dy) clamped to the device rectangle, or ``None`` if the point lies
    outside the emulated viewport (e.g. title toolbar strip).
    """
    if client_w <= 0 or client_h <= 0 or device_w <= 0 or device_h <= 0:
        return None
    scale = min(client_w / device_w, client_h / device_h)
    vw = device_w * scale
    vh = device_h * scale
    off_x = (client_w - vw) * 0.5
    off_y = client_h - vh
    if not (off_x <= client_x < off_x + vw and off_y <= client_y < off_y + vh):
        return None
    nx = client_x - off_x
    ny = client_y - off_y
    dx = int(round(nx / scale))
    dy = int(round(ny / scale))
    dx = max(0, min(device_w - 1, dx))
    dy = max(0, min(device_h - 1, dy))
    return dx, dy


# ── AdbInput class ─────────────────────────────────────────────────────────────

class AdbInput:
    """
    Sends touch-input commands to a connected ADB device (LDPlayer emulator).

    All coordinates are in device (client) pixels — the same space as
    OpenCV screenshots, not Windows screen coordinates.
    """

    def __init__(
        self,
        adb_path: Optional[str] = None,
        device_serial: Optional[str] = None,
    ) -> None:
        self._adb_path = adb_path or _DEFAULT_ADB_PATH
        self._device_serial = device_serial
        self._wm_size_cache: Optional[Tuple[int, int, float]] = None
        self._wm_size_cache_ttl_sec = 10.0
        self._getevent_log_thread: Optional[threading.Thread] = None
        self._getevent_proc: Optional[subprocess.Popen] = None
        self._mouse_log_debounce_xy: Optional[Tuple[int, int]] = None
        self._mouse_log_debounce_t: float = 0.0

    def _shell_text(self, *shell_args: str) -> Optional[str]:
        """Run ``adb shell ...`` and return stdout text, or None on failure."""
        args: List[str] = []
        if self._device_serial:
            args += ["-s", self._device_serial]
        args += ["shell"] + list(shell_args)
        return _run_adb_text(self._adb_path, args)

    def get_device_screen_size(self, refresh: bool = False) -> Optional[Tuple[int, int]]:
        """
        Return ``(width, height)`` from ``wm size`` (same space as screencap / input tap).

        Results are cached briefly to avoid spawning adb on every mouse click.
        """
        now = time.monotonic()
        if (
            not refresh
            and self._wm_size_cache is not None
            and now - self._wm_size_cache[2] < self._wm_size_cache_ttl_sec
        ):
            return self._wm_size_cache[0], self._wm_size_cache[1]
        out = self._shell_text("wm", "size")
        parsed = parse_wm_size_output(out or "")
        if parsed:
            w, h = parsed
            self._wm_size_cache = (w, h, now)
            return w, h
        return None

    # ── Device detection ───────────────────────────────────────────────────────

    def detect_and_connect(self) -> Optional[str]:
        """
        Auto-detect a running LDPlayer instance and store its serial.

        Calls :func:`detect_ldplayer_device` and, on success, updates
        ``self._device_serial`` so all subsequent ``tap`` / ``swipe`` calls
        target the detected device.

        Returns the serial string, or ``None`` if nothing was found.
        """
        serial = detect_ldplayer_device(self._adb_path)
        if serial:
            self._device_serial = serial
            log.info("[ADB Input] Device serial: %s", serial)
        else:
            log.warning(
                "[ADB Input] No LDPlayer device detected.  "
                "Set ldplayer_device_serial in config.yaml to specify manually."
            )
        return serial

    # ── Touch → MOUSE-LOG (getevent, no host Win32) ─────────────────────────────

    def start_getevent_mouse_log(self) -> None:
        """
        Background thread: ``adb shell getevent -lt`` → print ``[MOUSE-LOG]`` on each
        touch report using **device** coordinates (same space as screencap / ``input tap``).
        """
        if self._getevent_log_thread is not None and self._getevent_log_thread.is_alive():
            return
        self._getevent_log_thread = threading.Thread(
            target=self._getevent_mouse_log_loop,
            name="adb-getevent-mouselog",
            daemon=True,
        )
        self._getevent_log_thread.start()
        log.info("[ADB] MOUSE-LOG: listening via `getevent -lt` (device touch coordinates).")

    def _emit_mouse_log_line(self, dx: int, dy: int) -> None:
        """Debounce and print the same block as PC ``WindowCapture`` mouse debug."""
        now = time.monotonic()
        if self._mouse_log_debounce_xy == (dx, dy) and now - self._mouse_log_debounce_t < 0.08:
            return
        self._mouse_log_debounce_xy = (dx, dy)
        self._mouse_log_debounce_t = now

        dev = self.get_device_screen_size()
        if not dev:
            return
        dw, dh = dev
        if dw <= 0 or dh <= 0:
            return
        rel_x = dx / dw
        rel_y = dy / dh
        if not (0 <= rel_x <= 1 and 0 <= rel_y <= 1):
            return

        print(f"\n[MOUSE-LOG] Inside Game Window:")
        print(f"  - Local Pixel: ({dx}, {dy})")
        print(f"  - Ratio:       x={rel_x:.4f}, y={rel_y:.4f}")
        print(f"  - YAML ROI (copy this):")
        print(f"    roi_center_x: {rel_x:.2f}")
        print(f"    roi_center_y: {rel_y:.2f}")

    def _getevent_mouse_log_loop(self) -> None:
        cmd = [self._adb_path]
        if self._device_serial:
            cmd += ["-s", self._device_serial]
        cmd += ["shell", "getevent", "-lt"]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            log.warning("[ADB] MOUSE-LOG: could not spawn getevent: %s", exc)
            return

        self._getevent_proc = proc
        mt_x: Optional[int] = None
        mt_y: Optional[int] = None
        abs_x: Optional[int] = None
        abs_y: Optional[int] = None

        try:
            if proc.stdout is None:
                return
            for raw in proc.stdout:
                line = raw.strip()
                m = _RE_GETEVENT_MT_X.search(line)
                if m:
                    mt_x = int(m.group(1), 16)
                m = _RE_GETEVENT_MT_Y.search(line)
                if m:
                    mt_y = int(m.group(1), 16)
                m = _RE_GETEVENT_ABS_X.search(line)
                if m:
                    abs_x = int(m.group(1), 16)
                m = _RE_GETEVENT_ABS_Y.search(line)
                if m:
                    abs_y = int(m.group(1), 16)

                if "SYN_REPORT" not in line:
                    continue

                dx: Optional[int] = None
                dy: Optional[int] = None
                if mt_x is not None and mt_y is not None:
                    dx, dy = mt_x, mt_y
                elif abs_x is not None and abs_y is not None:
                    dx, dy = abs_x, abs_y

                mt_x = mt_y = None
                abs_x = abs_y = None

                if dx is not None and dy is not None:
                    self._emit_mouse_log_line(dx, dy)
        except Exception as exc:
            log.debug("[ADB] getevent MOUSE-LOG loop exited: %s", exc)
        finally:
            try:
                proc.kill()
            except Exception:
                pass
            self._getevent_proc = None

    # ── Internal command runner ────────────────────────────────────────────────

    def _run(self, args: List[str]) -> bool:
        """Execute an adb command. Returns True on success."""
        cmd = [self._adb_path]
        if self._device_serial:
            cmd += ["-s", self._device_serial]
        cmd += args
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=_ADB_TIMEOUT_SEC
            )
            if result.returncode != 0:
                log.warning(
                    "[ADB] command failed (rc=%d): %s",
                    result.returncode,
                    " ".join(args),
                )
                return False
            return True
        except subprocess.TimeoutExpired:
            log.warning("[ADB] command timed out: %s", " ".join(args))
            return False
        except FileNotFoundError:
            log.error(
                "[ADB] adb not found at '%s'. "
                "Check ldplayer_adb_path in config.yaml or install ADB.",
                self._adb_path,
            )
            return False
        except Exception as exc:
            log.warning("[ADB] command error: %s", exc)
            return False

    # ── Input commands ─────────────────────────────────────────────────────────

    def tap(self, x: int, y: int) -> bool:
        """Tap at device coordinates (x, y)."""
        return self._run(["shell", "input", "tap", str(int(x)), str(int(y))])

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 800,
    ) -> bool:
        """
        Swipe from (x1, y1) to (x2, y2).

        Args:
            x1, y1:      Start position in device pixels.
            x2, y2:      End position in device pixels.
            duration_ms: Swipe duration in milliseconds (default 800 ms).
        """
        return self._run([
            "shell", "input", "swipe",
            str(int(x1)), str(int(y1)),
            str(int(x2)), str(int(y2)),
            str(int(duration_ms)),
        ])

    def wheel_zoom_out_approx(
        self,
        center_x: int,
        center_y: int,
        times: int = 5,
        interval_sec: float = 0.1,
        arm_px: int = 200,
        duration_ms: int = 150,
    ) -> None:
        """
        Approximate a mouse-wheel \"zoom out\" using short upward swipes from the center.

        Used when the game is driven only via ADB (no Win32 mouse wheel).
        """
        cx, cy = int(center_x), int(center_y)
        for _ in range(max(1, int(times))):
            y1 = cy + arm_px
            y2 = cy - arm_px
            self.swipe(cx, y1, cx, y2, duration_ms)
            time.sleep(max(0.0, float(interval_sec)))


# ── Module-level singleton ─────────────────────────────────────────────────────
# Set once by main.py after emulator mode is determined; None in PC mode.

_instance: Optional[AdbInput] = None


def get_adb_input() -> Optional[AdbInput]:
    """Return the active AdbInput instance, or None when in PC (win32) mode."""
    return _instance


def set_adb_input(inst: Optional[AdbInput]) -> None:
    """Set (or clear) the global AdbInput instance. Called once by main.py."""
    global _instance
    prev = _instance
    if prev is not None and prev is not inst:
        proc = getattr(prev, "_getevent_proc", None)
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
            prev._getevent_proc = None
        prev._getevent_log_thread = None
    _instance = inst
