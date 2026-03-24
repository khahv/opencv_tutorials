"""
screenshot_provider.py
----------------------
Abstracts screenshot capture for different emulator modes.

  Win32ScreenshotProvider  — captures via win32gui + mss (default, PC / native window)
  AdbScreenshotProvider    — captures via LDPlayer ADB  (adb exec-out screencap -p)

  ScreenshotCaptureService — single pipeline: only capture_frame() performs a grab;
                             get_cached() returns the last frame for detectors and
                             wincap.get_screenshot() (no duplicate screencap).

Usage:
    service = create_screenshot_capture_service("pc", wincap=wincap)
    img = service.capture_frame()   # main loop / explicit refresh only
    cached = service.get_cached()   # readers (detectors, events use tick screenshot when possible)
"""

import logging
import subprocess
import threading
from typing import Any, Optional

import numpy as np

log = logging.getLogger("kha_lastz")

# Set by main / game_surface after connect; cleared on disconnect.
_active_capture_service: Optional["ScreenshotCaptureService"] = None


def set_active_capture_service(service: Optional["ScreenshotCaptureService"]) -> None:
    """Install the single capture service (PC or LDPlayer). None on disconnect."""
    global _active_capture_service
    _active_capture_service = service


def get_active_capture_service() -> Optional["ScreenshotCaptureService"]:
    """Return the active ScreenshotCaptureService, or None if not connected."""
    return _active_capture_service


class Win32ScreenshotProvider:
    """
    Captures screenshots from a Windows game window using win32gui + mss.

    Thread-safe: maintains a per-thread mss instance via threading.local so this
    provider can be called simultaneously from multiple threads (e.g. game loop +
    detector thread).
    """

    def __init__(self, wincap) -> None:
        self._wincap = wincap
        self._mss_local = threading.local()

    def get_screenshot(self) -> Optional[np.ndarray]:
        """
        Return a BGR ndarray of the game window client area, or None if the window
        is invalid or unavailable.
        """
        import mss
        import win32gui

        wincap = self._wincap
        if not wincap or not wincap.hwnd:
            return None

        try:
            wincap.refresh_geometry()
        except Exception:
            pass

        if wincap.w <= 0 or wincap.h <= 0:
            return None

        if not hasattr(self._mss_local, "mss") or self._mss_local.mss is None:
            self._mss_local.mss = mss.mss()
        _mss = self._mss_local.mss

        left, top = win32gui.ClientToScreen(wincap.hwnd, (0, 0))
        monitor = {"left": left, "top": top, "width": wincap.w, "height": wincap.h}
        raw = _mss.grab(monitor)
        img = np.array(raw)[..., :3]
        return np.ascontiguousarray(img)


class AdbScreenshotProvider:
    """
    Captures screenshots via ADB (for LDPlayer emulator mode).

    Runs: adb exec-out screencap -p  → raw PNG bytes → BGR ndarray.
    The default adb path targets LDPlayer 9's bundled adb.exe.
    """

    _DEFAULT_ADB_PATH = r"C:\LDPlayer\LDPlayer9\adb.exe"
    _ADB_TIMEOUT_SEC = 8

    def __init__(
        self,
        adb_path: Optional[str] = None,
        device_serial: Optional[str] = None,
    ) -> None:
        self._adb_path = adb_path or self._DEFAULT_ADB_PATH
        self._device_serial = device_serial

    def get_screenshot(self) -> Optional[np.ndarray]:
        """
        Return a BGR ndarray captured via ADB screencap, or None on failure.
        """
        try:
            cmd = [self._adb_path]
            if self._device_serial:
                cmd += ["-s", self._device_serial]
            cmd += ["exec-out", "screencap", "-p"]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=self._ADB_TIMEOUT_SEC,
            )
            if result.returncode != 0 or not result.stdout:
                log.warning(
                    "[ADB] screencap failed (rc=%d, stderr=%s)",
                    result.returncode,
                    result.stderr[:200] if result.stderr else "",
                )
                return None

            buf = np.frombuffer(result.stdout, dtype=np.uint8)
            import cv2
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                log.warning("[ADB] screencap: cv2.imdecode returned None (bad PNG data?)")
            return img

        except subprocess.TimeoutExpired:
            log.warning("[ADB] screencap timed out after %ds", self._ADB_TIMEOUT_SEC)
            return None
        except FileNotFoundError:
            log.error(
                "[ADB] adb not found at '%s'. "
                "Install ADB or set ldplayer_adb_path in config.yaml.",
                self._adb_path,
            )
            return None
        except Exception as exc:
            log.warning("[ADB] screencap error: %s", exc)
            return None


class ScreenshotCaptureService:
    """
    Single capture pipeline for both PC (Win32) and LDPlayer (ADB).

    Only capture_frame() calls the backend grab. All other code should use
    get_cached() or the per-tick ``screenshot`` passed into runner.update.
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self._lock = threading.Lock()
        self._last: Optional[np.ndarray] = None

    def capture_frame(self) -> Optional[np.ndarray]:
        """Perform one real screen grab and store it. Main game loop and reconnect use this."""
        with self._lock:
            img = self._backend.get_screenshot()
            self._last = img
            return img

    def get_cached(self) -> Optional[np.ndarray]:
        """Last frame from capture_frame(); no grab. Safe for background detector thread."""
        with self._lock:
            return self._last


def create_screenshot_capture_service(
    emulator: str,
    wincap=None,
    adb_path: Optional[str] = None,
    device_serial: Optional[str] = None,
) -> ScreenshotCaptureService:
    """Build backend + ScreenshotCaptureService (PC or LDPlayer)."""
    backend = create_screenshot_provider(
        emulator, wincap=wincap, adb_path=adb_path, device_serial=device_serial
    )
    return ScreenshotCaptureService(backend)


def create_screenshot_provider(
    emulator: str,
    wincap=None,
    adb_path: Optional[str] = None,
    device_serial: Optional[str] = None,
):
    """
    Factory — return the appropriate ScreenshotProvider for the chosen emulator.

    Args:
        emulator:      "pc" (default) or "ldplayer"
        wincap:        WindowCapture instance (required for "pc" mode)
        adb_path:      Path to adb.exe (for "ldplayer"; uses LDPlayer default if None)
        device_serial: ADB device serial (for "ldplayer", optional)

    Returns:
        Win32ScreenshotProvider or AdbScreenshotProvider
    """
    if emulator == "ldplayer":
        return AdbScreenshotProvider(adb_path=adb_path, device_serial=device_serial)
    return Win32ScreenshotProvider(wincap=wincap)
