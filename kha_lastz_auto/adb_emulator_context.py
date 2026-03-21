"""
ADB-only game surface context for LDPlayer mode.

Provides the same duck-typed fields used by bot steps (``w``, ``h``, ``offset_*``,
``get_screen_position``, ``get_screenshot``) in **device pixel** space — matching
``adb exec-out screencap`` and ``adb shell input tap``.

Optional ``[MOUSE-LOG]`` for ROI tuning uses **only ADB** (``getevent`` touch
coordinates on the device), not Win32 or pynput.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

log = logging.getLogger("kha_lastz")


class AdbEmulatorContext:
    """
    Bounds and helpers for the emulated Android display via ADB only.

    ``get_screen_position`` returns its argument unchanged (device pixels); legacy
    callers use that name for Win32 screen mapping on PC mode.
    """

    is_using_adb = True
    hwnd = None
    auto_focus = False
    offset_x = 0
    offset_y = 0
    cropped_x = 0
    cropped_y = 0

    def __init__(self, screenshot_provider: Any, enable_mouse_log: bool = True) -> None:
        self._provider = screenshot_provider
        self.w = 0
        self.h = 0
        if enable_mouse_log:
            try:
                import adb_input as adb_mod

                adb = adb_mod.get_adb_input()
                if adb is not None:
                    adb.start_getevent_mouse_log()
            except Exception as exc:
                log.warning("[AdbEmulatorContext] MOUSE-LOG (getevent) not started: %s", exc)

    def refresh_geometry(self) -> None:
        """Refresh ``w`` / ``h`` from ``wm size`` or the last screencap shape."""
        import adb_input as adb_mod

        adb = adb_mod.get_adb_input()
        if adb is not None:
            sz = adb.get_device_screen_size()
            if sz:
                self.w, self.h = int(sz[0]), int(sz[1])
                return
        img = self._provider.get_screenshot() if self._provider else None
        if img is not None and getattr(img, "shape", None) is not None and len(img.shape) >= 2:
            self.h, self.w = int(img.shape[0]), int(img.shape[1])

    def get_screen_position(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Pass through device (screenshot) pixel coordinates."""
        return (int(pos[0]), int(pos[1]))

    def get_screenshot(self):
        """Capture via the injected ADB screenshot provider."""
        return self._provider.get_screenshot() if self._provider else None

    def focus_window(self, force: bool = False) -> None:
        """No-op: there is no host window to focus in pure ADB mode."""

    def resize_to_client(self, target_w: int, target_h: int) -> bool:
        log.debug(
            "[AdbEmulatorContext] resize_to_client(%s,%s) ignored — set resolution in LDPlayer.",
            target_w,
            target_h,
        )
        return False
