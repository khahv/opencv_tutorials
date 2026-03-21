"""
Game surface attachment: LDPlayer via ADB vs PC native window via Win32.

Keeps ``main.py`` free of long if/else emulator branches. All capture + reconnect
logic for each mode lives here.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

log = logging.getLogger("kha_lastz")


def is_game_surface_valid(wincap: Any) -> bool:
    """True when the current surface can be used for screenshots / input."""
    try:
        if wincap is None:
            return False
        if getattr(wincap, "is_using_adb", False):
            return wincap.w > 0 and wincap.h > 0
        import win32gui as _wg

        return bool(wincap.hwnd and _wg.IsWindow(wincap.hwnd))
    except Exception:
        return False


def should_skip_focus_loop(wincap: Any) -> bool:
    """Pure ADB mode has no HWND to focus; PC uses ``focus_loop``."""
    return wincap is None or getattr(wincap, "is_using_adb", False)


def apply_post_connect_window_prefs(
    wincap: Any,
    *,
    auto_focus: bool,
    win_w: Optional[int],
    win_h: Optional[int],
    logger: logging.Logger,
) -> None:
    """Set auto_focus and optional client resize (PC only; ADB context ignores resize)."""
    if wincap is None:
        return
    wincap.auto_focus = auto_focus
    if getattr(wincap, "is_using_adb", False):
        return
    if win_w and win_h:
        wincap.resize_to_client(win_w, win_h)
        logger.info("Window resized to {}x{} (target)".format(wincap.w, wincap.h))
    else:
        logger.info(
            "window_width/height not set — keeping current window size {}x{}".format(
                wincap.w, wincap.h,
            )
        )


def sync_lastz_pid_from_wincap(
    wincap: Any,
    lastz_pid: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Populate ``lastz_pid['pid']`` from the game window process (PC only)."""
    lastz_pid["pid"] = None
    if wincap is None or not getattr(wincap, "hwnd", None):
        return
    try:
        import win32process

        _, pid = win32process.GetWindowThreadProcessId(wincap.hwnd)
        lastz_pid["pid"] = pid
        logger.info("LastZ process PID: %s", pid)
    except Exception as exc:
        logger.warning("Could not get LastZ process PID: %s", exc)


class LdplayerAdbSurface:
    """Connect screenshot + emulator context through ADB only."""

    @staticmethod
    def connect(
        *,
        adb_path: Optional[str],
        device_serial: Optional[str],
        logger: logging.Logger,
    ) -> Tuple[Any, Any]:
        """
        Register global ``AdbInput``, build ``AdbScreenshotProvider`` + ``AdbEmulatorContext``.

        Returns ``(wincap, screenshot_provider)`` (wincap is always created; screencap may fail later).
        """
        from adb_emulator_context import AdbEmulatorContext
        from adb_input import AdbInput, set_adb_input
        from screenshot_provider import create_screenshot_provider

        logger.info("LDPlayer mode: game surface via ADB only (no Win32 window capture).")
        adb_inst = AdbInput(adb_path=adb_path, device_serial=device_serial)
        if device_serial:
            logger.info("[ADB Input] Using manual device serial: %s", device_serial)
        else:
            logger.info("[ADB Input] Auto-detecting LDPlayer device...")
            adb_inst.detect_and_connect()
        set_adb_input(adb_inst)
        logger.info(
            "[ADB Input] Ready — device=%s  adb=%s",
            adb_inst._device_serial or "none (commands will target default device)",
            adb_path or "default",
        )
        screenshot_provider = create_screenshot_provider(
            "ldplayer",
            wincap=None,
            adb_path=adb_path,
            device_serial=adb_inst._device_serial,
        )
        wincap = AdbEmulatorContext(screenshot_provider)
        wincap.refresh_geometry()
        try:
            probe = screenshot_provider.get_screenshot()
        except Exception as exc:
            logger.warning("[ADB] Initial screencap failed: %s", exc)
            probe = None
        if probe is not None:
            logger.info(
                "[ADB] Emulator context %dx%d (screencap OK).",
                wincap.w,
                wincap.h,
            )
        else:
            logger.warning(
                "[ADB] Initial screencap failed — UI may show disconnected until ADB responds.",
            )
        logger.info(
            "[Screenshot] Provider: %s (emulator=ldplayer)",
            type(screenshot_provider).__name__,
        )
        return wincap, screenshot_provider

    @staticmethod
    def try_reconnect_pair(
        *,
        adb_path: Optional[str],
        device_serial: Optional[str],
        logger: logging.Logger,
    ) -> Optional[Tuple[Any, Any]]:
        """
        One reconnect attempt. Returns ``(wincap, screenshot_provider)`` or ``None``.
        """
        from adb_emulator_context import AdbEmulatorContext
        from adb_input import AdbInput, set_adb_input
        from screenshot_provider import create_screenshot_provider

        try:
            adb_w = AdbInput(adb_path=adb_path, device_serial=device_serial)
            if not device_serial:
                adb_w.detect_and_connect()
            set_adb_input(adb_w)
            new_sp = create_screenshot_provider(
                "ldplayer",
                wincap=None,
                adb_path=adb_path,
                device_serial=adb_w._device_serial,
            )
            if new_sp.get_screenshot() is None:
                return None
            new_wincap = AdbEmulatorContext(new_sp)
            new_wincap.refresh_geometry()
            logger.info("[WindowWatcher] ADB emulator connected (%dx%d).", new_wincap.w, new_wincap.h)
            return new_wincap, new_sp
        except Exception:
            return None

    @staticmethod
    def watcher_step(
        wincap: Any,
        screenshot_provider: Any,
        *,
        adb_path: Optional[str],
        device_serial: Optional[str],
        logger: logging.Logger,
        update_vision_scale: Callable[[], None],
    ) -> Tuple[Any, Any]:
        """One 3s-tick: keep pair if valid, else try ADB reconnect."""
        if wincap is not None and is_game_surface_valid(wincap):
            return wincap, screenshot_provider
        pair = LdplayerAdbSurface.try_reconnect_pair(
            adb_path=adb_path,
            device_serial=device_serial,
            logger=logger,
        )
        if pair is not None:
            update_vision_scale()
            return pair[0], pair[1]
        return wincap, screenshot_provider


class PcWin32Surface:
    """Attach to the native game window and capture via mss."""

    @staticmethod
    def connect_blocking(
        *,
        window_name: str,
        adb_path: Optional[str],
        logger: logging.Logger,
        timeout_sec: float = 90.0,
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        First ``WindowCapture`` attempt in a side thread (avoids FindWindow freeze on Ctrl+C).

        Returns ``(wincap, screenshot_provider)``; either may be ``None`` if the window is missing.
        """
        from screenshot_provider import create_screenshot_provider
        from windowcapture import WindowCapture

        import adb_input as adb_mod

        adb_mod.set_adb_input(None)
        result: list = []
        done = threading.Event()

        def _create() -> None:
            try:
                result.append(WindowCapture(window_name))
            except Exception:
                pass
            finally:
                done.set()

        logger.info("Connecting to game window '%s'...", window_name)
        sys.stdout.flush()
        threading.Thread(target=_create, daemon=True).start()
        try:
            if not done.wait(timeout=timeout_sec):
                logger.warning(
                    "Window '%s' not found or timeout (%.0fs) — UI will start in disconnected mode.",
                    window_name,
                    timeout_sec,
                )
        except KeyboardInterrupt:
            raise

        wincap = result[0] if result else None
        if wincap is None:
            logger.warning(
                "Window '%s' not found — starting UI in disconnected mode. "
                "Launch the game (or change emulator in Settings) and the bot will auto-connect.",
                window_name,
            )
            return None, None

        sp = create_screenshot_provider("pc", wincap=wincap, adb_path=adb_path)
        logger.info("[Screenshot] Provider: %s (emulator=pc)", type(sp).__name__)
        return wincap, sp

    @staticmethod
    def try_attach_once(
        *,
        window_name: str,
        adb_path: Optional[str],
        auto_focus: bool,
        win_w: Optional[int],
        win_h: Optional[int],
        lastz_pid: Dict[str, Any],
        logger: logging.Logger,
    ) -> Optional[Tuple[Any, Any]]:
        """Single successful window attach, or ``None``."""
        from screenshot_provider import create_screenshot_provider
        from windowcapture import WindowCapture

        try:
            new_wincap = WindowCapture(window_name)
            new_wincap.auto_focus = auto_focus
            if win_w and win_h:
                new_wincap.resize_to_client(win_w, win_h)
            new_sp = create_screenshot_provider("pc", wincap=new_wincap, adb_path=adb_path)
            try:
                import win32process

                _, pid = win32process.GetWindowThreadProcessId(new_wincap.hwnd)
                lastz_pid["pid"] = pid
            except Exception:
                pass
            logger.info("[WindowWatcher] Connected to '%s'.", window_name)
            return new_wincap, new_sp
        except Exception:
            return None

    @staticmethod
    def maybe_autostart_lastz(
        *,
        window_name: str,
        lastz_exe_path: str,
        bot_paused: Dict[str, Any],
        last_attempt_holder: Dict[str, float],
        min_interval_sec: float = 15.0,
        logger: logging.Logger,
    ) -> None:
        """Launch LastZ.exe if enabled, window missing, and interval elapsed."""
        if not lastz_exe_path or not os.path.isfile(lastz_exe_path):
            return
        if bot_paused.get("paused"):
            return
        now = time.time()
        if now - last_attempt_holder.get("t", 0.0) < min_interval_sec:
            return
        try:
            import win32gui as _wg

            if _wg.FindWindow(None, window_name):
                return
        except Exception:
            return
        last_attempt_holder["t"] = now
        logger.info("[WindowWatcher] Auto-starting LastZ: %s", lastz_exe_path)
        subprocess.Popen(
            lastz_exe_path,
            cwd=os.path.dirname(lastz_exe_path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    @staticmethod
    def watcher_step(
        wincap: Any,
        screenshot_provider: Any,
        *,
        window_name: str,
        adb_path: Optional[str],
        auto_focus: bool,
        win_w: Optional[int],
        win_h: Optional[int],
        lastz_pid: Dict[str, Any],
        lastz_exe_path: str,
        auto_start_lastz: bool,
        bot_paused: Dict[str, Any],
        autostart_state: Dict[str, float],
        logger: logging.Logger,
        update_vision_scale: Callable[[], None],
    ) -> Tuple[Any, Any]:
        """One 3s-tick: validate HWND, autostart LastZ if needed, or attach window."""
        if wincap is not None:
            if not is_game_surface_valid(wincap):
                logger.warning("[WindowWatcher] Window handle became invalid — disconnecting.")
                return None, None
            return wincap, screenshot_provider

        if auto_start_lastz:
            PcWin32Surface.maybe_autostart_lastz(
                window_name=window_name,
                lastz_exe_path=lastz_exe_path,
                bot_paused=bot_paused,
                last_attempt_holder=autostart_state,
                logger=logger,
            )

        pair = PcWin32Surface.try_attach_once(
            window_name=window_name,
            adb_path=adb_path,
            auto_focus=auto_focus,
            win_w=win_w,
            win_h=win_h,
            lastz_pid=lastz_pid,
            logger=logger,
        )
        if pair is not None:
            update_vision_scale()
            return pair[0], pair[1]
        return wincap, screenshot_provider


def initial_pc_connect_or_none(
    *,
    window_name: str,
    adb_path: Optional[str],
    logger: logging.Logger,
    timeout_sec: float = 90.0,
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    PC bootstrap: blocking first capture. On success returns pair; on missing window ``(None, None)``.

    Does **not** set deferred provider — returns ``(None, None)`` and caller logs deferred message.
    """
    try:
        wincap, sp = PcWin32Surface.connect_blocking(
            window_name=window_name,
            adb_path=adb_path,
            logger=logger,
            timeout_sec=timeout_sec,
        )
    except KeyboardInterrupt:
        raise
    if wincap is None:
        logger.info("[Screenshot] Provider deferred — window not connected yet.")
        return None, None
    return wincap, sp
