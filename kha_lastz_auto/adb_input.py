"""
adb_input.py
------------
ADB-based input (tap / swipe) for LDPlayer emulator mode.

Coordinates are always in **device (client) pixels** — the same coordinate
space as the game screenshot — NOT Windows screen pixels.

Commands issued:
    tap:   adb shell input tap   <x> <y>
    swipe: adb shell input swipe <x1> <y1> <x2> <y2> [duration_ms]

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
import subprocess
from typing import Optional, List

log = logging.getLogger("kha_lastz")

_DEFAULT_ADB_PATH = r"C:\LDPlayer\LDPlayer9\adb.exe"
_ADB_TIMEOUT_SEC = 5

# LDPlayer assigns one TCP port per emulator instance starting at 5555,
# incrementing by 2 (5555, 5557, 5559 …).  We probe the first 8 slots.
_LDPLAYER_ADB_PORTS: List[int] = [5555, 5557, 5559, 5561, 5563, 5565, 5567, 5569]


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


# ── Module-level singleton ─────────────────────────────────────────────────────
# Set once by main.py after emulator mode is determined; None in PC mode.

_instance: Optional[AdbInput] = None


def get_adb_input() -> Optional[AdbInput]:
    """Return the active AdbInput instance, or None when in PC (win32) mode."""
    return _instance


def set_adb_input(inst: Optional[AdbInput]) -> None:
    """Set (or clear) the global AdbInput instance. Called once by main.py."""
    global _instance
    _instance = inst
