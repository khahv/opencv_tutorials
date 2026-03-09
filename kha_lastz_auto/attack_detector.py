import time
from vision import Vision

try:
    import zalo_clicker
except ImportError:
    zalo_clicker = None


class AttackDetector:
    """Detect being-attacked state by matching the BeingAttackedWarning icon.

    - Attack starts:  icon found in any single frame.
    - Attack ends:    icon absent for `clear_sec` consecutive seconds.
    Zalo message is configured in function YAML (event_type: send_zalo) and triggered by config.
    """

    def __init__(self, warning_template_path: str,
                 threshold: float = 0.6, clear_sec: float = 10.0):
        self._vision = Vision(warning_template_path)
        self._threshold = threshold
        self._clear_sec = clear_sec
        self._attacked = False
        self._clear_since = None

    def update(self, screenshot, log):
        """Call once per captured frame.

        Returns:
            "started"  — attack just began this frame
            "ended"    — attack just ended this frame
            None       — no state change
        """
        icon = self._vision.exists(screenshot, threshold=self._threshold)
        now = time.time()

        if not self._attacked:
            if icon:
                self._attacked = True
                self._clear_since = None
                log.info("[Alert] House is being attacked!")
                if zalo_clicker:
                    try:
                        zalo_clicker.run_zalo_click(logger=log)
                    except Exception as e:
                        log.warning("[ZaloClicker] %s", e)
                return "started"
        else:
            if icon:
                self._clear_since = None          # still attacked → reset countdown
            else:
                if self._clear_since is None:
                    self._clear_since = now       # start countdown
                elif now - self._clear_since >= self._clear_sec:
                    self._attacked = False
                    self._clear_since = None
                    log.info("[Alert] Attack has ended.")
                    return "ended"
        return None
