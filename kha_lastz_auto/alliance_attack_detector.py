import time
from vision import Vision


class AllianceAttackDetector:
    """Detect alliance-being-attacked state by matching the AllianceBeingAttackedWarning icon.

    - Attack starts:  icon found in any single frame.
    - Attack ends:    icon absent for `clear_sec` consecutive seconds.
    Zalo notification is configured in function YAML (event_type: send_zalo) and triggered by config.
    """

    def __init__(self, warning_template_path: str,
                 threshold: float = 0.6, clear_sec: float = 10.0):
        self._vision = Vision(warning_template_path)
        self._threshold = threshold
        self._clear_sec = clear_sec
        self._attacked = False
        self._clear_since = None

    def reset(self):
        """Reset state so the next update re-evaluates from scratch (e.g. after Is Running toggled back ON)."""
        self._attacked = False
        self._clear_since = None

    def update(self, screenshot, log):
        """Call once per captured frame.

        Returns:
            "started"  — alliance attack just began this frame
            "ended"    — alliance attack just ended this frame
            None       — no state change
        """
        icon = self._vision.exists(screenshot, threshold=self._threshold)
        now = time.time()

        if not self._attacked:
            if icon:
                self._attacked = True
                self._clear_since = None
                log.info("[Alert] Alliance is being attacked!")
                return "started"
        else:
            if icon:
                self._clear_since = None
            else:
                if self._clear_since is None:
                    self._clear_since = now
                elif now - self._clear_since >= self._clear_sec:
                    self._attacked = False
                    self._clear_since = None
                    log.info("[Alert] Alliance attack has ended.")
                    return "ended"
        return None
