import time
from vision import Vision


class TreasureDetector:
    """Detect treasure (Treasure Helicopter) by matching a treasure icon.

    - Treasure appeared: icon found in any single frame.
    - Treasure gone:      icon absent for `clear_sec` consecutive seconds.
    Zalo notification is configured in function YAML (event_type: send_zalo) and triggered by config.
    """

    def __init__(self, treasure_template_path: str,
                 threshold: float = 0.6, clear_sec: float = 10.0):
        self._vision = Vision(treasure_template_path)
        self._threshold = threshold
        self._clear_sec = clear_sec
        self._treasure_visible = False
        self._clear_since = None

    def update(self, screenshot, log):
        """Call once per captured frame.

        Returns:
            "started"  — treasure just appeared this frame
            "ended"    — treasure just disappeared this frame
            None       — no state change
        """
        icon = self._vision.exists(screenshot, threshold=self._threshold)
        now = time.time()

        if not self._treasure_visible:
            if icon:
                self._treasure_visible = True
                self._clear_since = None
                log.info("[Alert] Treasure detected!")
                return "started"
        else:
            if icon:
                self._clear_since = None  # still visible → reset countdown
            else:
                if self._clear_since is None:
                    self._clear_since = now  # start countdown
                elif now - self._clear_since >= self._clear_sec:
                    self._treasure_visible = False
                    self._clear_since = None
                    log.info("[Alert] Treasure no longer visible.")
                    return "ended"
        return None
