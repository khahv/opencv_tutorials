import time
from vision import Vision


class AttackDetector:
    """Detect being-attacked state by matching the BeingAttackedWarning icon.

    - Attack starts:  icon found in any single frame.
    - Attack ends:    icon absent for `clear_sec` consecutive seconds.
    """

    def __init__(self, warning_template_path: str,
                 threshold: float = 0.75, clear_sec: float = 10.0):
        self._vision = Vision(warning_template_path)
        self._threshold = threshold
        self._clear_sec = clear_sec
        self._attacked = False
        self._clear_since = None

    def update(self, screenshot, log) -> None:
        """Call once per captured frame. Logs state changes only."""
        icon = bool(self._vision.find(screenshot, threshold=self._threshold))
        now = time.time()

        if not self._attacked:
            if icon:
                self._attacked = True
                self._clear_since = None
                log.info("[Alert] House is being attacked!")
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
