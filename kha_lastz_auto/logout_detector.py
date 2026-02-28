import time
from vision import Vision


class LogoutDetector:
    """Detect logged-out state by matching the PasswordSlot icon.

    - Logged out starts:  icon found for `confirm_sec` consecutive seconds.
    - Logged out ends:    icon absent for `clear_sec` consecutive seconds.

    Using a short `confirm_sec` avoids false positives from brief UI flickers.
    """

    def __init__(self, template_path: str,
                 threshold: float = 0.75,
                 confirm_sec: float = 1.0,
                 clear_sec: float = 5.0):
        self._vision = Vision(template_path)
        self._threshold = threshold
        self._confirm_sec = confirm_sec
        self._clear_sec = clear_sec
        self._logged_out = False
        self._seen_since = None    # when icon first appeared (confirming logout)
        self._clear_since = None   # when icon first disappeared (confirming login)

    def update(self, screenshot, log):
        """Call once per captured frame.

        Returns:
            "started"  — logged-out state just detected this frame
            "ended"    — logged back in (login screen gone)
            None       — no state change
        """
        icon = bool(self._vision.find(screenshot, threshold=self._threshold))
        now = time.time()

        if not self._logged_out:
            if icon:
                if self._seen_since is None:
                    self._seen_since = now
                elif now - self._seen_since >= self._confirm_sec:
                    self._logged_out = True
                    self._seen_since = None
                    self._clear_since = None
                    log.info("[Alert] Logged out detected! Triggering PinLoggin.")
                    return "started"
            else:
                self._seen_since = None
        else:
            if icon:
                self._clear_since = None   # still on login screen → reset countdown
            else:
                if self._clear_since is None:
                    self._clear_since = now
                elif now - self._clear_since >= self._clear_sec:
                    self._logged_out = False
                    self._clear_since = None
                    log.info("[Alert] Login successful, logged-out state cleared.")
                    return "ended"
        return None
