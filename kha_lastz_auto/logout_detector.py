import time
from vision import Vision


class LogoutDetector:
    """Detect logged-out state by matching the PasswordSlot icon.

    - Logged out starts:  icon found for `confirm_sec` consecutive seconds.
    - Logged out ends:    icon absent for `clear_sec` consecutive seconds.
    - Re-trigger:         if still on login screen after `retry_sec` seconds with no handler
                          running, emit "started" again so the scheduler can retry PinLoggin.

    Using a short `confirm_sec` avoids false positives from brief UI flickers.
    """

    def __init__(self, template_path: str,
                 threshold: float = 0.75,
                 confirm_sec: float = 1.0,
                 clear_sec: float = 5.0,
                 retry_sec: float = 60.0):
        self._vision = Vision(template_path)
        self._threshold = threshold
        self._confirm_sec = confirm_sec
        self._clear_sec = clear_sec
        self._retry_sec = retry_sec
        self._logged_out = False
        self._seen_since = None      # when icon first appeared (confirming logout)
        self._clear_since = None     # when icon first disappeared (confirming login)
        self._last_trigger_time = 0.0  # last time "started" was emitted

    def reset(self):
        """Reset state so the next update re-evaluates from scratch (e.g. after Is Running toggled back ON)."""
        self._logged_out = False
        self._seen_since = None
        self._clear_since = None
        self._last_trigger_time = 0.0

    def update(self, screenshot, log):
        """Call once per captured frame.

        Returns:
            "started"  — logged-out state just detected this frame (or re-triggered after retry_sec)
            "ended"    — logged back in (login screen gone)
            None       — no state change
        """
        icon = self._vision.exists(screenshot, threshold=self._threshold)
        now = time.time()

        if not self._logged_out:
            if icon:
                if self._seen_since is None:
                    self._seen_since = now
                elif now - self._seen_since >= self._confirm_sec:
                    self._logged_out = True
                    self._seen_since = None
                    self._clear_since = None
                    self._last_trigger_time = now
                    log.info("[Alert] Logged out detected! Triggering PinLoggin.")
                    return "started"
            else:
                self._seen_since = None
        else:
            if icon:
                self._clear_since = None   # still on login screen → reset countdown
                # Re-trigger if still stuck on login screen after retry_sec
                if now - self._last_trigger_time >= self._retry_sec:
                    self._last_trigger_time = now
                    log.info("[Alert] Still on login screen after %.0fs — re-triggering PinLoggin.", self._retry_sec)
                    return "started"
            else:
                if self._clear_since is None:
                    self._clear_since = now
                elif now - self._clear_since >= self._clear_sec:
                    self._logged_out = False
                    self._clear_since = None
                    self._last_trigger_time = 0.0
                    log.info("[Alert] Login successful, logged-out state cleared.")
                    return "ended"
        return None
