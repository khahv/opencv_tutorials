from vision import Vision


class ExitBannerDetector:
    """Detect 'Exit the game?' banner and click corner to dismiss.

    Checks every `check_every` detector ticks (each tick = detector interval seconds).
    """

    def __init__(self, template_path: str, min_match_count: int = 15, check_every: int = 5):
        self._vision      = Vision(template_path)
        self._threshold   = min_match_count
        self._check_every = check_every
        self._tick        = 0

    def update(self, screenshot, wincap, log) -> bool:
        """Call once per detector tick.

        Returns True if banner was detected (and corner click was triggered).
        Caller is responsible for performing the actual click via the returned coords.
        """
        self._tick += 1
        if self._tick % self._check_every != 0:
            return False
        return self._vision.exists(screenshot, min_match_count=self._threshold)

    def corner_screen_pos(self, wincap):
        """Top-right corner of the game window — where the X button sits."""
        return wincap.get_screen_position((int(wincap.w * 0.95), int(wincap.h * 0.05)))
