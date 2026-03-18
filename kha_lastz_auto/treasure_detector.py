import time
from vision import Vision, get_global_scale


class TreasureDetector:
    """Detect treasure (Treasure Helicopter) by matching a treasure icon.

    - Treasure appeared: icon found in any single frame (or re-trigger when still visible, see below).
    - Treasure gone:      icon absent for `clear_sec` consecutive seconds.
    - Re-trigger: while treasure stays visible, emit "started" again every `re_trigger_interval_sec`
      so ClickTreasure can run even if the first trigger was missed (e.g. bot started with treasure already on screen).
    - Optional ROI: when roi_center_x, roi_center_y, roi_padding are set, only search within that region.
    """

    def __init__(self, treasure_template_path: str,
                 threshold: float = 0.5, clear_sec: float = 10.0,
                 re_trigger_interval_sec: float = 60.0,
                 roi_center_x: float = None, roi_center_y: float = None, roi_padding: float = None):
        self._vision = Vision(treasure_template_path)
        self._threshold = threshold
        self._clear_sec = clear_sec
        self._re_trigger_interval_sec = re_trigger_interval_sec
        self._roi_center_x = roi_center_x
        self._roi_center_y = roi_center_y
        self._roi_padding = roi_padding if roi_padding is not None else 2.0
        self._treasure_visible = False
        self._clear_since = None
        self._last_started_at = None  # when we last emitted "started" (for re-trigger while visible)

    def reset(self):
        """Reset state so the next update re-evaluates from scratch (e.g. after Is Running toggled back ON)."""
        self._treasure_visible = False
        self._clear_since = None
        self._last_started_at = None

    def update(self, screenshot, log):
        """Call once per captured frame.

        Returns:
            "started"  — treasure just appeared this frame, or re-trigger while still visible
            "ended"    — treasure just disappeared this frame
            None       — no state change
        """
        search_img = screenshot
        crop_info = "full"
        if self._roi_center_x is not None and self._roi_center_y is not None and screenshot is not None:
            sh, sw = screenshot.shape[:2]
            scale = get_global_scale()
            nw_px = max(1, int(self._vision.needle_w * scale))
            nh_px = max(1, int(self._vision.needle_h * scale))
            cx_px = int(self._roi_center_x * sw)
            cy_px = int(self._roi_center_y * sh)
            half_w = int(nw_px * self._roi_padding)
            half_h = int(nh_px * self._roi_padding)
            rx = max(0, cx_px - half_w)
            ry = max(0, cy_px - half_h)
            rx2 = min(sw, cx_px + half_w)
            ry2 = min(sh, cy_px + half_h)
            search_img = screenshot[ry:ry2, rx:rx2]
            crop_h, crop_w = search_img.shape[:2]
            crop_info = (
                "roi=({},{})→({},{}) crop={}x{} "
                "needle={}x{} scale={:.2f}".format(
                    rx, ry, rx2, ry2, crop_w, crop_h,
                    self._vision.needle_w, self._vision.needle_h, scale,
                )
            )

        score = self._vision.match_score(search_img)
        icon = score >= self._threshold
        log.debug(
            "[TreasureDetector] score=%.3f threshold=%.2f found=%s visible=%s %s",
            score, self._threshold, icon, self._treasure_visible, crop_info,
        )
        now = time.time()

        if not self._treasure_visible:
            if icon:
                self._treasure_visible = True
                self._clear_since = None
                self._last_started_at = now
                log.info("[Alert] Treasure detected!")
                return "started"
        else:
            if icon:
                self._clear_since = None  # still visible → reset countdown
                # Re-trigger so ClickTreasure can run if first trigger was missed
                if self._re_trigger_interval_sec > 0 and self._last_started_at is not None:
                    if now - self._last_started_at >= self._re_trigger_interval_sec:
                        self._last_started_at = now
                        log.info("[Alert] Treasure still visible, re-triggering.")
                        return "started"
            else:
                if self._clear_since is None:
                    self._clear_since = now  # start countdown
                elif now - self._clear_since >= self._clear_sec:
                    self._treasure_visible = False
                    self._clear_since = None
                    self._last_started_at = None
                    log.info("[Alert] Treasure no longer visible.")
                    return "ended"
        return None
