import cv2 as cv
import numpy as np
import logging
import threading
import time
import functools

_log = logging.getLogger("kha_lastz")


# ---------------------------------------------------------------------------
# Performance decorator
# ---------------------------------------------------------------------------
def timeit(func):
    """Decorator: log execution time of find/exists in milliseconds, including template name."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        # args[0] is `self` for Vision methods — grab needle name if available
        needle = getattr(args[0], 'needle_name', '?') if args else '?'
        _log.debug("[timeit] Vision.%s [%s] took %.2f ms", func.__name__, needle, elapsed_ms)
        return result
    return wrapper


# ---------------------------------------------------------------------------
# Hue helpers (unchanged)
# ---------------------------------------------------------------------------
def _hue_to_dominant_color(hue):
    """Map OpenCV hue (0-180) to dominant color name."""
    if hue is None:
        return None
    h = float(hue)
    if h < 8 or h >= 172:
        return "red"
    if h < 22:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 75:
        return "green"
    if h < 100:
        return "cyan"
    if h < 130:
        return "blue"
    if h < 160:
        return "purple"
    return "magenta"


# ---------------------------------------------------------------------------
# Global scale (set once from main.py based on window-size ratio)
# ---------------------------------------------------------------------------
_global_scale = 1.0

# Per-frame gray/edge cache — shared within a thread, safe across threads via
# threading.local().  We keep a reference to the src array so Python cannot
# reuse the same memory address for a new frame while the cache is still live.
_frame_cache = threading.local()


def _get_gray(img):
    if getattr(_frame_cache, 'gray_src', None) is img:
        return _frame_cache.gray_img
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _frame_cache.gray_src = img
    _frame_cache.gray_img = gray
    return gray





def set_global_scale(scale):
    global _global_scale
    _global_scale = float(scale)


def get_global_scale():
    return _global_scale


# ---------------------------------------------------------------------------
# Multi-scale pyramid matching (edge-based, robust across resolutions)
# ---------------------------------------------------------------------------
# Scale factors tried during multi-scale search.
# 1.0 = same size as stored needle; 0.8/1.2 cover ±20% resolution variance.
_PYRAMID_SCALES = np.linspace(0.85, 1.15, 5)  # [0.85, 0.925, 1.0, 1.075, 1.15]

# Canny thresholds for needle and haystack edges



def _match_multiscale(haystack_img, needle_img, method, threshold, base_scale=1.0, debug_log=False, needle_name='?'):
    """
    Search needle_img inside haystack_img at multiple scales.
    Supports both Grayscale and BGR color matching (depends on inputs).

    base_scale  = _global_scale (e.g. 0.5 when window is half of reference).
                  Applied first so the needle is resized to the same pixel
                  density as the haystack before pyramid variations are tried.
    pyramid_s   = fine-grained ±20 % adjustment around base_scale to handle
                  slight resolution differences between captures.

    Returns a list of (x, y, w, h) tuples in *haystack_img* coordinate space.
    """
    nh_base, nw_base = needle_img.shape[:2]
    hits = []
    best_score = -1.0

    for s in _PYRAMID_SCALES:
        effective = base_scale * s          # e.g. 0.5 * 1.0 = 0.50, 0.5 * 1.15 = 0.575
        nw = max(4, int(nw_base * effective))
        nh = max(4, int(nh_base * effective))

        # Skip if scaled needle is larger than the haystack
        if nw > haystack_img.shape[1] or nh > haystack_img.shape[0]:
            if debug_log:
                _log.debug("[vision:debug] [%s] scale=%.3f  needle %dx%d > haystack %dx%d → SKIP",
                           needle_name, effective, nw, nh,
                           haystack_img.shape[1], haystack_img.shape[0])
            continue

        # Resize the needle image
        # INTER_LINEAR keeps contrast higher than INTER_AREA for tiny UI elements.
        needle_scaled = cv.resize(needle_img, (nw, nh),
                                  interpolation=cv.INTER_LINEAR)

        result = cv.matchTemplate(haystack_img, needle_scaled, method)
        _, max_val, _, max_loc = cv.minMaxLoc(result)
        if max_val > best_score:
            best_score = max_val

        locs = list(zip(*np.where(result >= threshold)[::-1]))  # (x, y) pairs
        passed = len(locs) > 0

        if debug_log:
            _log.debug("[vision:debug] [%s] scale=%.3f  needle %dx%d  max_score=%.4f  threshold=%.2f  %s  best_loc=%s",
                       needle_name, effective, nw, nh, max_val, threshold,
                       "PASS (%d hits)" % len(locs) if passed else "FAIL",
                       max_loc)

        for loc in locs:
            hits.append((int(loc[0]), int(loc[1]), nw, nh))

    if debug_log:
        _log.debug("[vision:debug] [%s] base_scale=%.3f  best_score_overall=%.4f  total_hits=%d",
                   needle_name, base_scale, best_score, len(hits))

    return hits


# ---------------------------------------------------------------------------
# Vision class
# ---------------------------------------------------------------------------
class Vision:

    needle_img   = None
    needle_w     = 0
    needle_h     = 0
    method       = None
    needle_gray  = None
    needle_bgr   = None

    def __init__(self, needle_img_path, method=cv.TM_CCOEFF_NORMED):
        img = cv.imread(needle_img_path, cv.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError('Khong tim thay anh: {}'.format(needle_img_path))

        # Drop alpha channel if present
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)

        self.needle_img  = img
        import os
        self.needle_name = os.path.splitext(os.path.basename(needle_img_path))[0]
        self.needle_w    = img.shape[1]
        self.needle_h    = img.shape[0]
        self.method      = method

        # Pre-compute grayscale
        self.needle_gray = (cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                            if len(img.shape) == 3 else img)
        
        # Pre-compute BGR for color matching
        self.needle_bgr = img if len(img.shape) == 3 else cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @timeit
    def exists(self, haystack_img, threshold=0.5, debug_log=False) -> bool:
        """
        Fast existence check using edge + pyramid matching.
        Returns True if any match is found above *threshold*.
        Set debug_log=True to see per-scale scores in the log.
        """
        norm = self._normalize_haystack(haystack_img)
        if norm is None:
            if debug_log:
                _log.debug("[vision:debug] [%s] exists: needle too large for haystack → False", self.needle_name)
            return False

        haystack_gray = _get_gray(norm)
        hits = _match_multiscale(haystack_gray, self.needle_gray,
                                 self.method, threshold,
                                 base_scale=_global_scale,
                                 debug_log=debug_log,
                                 needle_name=self.needle_name)
        return len(hits) > 0

    @timeit
    def find(self, haystack_img, threshold=0.5, debug_mode=None, is_color=False, debug_log=False):
        """
        Multi-scale template matching.

        *is_color* determines if we match on Grayscale (default) or BGR color matrices.

        Returns a list of (cx, cy) centre-points in *original* window coordinates.
        Set debug_log=True to see per-scale scores in the log.
        """
        norm = self._normalize_haystack(haystack_img)
        if norm is None:
            if debug_log:
                _log.debug("[vision:debug] [%s] find: needle too large for haystack → []", self.needle_name)
            return []

        if is_color:
            haystack_target = norm if len(norm.shape) == 3 else cv.cvtColor(norm, cv.COLOR_GRAY2BGR)
            needle_target = self.needle_bgr
        else:
            haystack_target = _get_gray(norm)
            needle_target = self.needle_gray

        hits = _match_multiscale(haystack_target, needle_target,
                                 self.method, threshold,
                                 base_scale=_global_scale,
                                 debug_log=debug_log,
                                 needle_name=self.needle_name)
        if not hits:
            return []

        # Group overlapping rectangles across all pyramid scales
        rects = []
        for x, y, w, h in hits:
            r = [x, y, w, h]
            rects.append(r)
            rects.append(r)   # groupRectangles needs each rect twice

        grouped, _ = cv.groupRectangles(rects, groupThreshold=1, eps=0.5)
        if not len(grouped):
            return []

        # Coordinates are already in native window space (no upscaling was done)
        points = []
        for x, y, w, h in grouped:
            cx = int(x + w // 2)
            cy = int(y + h // 2)
            points.append((cx, cy, w, h))

            if debug_mode == "rectangles":
                cv.rectangle(haystack_img,
                             (x, y),
                             (x + w, y + h),
                             (0, 255, 0), 2)
            elif debug_mode == "points":
                cv.drawMarker(haystack_img,
                              (cx, cy),
                              (255, 0, 255),
                              cv.MARKER_CROSS, 40, 2)

        if debug_mode in ("rectangles", "points"):
            cv.imshow("Matches", haystack_img)

        return points

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_haystack(self, haystack_img):
        """
        Return the haystack at its NATIVE resolution — no upscaling.

        Previously we upscaled the screenshot to match the template's reference
        resolution. This introduced blur (INTER_LINEAR on 2× scale) which
        degraded Canny edge quality, causing match failures.

        Instead, we now downscale the *needle* inside _match_multiscale via
        `base_scale = _global_scale`.  This keeps Canny running on sharp,
        unmodified pixels from the game capture.

        Sanity-check: if even the smallest pyramid-scaled needle (base*0.85)
        exceeds the haystack dimensions, there is no point trying.
        """
        min_scale = _global_scale * min(_PYRAMID_SCALES)
        nw_min = max(4, int(self.needle_w * min_scale))
        nh_min = max(4, int(self.needle_h * min_scale))
        if nw_min > haystack_img.shape[1] or nh_min > haystack_img.shape[0]:
            return None
        return haystack_img