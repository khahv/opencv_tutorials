import cv2 as cv
import numpy as np
import logging
import threading
import time
import functools
import os

_log = logging.getLogger("kha_lastz")

# ---------------------------------------------------------------------------
# Performance decorator
# ---------------------------------------------------------------------------
def timeit(func):
    """Decorator: log execution time of find/exists in milliseconds."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        needle = getattr(args[0], 'needle_name', '?') if args else '?'
        if elapsed_ms > 20:
            _log.debug("[timeit] Vision.%s [%s] took %.2f ms", func.__name__, needle, elapsed_ms)
        return result
    return wrapper

# ---------------------------------------------------------------------------
# Hue helpers (kept for backward compatibility)
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
# Global scale: set once from main.py based on window_width / reference_width
# ---------------------------------------------------------------------------
_global_scale = 1.0

# Per-frame gray cache: BGR→Gray is expensive on full-res screenshots.
# threading.local() ensures each thread (detector + main) has its own cache.
_gray_cache = threading.local()


def _get_gray(img):
    """Return grayscale version of img. Cache per-frame to avoid repeated conversion."""
    if getattr(_gray_cache, 'src', None) is img:
        return _gray_cache.img
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _gray_cache.src = img
    _gray_cache.img = gray
    return gray


def set_global_scale(scale):
    global _global_scale
    _global_scale = float(scale)


def get_global_scale():
    return _global_scale


# ---------------------------------------------------------------------------
# Vision class
# ---------------------------------------------------------------------------
class Vision:
    """
    Template matcher supporting two modes:
      - Default (is_color=False): Distance Transform (Chamfer) Matching on edges.
        Robust to lighting/color variation. Good for icons/shapes.
      - Color (is_color=True): BGR TM_CCOEFF_NORMED.
        Good for text buttons, colourful UI elements with unique colours.

    Scale handling (correct approach):
      Templates are captured at reference resolution (e.g. 1080x1920).
      The game window runs at a smaller resolution (e.g. 540x960).
      scale = window / reference  (e.g. 0.5)

      We DOWNSCALE the needle to match the haystack, NOT upscale the haystack.
      This keeps the haystack pixel-perfect (no interpolation artefacts) and
      works for any arbitrary window size, not just exact 2x multiples.
    """

    needle_img   = None
    needle_w     = 0
    needle_h     = 0
    method       = None

    def __init__(self, needle_img_path, method=cv.TM_CCOEFF_NORMED):
        img = cv.imread(needle_img_path, cv.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError('Khong tim thay anh: {}'.format(needle_img_path))

        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)

        self.needle_img  = img
        self.needle_name = os.path.splitext(os.path.basename(needle_img_path))[0]
        self.needle_w    = img.shape[1]
        self.needle_h    = img.shape[0]
        self.method      = method

        # Pre-compute reference-resolution grayscale + edges (for DT matching)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        self.needle_gray_ref  = gray                       # full-res gray (1080p)
        self.needle_edges_ref = cv.Canny(gray, 50, 150)   # full-res edges

        _log.info("[vision:info] [%s] Load Distance Transform Model thành công (%dx%d).",
                  self.needle_name, self.needle_w, self.needle_h)

    # ------------------------------------------------------------------
    # Internal helpers: return scaled needle artefacts for current window
    # ------------------------------------------------------------------
    def _scaled_needle_color(self, scale):
        """Needle BGR image scaled to current window resolution."""
        if scale == 1.0:
            return self.needle_img
        nw = max(1, int(round(self.needle_w * scale)))
        nh = max(1, int(round(self.needle_h * scale)))
        interp = cv.INTER_AREA if scale < 1.0 else cv.INTER_LINEAR
        return cv.resize(self.needle_img, (nw, nh), interpolation=interp)

    def _scaled_needle_gray(self, scale):
        """Needle grayscale image scaled to current window resolution."""
        if scale == 1.0:
            return self.needle_gray_ref
        nw = max(1, int(round(self.needle_w * scale)))
        nh = max(1, int(round(self.needle_h * scale)))
        interp = cv.INTER_AREA if scale < 1.0 else cv.INTER_LINEAR
        return cv.resize(self.needle_gray_ref, (nw, nh), interpolation=interp)

    def _scaled_needle_edges(self, scale):
        """Needle edge mask scaled to current window resolution."""
        if scale == 1.0:
            return self.needle_edges_ref
        nw = max(1, int(round(self.needle_w * scale)))
        nh = max(1, int(round(self.needle_h * scale)))
        # Scale binary edges then re-threshold to keep clean binary mask
        interp = cv.INTER_AREA if scale < 1.0 else cv.INTER_LINEAR
        scaled = cv.resize(self.needle_edges_ref, (nw, nh), interpolation=interp)
        _, binary = cv.threshold(scaled, 64, 255, cv.THRESH_BINARY)
        return binary

    # ------------------------------------------------------------------
    @timeit
    def exists(self, haystack_img, threshold=0.5, debug_log=False) -> bool:
        """Fast existence check — no groupRectangles, no coordinate output."""
        scale = _global_scale
        needle_gray = self._scaled_needle_gray(scale)
        nw, nh = needle_gray.shape[1], needle_gray.shape[0]
        if nw > haystack_img.shape[1] or nh > haystack_img.shape[0]:
            return False
        hay_gray = _get_gray(haystack_img)
        result = cv.matchTemplate(hay_gray, needle_gray, cv.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv.minMaxLoc(result)
        return max_val >= threshold

    # ------------------------------------------------------------------
    @timeit
    def find(self, haystack_img, threshold=0.5, debug_mode=None, is_color=False,
             debug_log=False, multi=False, color_tolerance=None, ratio_test=None, min_inliers=None):
        """
        Find needle in haystack. Returns list of (cx, cy, w, h) in haystack coords.

        is_color=False     → Distance Transform (Chamfer) matching (default)
        is_color=True      → BGR TM_CCOEFF_NORMED matching (match_color: true in YAML)
        multi=True         → Return ALL matches above threshold (for match_count with count > 1)
        color_tolerance    → Max mean BGR distance (0-255) between matched region and template.
                             None = disabled. E.g. 30 = strict color check.
        """
        scale = _global_scale

        if is_color:
            if multi:
                return self._find_color_multi(haystack_img, threshold, scale, debug_log, color_tolerance)
            return self._find_color(haystack_img, threshold, scale, debug_log, color_tolerance)
        else:
            return self._find_dt(haystack_img, threshold, scale, debug_log)

    # ------------------------------------------------------------------
    def _find_color(self, haystack_img, threshold, scale, debug_log, color_tolerance=None):
        """BGR template matching. Returns single best match above threshold."""
        needle = self._scaled_needle_color(scale)
        nw, nh = needle.shape[1], needle.shape[0]
        if nw > haystack_img.shape[1] or nh > haystack_img.shape[0]:
            return []

        # Ensure haystack is BGR
        hay = haystack_img if len(haystack_img.shape) == 3 else cv.cvtColor(haystack_img, cv.COLOR_GRAY2BGR)

        result = cv.matchTemplate(hay, needle, cv.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv.minMaxLoc(result)

        if max_val < threshold:
            if debug_log:
                _log.debug("[vision:debug] [%s] Color Match Failed: score %.2f < %.2f",
                           self.needle_name, max_val, threshold)
            return []

        x, y = max_loc

        # Optional: strict color check — reject if mean BGR differs too much from template
        if color_tolerance is not None:
            region = hay[y:y+nh, x:x+nw].astype(np.float32)
            needle_f = needle.astype(np.float32)
            color_dist = float(np.linalg.norm(
                np.mean(region.reshape(-1, 3), axis=0) - np.mean(needle_f.reshape(-1, 3), axis=0)
            ))
            if color_dist > color_tolerance:
                if debug_log:
                    _log.debug("[vision:debug] [%s] Color Reject: mean BGR dist=%.1f > tol=%.1f",
                               self.needle_name, color_dist, color_tolerance)
                return []

        cx = int(x + nw // 2)
        cy = int(y + nh // 2)

        if debug_log:
            _log.debug("[vision:debug] [%s] Color Match OK! Score: %.2f (Thresh: %.2f)",
                       self.needle_name, max_val, threshold)

        return [(cx, cy, nw, nh)]

    # ------------------------------------------------------------------
    def _find_color_multi(self, haystack_img, threshold, scale, debug_log, color_tolerance=None):
        """BGR template matching. Returns ALL matches above threshold (groupRectangles)."""
        needle = self._scaled_needle_color(scale)
        nw, nh = needle.shape[1], needle.shape[0]
        if nw > haystack_img.shape[1] or nh > haystack_img.shape[0]:
            return []

        hay = haystack_img if len(haystack_img.shape) == 3 else cv.cvtColor(haystack_img, cv.COLOR_GRAY2BGR)

        result = cv.matchTemplate(hay, needle, cv.TM_CCOEFF_NORMED)
        locs = list(zip(*np.where(result >= threshold)[::-1]))
        if not locs:
            return []

        # Optional color filter before groupRectangles
        if color_tolerance is not None:
            needle_mean = np.mean(needle.astype(np.float32).reshape(-1, 3), axis=0)
            filtered = []
            for loc in locs:
                x, y = int(loc[0]), int(loc[1])
                region = hay[y:y+nh, x:x+nw].astype(np.float32)
                color_dist = float(np.linalg.norm(
                    np.mean(region.reshape(-1, 3), axis=0) - needle_mean
                ))
                if color_dist <= color_tolerance:
                    filtered.append(loc)
            locs = filtered
            if not locs:
                return []

        rects = []
        for loc in locs:
            r = [int(loc[0]), int(loc[1]), nw, nh]
            rects.append(r)
            rects.append(r)

        rects, _ = cv.groupRectangles(rects, groupThreshold=1, eps=0.5)
        if not len(rects):
            return []

        points = []
        for x, y, w, h in rects:
            points.append((int(x + w // 2), int(y + h // 2), w, h))

        if debug_log:
            _log.debug("[vision:debug] [%s] Color Multi-Match: found %d (Thresh: %.2f)",
                       self.needle_name, len(points), threshold)
        return points


    # ------------------------------------------------------------------
    def _find_dt(self, haystack_img, threshold, scale, debug_log):
        """Distance Transform (Chamfer) matching. Returns single best match."""
        needle_edges = self._scaled_needle_edges(scale)
        nw, nh = needle_edges.shape[1], needle_edges.shape[0]
        if nw > haystack_img.shape[1] or nh > haystack_img.shape[0]:
            return []

        # 1. Haystack edges
        hay_gray  = _get_gray(haystack_img)
        edges_hay = cv.Canny(hay_gray, 50, 150)

        # 2. Invert: edges → 0 (DT measures distance to zero pixels)
        inv_hay = cv.bitwise_not(edges_hay)

        # 3. Distance transform map
        dist_map = cv.distanceTransform(inv_hay, cv.DIST_L2, cv.DIST_MASK_PRECISE)

        # 4. Slide binary needle edges over distance map (Chamfer score)
        needle_binary = (needle_edges / 255.0).astype(np.float32)
        match_result  = cv.matchTemplate(dist_map, needle_binary, cv.TM_CCORR)

        # 5. Best match = minimum summed distance
        min_val, _, min_loc, _ = cv.minMaxLoc(match_result)

        total_edge_pixels = np.count_nonzero(needle_binary)
        if total_edge_pixels == 0:
            return []

        avg_dist = min_val / total_edge_pixels

        # Threshold mapping: 0.8 → ≤2px avg dist, 0.5 → ≤5px
        max_allowed_dist = (1.0 - threshold) * 10.0

        if avg_dist > max_allowed_dist:
            if debug_log:
                _log.debug("[vision:debug] [%s] DT Failed: Avg Dist=%.2fpx > %.2fpx (thresh=%.2f)",
                           self.needle_name, avg_dist, max_allowed_dist, threshold)
            return []

        if debug_log:
            _log.debug("[vision:debug] [%s] Match OK! Avg Dist = %.2fpx (Cho phep <= %.2fpx)",
                       self.needle_name, avg_dist, max_allowed_dist)

        x, y = min_loc
        cx = int(x + nw // 2)
        cy = int(y + nh // 2)

        return [(cx, cy, nw, nh)]