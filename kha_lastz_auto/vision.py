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
    """Decorator: log execution time + result summary for find/exists."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        needle = getattr(args[0], 'needle_name', '?') if args else '?'
        if elapsed_ms > 20:
            # Summarize return value to make logs actionable.
            summary = ""
            if isinstance(result, bool):
                summary = " -> {}".format("true" if result else "false")
            elif isinstance(result, (list, tuple)):
                n = len(result)
                if n == 0:
                    summary = " -> 0"
                else:
                    first = result[0]
                    if isinstance(first, (list, tuple)) and len(first) >= 2:
                        summary = " -> {} (first=({},{}))".format(n, int(first[0]), int(first[1]))
                    else:
                        summary = " -> {}".format(n)
            elif result is None:
                summary = " -> None"
            else:
                summary = " -> {}".format(type(result).__name__)
            _log.debug("[timeit] Vision.%s [%s] took %.2f ms%s", func.__name__, needle, elapsed_ms, summary)
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
# Vision class — matchTemplate only (gray or BGR). No Distance Transform.
# ---------------------------------------------------------------------------
class Vision:
    """
    Template matcher using cv.matchTemplate only:
      - is_color=False (default): grayscale matchTemplate.
      - is_color=True: BGR matchTemplate (needle and haystack kept in color).

    Scale: haystack is resized to reference resolution (norm = resize(haystack, 1/scale))
    before matching; needle stays at reference size. Results are scaled back to haystack coords.
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

        if len(img.shape) == 3:
            self.needle_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            self.needle_gray = img

        _log.info("[vision:info] [%s] Load template thành công (%dx%d).",
                  self.needle_name, self.needle_w, self.needle_h)

    def _norm_haystack(self, haystack_img):
        """Resize haystack to reference resolution for matching (like original vision)."""
        scale = _global_scale
        if scale == 1.0:
            return haystack_img, 1.0
        nw = max(4, int(haystack_img.shape[1] / scale))
        nh = max(4, int(haystack_img.shape[0] / scale))
        interp = cv.INTER_AREA if scale > 1.0 else cv.INTER_LINEAR
        norm = cv.resize(haystack_img, (nw, nh), interpolation=interp)
        return norm, scale

    def match_score(self, haystack_img) -> float:
        """Return the best grayscale matchTemplate score (0..1).

        Returns -1.0 when the haystack is too small to match against the needle,
        so callers can distinguish "couldn't search" from "searched and scored low".
        """
        norm, _scale = self._norm_haystack(haystack_img)
        if self.needle_w > norm.shape[1] or self.needle_h > norm.shape[0]:
            return -1.0
        norm_gray = _get_gray(norm)
        result = cv.matchTemplate(norm_gray, self.needle_gray, self.method)
        _, max_val, _, _ = cv.minMaxLoc(result)
        return float(max_val)

    @timeit
    def exists(self, haystack_img, threshold=0.5, debug_log=False) -> bool:
        """Fast existence check — matchTemplate on grayscale only."""
        norm, scale = self._norm_haystack(haystack_img)
        if self.needle_w > norm.shape[1] or self.needle_h > norm.shape[0]:
            return False
        norm_gray = _get_gray(norm)
        result = cv.matchTemplate(norm_gray, self.needle_gray, self.method)
        _, max_val, _, _ = cv.minMaxLoc(result)
        return max_val >= threshold

    @timeit
    def find(self, haystack_img, threshold=0.5, debug_mode=None, is_color=False,
             debug_log=False, multi=False, color_tolerance=None, ratio_test=None, min_inliers=None,
             log_all_scores=False, log_scores_floor=0.3, log_scores_top_k=20):
        """
        Find needle in haystack. Returns list of (cx, cy, w, h) in haystack coords.
        is_color=False → grayscale matchTemplate.
        is_color=True  → BGR matchTemplate (color preserved).
        multi=True + is_color → return all matches (groupRectangles).
        color_tolerance → optional BGR mean distance filter when is_color=True.
        log_all_scores=True → log top-K candidate positions and scores (including below threshold)
                              for threshold tuning. Marks each ✓ (pass) or ✗ (fail).
        """
        norm, scale = self._norm_haystack(haystack_img)
        nw, nh = self.needle_w, self.needle_h
        if nw > norm.shape[1] or nh > norm.shape[0]:
            return []

        if is_color:
            if len(norm.shape) == 2:
                norm = cv.cvtColor(norm, cv.COLOR_GRAY2BGR)
            needle_bgr = self.needle_img if len(self.needle_img.shape) == 3 else \
                cv.cvtColor(self.needle_img, cv.COLOR_GRAY2BGR)
            result = cv.matchTemplate(norm, needle_bgr, self.method)
        else:
            norm_gray = _get_gray(norm)
            result = cv.matchTemplate(norm_gray, self.needle_gray, self.method)

        if log_all_scores:
            _floor = min(log_scores_floor, max(0.0, threshold - 0.3))
            _score_locs = list(zip(*np.where(result >= _floor)[::-1]))
            if not _score_locs:
                # show at least the single best
                _, _bv, _, _bl = cv.minMaxLoc(result)
                _score_locs = [_bl]
            # (x, y, score) sorted desc
            _score_pts = sorted(
                [(int(loc[0]), int(loc[1]), float(result[loc[1], loc[0]])) for loc in _score_locs],
                key=lambda t: t[2], reverse=True
            )
            # group nearby to avoid duplicate peaks (within half needle size)
            _half_nw, _half_nh = max(nw // 2, 4), max(nh // 2, 4)
            _grouped: list = []
            for x, y, sc in _score_pts:
                if not any(abs(x - gx) < _half_nw and abs(y - gy) < _half_nh for gx, gy, _ in _grouped):
                    _grouped.append((x, y, sc))
            _top = _grouped[:log_scores_top_k]
            _parts = [
                "({},{}) {:.3f}{}".format(
                    int((x + nw // 2) * scale), int((y + nh // 2) * scale),
                    sc, " \u2713" if sc >= threshold else " \u2717"
                )
                for x, y, sc in _top
            ]
            _log.info("[vision:scores] [%s] threshold=%.2f | %s",
                      self.needle_name, threshold, " | ".join(_parts))

        locs = list(zip(*np.where(result >= threshold)[::-1]))
        if not locs:
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            # CCOEFF_NORMED / CCOEFF: max la tot; SQDIFF: min la tot
            best_val = max_val if self.method in (cv.TM_CCOEFF_NORMED, cv.TM_CCOEFF, cv.TM_CCORR_NORMED, cv.TM_CCORR) else min_val
            best_loc = max_loc if self.method in (cv.TM_CCOEFF_NORMED, cv.TM_CCOEFF, cv.TM_CCORR_NORMED, cv.TM_CCORR) else min_loc
            if debug_log and is_color:
                _log.debug("[vision:debug] [%s] Color Match Failed: best=%.3f (thresh %.2f) at (%d,%d)",
                           self.needle_name, float(best_val), threshold, best_loc[0], best_loc[1])
            elif debug_log and not is_color:
                _log.debug("[vision:debug] [%s] Gray Match Failed: best=%.3f (thresh %.2f) at (%d,%d)",
                           self.needle_name, float(best_val), threshold, best_loc[0], best_loc[1])
            return []

        if is_color and not multi:
            # Single best match
            _, max_val, _, max_loc = cv.minMaxLoc(result)
            if max_val < threshold:
                return []
            x, y = max_loc
            hay = norm if len(norm.shape) == 3 else cv.cvtColor(norm, cv.COLOR_GRAY2BGR)
            needle_bgr = self.needle_img if len(self.needle_img.shape) == 3 else \
                cv.cvtColor(self.needle_img, cv.COLOR_GRAY2BGR)
            if color_tolerance is not None:
                region = hay[y:y+nh, x:x+nw].astype(np.float32)
                needle_f = needle_bgr.astype(np.float32)
                color_dist = float(np.linalg.norm(
                    np.mean(region.reshape(-1, 3), axis=0) - np.mean(needle_f.reshape(-1, 3), axis=0)
                ))
                if color_dist > color_tolerance:
                    if debug_log:
                        _log.debug("[vision:debug] [%s] Color Reject: mean BGR dist=%.1f > tol=%.1f",
                                   self.needle_name, color_dist, color_tolerance)
                    return []
            if debug_log:
                _log.debug("[vision:debug] [%s] Color Match OK! Score: %.2f", self.needle_name, max_val)
            cx = int((x + nw // 2) * scale)
            cy = int((y + nh // 2) * scale)
            return [(cx, cy, int(nw * scale), int(nh * scale))]

        # Multiple matches or gray single: use groupRectangles
        rects = []
        for loc in locs:
            r = [int(loc[0]), int(loc[1]), nw, nh]
            rects.append(r)
            rects.append(r)
        rects, _ = cv.groupRectangles(rects, groupThreshold=1, eps=0.5)
        if not len(rects):
            return []

        if is_color and color_tolerance is not None:
            hay = norm if len(norm.shape) == 3 else cv.cvtColor(norm, cv.COLOR_GRAY2BGR)
            needle_bgr = self.needle_img if len(self.needle_img.shape) == 3 else \
                cv.cvtColor(self.needle_img, cv.COLOR_GRAY2BGR)
            needle_mean = np.mean(needle_bgr.astype(np.float32).reshape(-1, 3), axis=0)
            filtered = []
            for (x, y, w, h) in rects:
                region = hay[y:y+h, x:x+w].astype(np.float32)
                color_dist = float(np.linalg.norm(
                    np.mean(region.reshape(-1, 3), axis=0) - needle_mean
                ))
                if color_dist <= color_tolerance:
                    filtered.append((x, y, w, h))
            rects = [(x, y, w, h) for x, y, w, h in filtered] if filtered else []
            if not rects:
                return []

        points = []
        for x, y, w, h in rects:
            cx = int((x + w // 2) * scale)
            cy = int((y + h // 2) * scale)
            points.append((cx, cy, int(w * scale), int(h * scale)))

        if debug_mode in ("rectangles", "points"):
            for x, y, w, h in rects:
                ox = int(x * scale)
                oy = int(y * scale)
                ow = int(w * scale)
                oh = int(h * scale)
                if debug_mode == "rectangles":
                    cv.rectangle(haystack_img, (ox, oy), (ox + ow, oy + oh), (0, 255, 0), 2)
                else:
                    cv.drawMarker(haystack_img, (ox + ow // 2, oy + oh // 2),
                                 (255, 0, 255), cv.MARKER_CROSS, 40, 2)
            try:
                cv.imshow("Matches", haystack_img)
            except cv.error:
                pass

        if debug_log and points:
            _log.debug("[vision:debug] [%s] Match: %d (Thresh: %.2f)", self.needle_name, len(points), threshold)
        return points

    def find_multi_with_scores(self, haystack_img, threshold: float = 0.5,
                               is_color: bool = False, color_tolerance=None) -> list:
        """
        Like find(multi=True) but returns (cx, cy, w, h, score) tuples sorted by score desc.
        Use for selecting the top-N best matches (e.g. yellow trucks).
        """
        norm, scale = self._norm_haystack(haystack_img)
        nw, nh = self.needle_w, self.needle_h
        if nw > norm.shape[1] or nh > norm.shape[0]:
            return []

        if is_color:
            if len(norm.shape) == 2:
                norm = cv.cvtColor(norm, cv.COLOR_GRAY2BGR)
            needle_bgr = self.needle_img if len(self.needle_img.shape) == 3 else \
                cv.cvtColor(self.needle_img, cv.COLOR_GRAY2BGR)
            result = cv.matchTemplate(norm, needle_bgr, self.method)
        else:
            norm_gray = _get_gray(norm)
            result = cv.matchTemplate(norm_gray, self.needle_gray, self.method)

        locs = list(zip(*np.where(result >= threshold)[::-1]))
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

        # Optional color_tolerance filter
        if is_color and color_tolerance is not None:
            hay = norm if len(norm.shape) == 3 else cv.cvtColor(norm, cv.COLOR_GRAY2BGR)
            needle_bgr = self.needle_img if len(self.needle_img.shape) == 3 else \
                cv.cvtColor(self.needle_img, cv.COLOR_GRAY2BGR)
            needle_mean = np.mean(needle_bgr.astype(np.float32).reshape(-1, 3), axis=0)
            filtered = []
            for (x, y, w, h) in rects:
                region = hay[y:y+h, x:x+w].astype(np.float32)
                dist = float(np.linalg.norm(np.mean(region.reshape(-1, 3), axis=0) - needle_mean))
                if dist <= color_tolerance:
                    filtered.append((x, y, w, h))
            rects = filtered
            if not rects:
                return []

        results: list = []
        for (x, y, w, h) in rects:
            # Max score within the grouped region
            r_h = min(h, result.shape[0] - y)
            r_w = min(w, result.shape[1] - x)
            if r_h > 0 and r_w > 0:
                score = float(np.max(result[y:y + r_h, x:x + r_w]))
            else:
                score = float(result[max(0, y), max(0, x)])
            cx = int((x + w // 2) * scale)
            cy = int((y + h // 2) * scale)
            results.append((cx, cy, int(w * scale), int(h * scale), score))

        results.sort(key=lambda t: t[4], reverse=True)
        return results