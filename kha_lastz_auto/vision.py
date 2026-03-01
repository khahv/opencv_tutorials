import cv2 as cv
import numpy as np
import logging
import threading
_log = logging.getLogger("kha_lastz")


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


# Scale toan cuc: duoc set 1 lan duy nhat tu main.py dua tren
# ty le current_window_width / reference_width tu config.yaml.
# Tat ca Vision instances dung chung, khong co scan, khong co slow path.
_global_scale = 1.0

# Per-frame gray cache: converting 2400x1600 BGR→Gray is expensive; cache so all
# find() calls within one frame share the same gray image instead of re-converting.
# Use threading.local() so each thread has its own independent cache — avoids race
# conditions when the detector background thread and main thread call find() concurrently.
_gray_cache = threading.local()


def _get_gray(img):
    # Use identity check (is) + keep a reference to src so Python cannot reuse
    # its memory address for a new array while the cache is still valid.
    # id()-only check is unsafe: src goes out of scope → memory freed → new array
    # allocated at same address → stale gray returned for wrong frame.
    if getattr(_gray_cache, 'src', None) is img:
        return _gray_cache.img
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _gray_cache.src = img  # keep reference alive to prevent address reuse
    _gray_cache.img = gray
    return gray


def set_global_scale(scale):
    global _global_scale
    _global_scale = float(scale)


def get_global_scale():
    return _global_scale


class Vision:

    needle_img = None
    needle_w = 0
    needle_h = 0
    method = None

    def __init__(self, needle_img_path, method=cv.TM_CCOEFF_NORMED):
        self.needle_img = cv.imread(needle_img_path, cv.IMREAD_UNCHANGED)
        if self.needle_img is None:
            raise FileNotFoundError('Khong tim thay anh: {}'.format(needle_img_path))
        if len(self.needle_img.shape) == 3 and self.needle_img.shape[2] == 4:
            self.needle_img = cv.cvtColor(self.needle_img, cv.COLOR_BGRA2BGR)

        self.needle_w = self.needle_img.shape[1]
        self.needle_h = self.needle_img.shape[0]
        self.method = method
        # Pre-convert needle to grayscale — matchTemplate on 1-channel is ~3x faster than BGR
        if len(self.needle_img.shape) == 3:
            self.needle_gray = cv.cvtColor(self.needle_img, cv.COLOR_BGR2GRAY)
        else:
            self.needle_gray = self.needle_img

    def exists(self, haystack_img, threshold=0.5) -> bool:
        """Nhanh hon find(): chi kiem tra co match hay khong, khong tinh vi tri, khong groupRectangles."""
        scale = _global_scale
        if scale != 1.0:
            nw = max(4, int(haystack_img.shape[1] / scale))
            nh = max(4, int(haystack_img.shape[0] / scale))
            interp = cv.INTER_AREA if scale > 1.0 else cv.INTER_LINEAR
            norm = cv.resize(haystack_img, (nw, nh), interpolation=interp)
        else:
            norm = haystack_img
        if self.needle_w > norm.shape[1] or self.needle_h > norm.shape[0]:
            return False
        norm_gray = _get_gray(norm)
        result = cv.matchTemplate(norm_gray, self.needle_gray, self.method)
        _, max_val, _, _ = cv.minMaxLoc(result)
        return max_val >= threshold

    def find_color(self, haystack_img, threshold=0.5, debug_mode=None,
                   color_tolerance=None, min_saturation=None,
                   hue_range=None, hue_min_fraction=0.4, log_match_score=False):
        """
        Same as find() but matches on full BGR (3-channel) instead of grayscale.
        Use when color is the key discriminator (e.g. a yellow truck vs grey truck).
        ~3x slower than find() due to 3-channel matchTemplate, but more accurate.

        color_tolerance: BGR Euclidean mean-color distance cap.
          TM_CCOEFF_NORMED measures pattern correlation (not absolute color) so a
          grey truck with similar texture can still score high.  This check rejects
          candidates whose mean BGR is too far from the template's mean BGR.
          Typical value: 50–80.

        min_saturation: HSV saturation minimum (0–255).
          Rejects unsaturated (grey/white) regions.
          Yellow S≈135, grey S≈30.  Typical value: 80.

        hue_range: [min_h, max_h] in OpenCV HSV hue units (0–179).
          Filters by hue of saturated pixels (S >= 20).
          Two sub-checks, both must pass:
            1. Median hue of saturated pixels must be within [min_h, max_h].
               (Median is robust to flame/shadow outlier pixels unlike mean.)
            2. At least hue_min_fraction of saturated pixels must fall in range.
          Yellow H peak H=4-22, median≈17.  Purple H≈120.
          Example for yellow: hue_range=[3, 30], hue_min_fraction=0.4
        """
        scale = _global_scale
        if scale != 1.0:
            nw = max(4, int(haystack_img.shape[1] / scale))
            nh = max(4, int(haystack_img.shape[0] / scale))
            interp = cv.INTER_AREA if scale > 1.0 else cv.INTER_LINEAR
            norm = cv.resize(haystack_img, (nw, nh), interpolation=interp)
        else:
            norm = haystack_img

        # Ensure both are 3-channel BGR
        if len(norm.shape) == 2:
            norm = cv.cvtColor(norm, cv.COLOR_GRAY2BGR)
        needle_bgr = self.needle_img if len(self.needle_img.shape) == 3 else \
            cv.cvtColor(self.needle_img, cv.COLOR_GRAY2BGR)

        if self.needle_w > norm.shape[1] or self.needle_h > norm.shape[0]:
            return []

        result = cv.matchTemplate(norm, needle_bgr, self.method)
        locs = list(zip(*np.where(result >= threshold)[::-1]))
        if not locs:
            return []

        rects = []
        for loc in locs:
            r = [int(loc[0]), int(loc[1]), self.needle_w, self.needle_h]
            rects.append(r)
            rects.append(r)
        rects, _ = cv.groupRectangles(rects, groupThreshold=1, eps=0.5)
        if not len(rects):
            return []

        # Pre-compute template mean color once (used for color_tolerance check)
        if color_tolerance is not None:
            tpl_mean = np.mean(needle_bgr.reshape(-1, 3), axis=0).astype(float)

        points = []
        meta_list = []  # [{score, color_dist, mean_sat, median_hue, in_range_frac}, ...]
        for x, y, w, h in rects:
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(norm.shape[1], x + w)
            y2 = min(norm.shape[0], y + h)
            region = norm[y1:y2, x1:x2]
            if region.size == 0:
                continue

            _color_dist = _mean_sat = _median_hue = _in_range_frac = None  # for log_match_score
            _region_bgr = None  # mean BGR of region, for logging

            if color_tolerance is not None or log_match_score:
                region_mean = np.mean(region.reshape(-1, 3), axis=0).astype(float)
                if log_match_score:
                    _region_bgr = (float(region_mean[0]), float(region_mean[1]), float(region_mean[2]))
            if color_tolerance is not None:
                _color_dist = float(np.linalg.norm(tpl_mean - region_mean))
                if _color_dist > color_tolerance:
                    if debug_mode:
                        _log.info("[Vision.find_color] reject ({},{}) color_dist={:.1f} > tolerance={}".format(
                            x, y, _color_dist, color_tolerance))
                    continue
                if debug_mode:
                    _log.info("[Vision.find_color] accept ({},{}) color_dist={:.1f} <= tolerance={}".format(
                        x, y, _color_dist, color_tolerance))

            if min_saturation is not None or hue_range is not None:
                region_bgr = region if len(region.shape) == 3 else cv.cvtColor(region, cv.COLOR_GRAY2BGR)
                region_hsv = cv.cvtColor(region_bgr, cv.COLOR_BGR2HSV)

                if min_saturation is not None:
                    _mean_sat = float(np.mean(region_hsv[:, :, 1]))
                    if _mean_sat < min_saturation:
                        if debug_mode:
                            _log.info("[Vision.find_color] reject ({},{}) saturation={:.1f} < min={}".format(
                                x, y, _mean_sat, min_saturation))
                        continue
                    if debug_mode:
                        _log.info("[Vision.find_color] accept ({},{}) saturation={:.1f} >= min={}".format(
                            x, y, _mean_sat, min_saturation))

                if hue_range is not None:
                    min_h, max_h = hue_range[0], hue_range[1]
                    sat_mask = region_hsv[:, :, 1] >= 20
                    sat_hues = region_hsv[:, :, 0][sat_mask] if sat_mask.any() else region_hsv[:, :, 0].ravel()
                    _median_hue = float(np.median(sat_hues))
                    _in_range_frac = float(np.mean((sat_hues >= min_h) & (sat_hues <= max_h)))
                    if not (min_h <= _median_hue <= max_h) or _in_range_frac < hue_min_fraction:
                        if debug_mode:
                            _log.info("[Vision.find_color] reject ({},{}) hue_median={:.1f} in_range={:.0f}% (need [{},{}] frac>={:.0f}%)".format(
                                x, y, _median_hue, _in_range_frac * 100, min_h, max_h, hue_min_fraction * 100))
                        continue
                    if debug_mode:
                        _log.info("[Vision.find_color] accept ({},{}) hue_median={:.1f} in_range={:.0f}%".format(
                            x, y, _median_hue, _in_range_frac * 100))

            score = float(result[y, x]) if 0 <= y < result.shape[0] and 0 <= x < result.shape[1] else 0.0
            if log_match_score:
                _dominant_hue = _median_hue
                if _dominant_hue is None and _region_bgr is not None:
                    region_bgr = region if len(region.shape) == 3 else cv.cvtColor(region, cv.COLOR_GRAY2BGR)
                    region_hsv = cv.cvtColor(region_bgr, cv.COLOR_BGR2HSV)
                    sat_mask = region_hsv[:, :, 1] >= 20
                    sat_hues = region_hsv[:, :, 0][sat_mask] if sat_mask.any() else region_hsv[:, :, 0].ravel()
                    _dominant_hue = float(np.median(sat_hues)) if sat_hues.size else None
                _dominant_color = _hue_to_dominant_color(_dominant_hue) if _dominant_hue is not None else None
                meta_list.append({
                    "score": score,
                    "color_dist": _color_dist,
                    "mean_sat": _mean_sat,
                    "median_hue": _median_hue,
                    "in_range_frac": _in_range_frac,
                    "region_bgr": _region_bgr,
                    "dominant_color": _dominant_color,
                })

            cx = int((x + w // 2) * scale)
            cy = int((y + h // 2) * scale)
            if debug_mode:
                _log.info("[Vision.find_color] groupRect=({},{},{},{}) needle=({}x{}) scale={} → center=({},{})".format(
                    x, y, w, h, self.needle_w, self.needle_h, scale, cx, cy))
            points.append((cx, cy))
        if log_match_score and meta_list:
            return points, meta_list
        return points

    def find(self, haystack_img, threshold=0.5, debug_mode=None):
        """
        Normalize screenshot ve kich thuoc goc cua template roi match 1 lan.
        Scale duoc tinh truoc tu config (reference_width), khong co scan.
        Toa do tra ve la toa do goc trong haystack_img chua normalize.
        """
        scale = _global_scale

        # Normalize screenshot: thu nho (hoac phong to) de template vua khop
        if scale != 1.0:
            nw = max(4, int(haystack_img.shape[1] / scale))
            nh = max(4, int(haystack_img.shape[0] / scale))
            interp = cv.INTER_AREA if scale > 1.0 else cv.INTER_LINEAR
            norm = cv.resize(haystack_img, (nw, nh), interpolation=interp)
        else:
            norm = haystack_img

        if self.needle_w > norm.shape[1] or self.needle_h > norm.shape[0]:
            return []

        # Convert haystack to grayscale — use per-frame cache so multiple find() calls
        # on the same screenshot share one conversion instead of repeating it.
        norm_gray = _get_gray(norm)

        # Single matchTemplate tren normalized screenshot (grayscale)
        result = cv.matchTemplate(norm_gray, self.needle_gray, self.method)
        locs = list(zip(*np.where(result >= threshold)[::-1]))
        if not locs:
            return []

        # groupRectangles de khu overlap
        rects = []
        for loc in locs:
            r = [int(loc[0]), int(loc[1]), self.needle_w, self.needle_h]
            rects.append(r)
            rects.append(r)
        rects, _ = cv.groupRectangles(rects, groupThreshold=1, eps=0.5)
        if not len(rects):
            return []

        # Doi toa do tu normalized ve goc (nhan voi scale)
        points = []
        for x, y, w, h in rects:
            cx = int((x + w // 2) * scale)
            cy = int((y + h // 2) * scale)
            if debug_mode:
                _log.info("[Vision.find] groupRect=({},{},{},{}) needle=({}x{}) scale={} → center=({},{})".format(
                    x, y, w, h, self.needle_w, self.needle_h, scale, cx, cy))
            points.append((cx, cy))

            if debug_mode == 'rectangles':
                ox, oy = int(x * scale), int(y * scale)
                cv.rectangle(haystack_img, (ox, oy),
                             (ox + int(w * scale), oy + int(h * scale)),
                             color=(0, 255, 0), lineType=cv.LINE_4, thickness=2)
            elif debug_mode == 'points':
                cv.drawMarker(haystack_img, (cx, cy), color=(255, 0, 255),
                              markerType=cv.MARKER_CROSS, markerSize=40, thickness=2)

        if debug_mode in ('rectangles', 'points'):
            cv.imshow('Matches', haystack_img)

        return points
