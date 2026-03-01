import cv2 as cv
import numpy as np
import logging
import threading
_log = logging.getLogger("kha_lastz")

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
