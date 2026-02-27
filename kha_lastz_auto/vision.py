import cv2 as cv
import numpy as np

# Scale toan cuc: duoc set 1 lan duy nhat tu main.py dua tren
# ty le current_window_width / reference_width tu config.yaml.
# Tat ca Vision instances dung chung, khong co scan, khong co slow path.
_global_scale = 1.0


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

        # Single matchTemplate tren normalized screenshot
        result = cv.matchTemplate(norm, self.needle_img, self.method)
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
            points.append((cx, cy))

            if debug_mode == 'rectangles':
                ox, oy = int(x * scale), int(y * scale)
                cv.rectangle(haystack_img, (ox, oy),
                             (ox + int(w * scale), oy + int(h * scale)),
                             color=(0, 255, 0), lineType=cv.LINE_4, thickness=2)
            elif debug_mode == 'points':
                cv.drawMarker(haystack_img, (cx, cy), color=(255, 0, 255),
                              markerType=cv.MARKER_CROSS, markerSize=40, thickness=2)

        if debug_mode:
            cv.imshow('Matches', haystack_img)

        return points
