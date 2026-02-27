import cv2 as cv
import numpy as np

# Danh sach scale mac dinh: tu 50% den 200% template goc.
# Vision tu dong thu tung scale, khong can cau hinh o YAML.
_DEFAULT_SCALES = [
    0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
    1.00,
    1.05, 1.10, 1.15, 1.20, 1.30, 1.40, 1.50, 1.70, 2.00,
]


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

        # Cache scale tot nhat tim duoc lan truoc.
        # Fast path: thu scale nay truoc; neu tim duoc -> tra ve ngay.
        # Slow path: scan toan bo _DEFAULT_SCALES khi cache fail.
        self._cached_scale = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find(self, haystack_img, threshold=0.5, debug_mode=None):
        """
        Tim template trong haystack_img o nhieu scale.

        - Lan dau (hoac khi scale thay doi): thu toan bo _DEFAULT_SCALES,
          cache scale cho ket qua tot nhat.
        - Lan sau: thu cached scale truoc (fast path); fallback full scan
          neu khong tim thay.
        Khong can tham so 'scales' o YAML — tu dong thich nghi.
        """
        # ── Fast path: thu cached scale truoc ──────────────────────────
        if self._cached_scale is not None:
            pts = self._scan_one_scale(haystack_img, self._cached_scale, threshold)
            if pts:
                return self._apply_debug(haystack_img, pts, debug_mode)

        # ── Slow path: quet toan bo scales, NMS, cache ket qua ─────────
        candidates = []  # (score, cx, cy, scale)
        for scale in _DEFAULT_SCALES:
            if scale == self._cached_scale:
                continue  # da thu roi
            raw = self._raw_candidates(haystack_img, scale, threshold)
            candidates.extend(raw)

        if not candidates:
            # Khong tim duoc gi -> xoa cache (co the DPI thay doi tiep)
            self._cached_scale = None
            return []

        # NMS: loai diem trung lap, giu diem score cao nhat
        pts = self._nms(candidates)

        # Cap nhat cache: scale cua diem co score cao nhat
        best = max(candidates, key=lambda c: c[0])
        self._cached_scale = best[3]

        return self._apply_debug(haystack_img, pts, debug_mode)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_needle_at_scale(self, scale):
        if scale == 1.0:
            return self.needle_img, self.needle_w, self.needle_h
        nw = max(4, int(self.needle_w * scale))
        nh = max(4, int(self.needle_h * scale))
        interp = cv.INTER_AREA if scale < 1.0 else cv.INTER_LINEAR
        return cv.resize(self.needle_img, (nw, nh), interpolation=interp), nw, nh

    def _raw_candidates(self, haystack_img, scale, threshold):
        """Tra ve list (score, cx, cy, scale) cho 1 scale cu the."""
        needle, nw, nh = self._get_needle_at_scale(scale)
        if nw > haystack_img.shape[1] or nh > haystack_img.shape[0]:
            return []
        result = cv.matchTemplate(haystack_img, needle, self.method)
        ys, xs = np.where(result >= threshold)
        out = []
        for y, x in zip(ys, xs):
            out.append((float(result[y, x]), int(x) + nw // 2, int(y) + nh // 2, scale))
        return out

    def _scan_one_scale(self, haystack_img, scale, threshold):
        """Tim o 1 scale, tra ve list (cx, cy) dung groupRectangles."""
        needle, nw, nh = self._get_needle_at_scale(scale)
        if nw > haystack_img.shape[1] or nh > haystack_img.shape[0]:
            return []
        result = cv.matchTemplate(haystack_img, needle, self.method)
        locs = list(zip(*np.where(result >= threshold)[::-1]))
        if not locs:
            return []
        rects = []
        for loc in locs:
            r = [int(loc[0]), int(loc[1]), nw, nh]
            rects.append(r)
            rects.append(r)
        rects, _ = cv.groupRectangles(rects, groupThreshold=1, eps=0.5)
        return [(x + w // 2, y + h // 2) for x, y, w, h in rects] if len(rects) else []

    def _nms(self, candidates):
        """Non-Maximum Suppression tren list (score, cx, cy, scale)."""
        suppress_dist = max(self.needle_w, self.needle_h) * 0.4
        candidates = sorted(candidates, key=lambda c: -c[0])
        kept = []
        for score, cx, cy, scale in candidates:
            if not any(max(abs(cx - kx), abs(cy - ky)) < suppress_dist
                       for _, kx, ky, _ in kept):
                kept.append((score, cx, cy, scale))
        return [(cx, cy) for _, cx, cy, _ in kept]

    def _apply_debug(self, haystack_img, points, debug_mode):
        if debug_mode and points:
            for cx, cy in points:
                if debug_mode == 'rectangles':
                    hw, hh = self.needle_w // 2, self.needle_h // 2
                    cv.rectangle(haystack_img, (cx - hw, cy - hh), (cx + hw, cy + hh),
                                 color=(0, 255, 0), lineType=cv.LINE_4, thickness=2)
                elif debug_mode == 'points':
                    cv.drawMarker(haystack_img, (cx, cy), color=(255, 0, 255),
                                  markerType=cv.MARKER_CROSS, markerSize=40, thickness=2)
        if debug_mode:
            cv.imshow('Matches', haystack_img)
        return points
