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

    def __init__(self, needle_img_path, method=cv.TM_CCOEFF_NORMED):
        # We ignore 'method' since we use SIFT now, keeping it for backward compatibility of signature
        self.needle_img_path = needle_img_path
        self.needle_img = cv.imread(needle_img_path, cv.IMREAD_UNCHANGED)
        if self.needle_img is None:
            raise FileNotFoundError('Khong tim thay anh: {}'.format(needle_img_path))
        if len(self.needle_img.shape) == 3 and self.needle_img.shape[2] == 4:
            self.needle_img = cv.cvtColor(self.needle_img, cv.COLOR_BGRA2BGR)

        self.needle_w = self.needle_img.shape[1]
        self.needle_h = self.needle_img.shape[0]

        if len(self.needle_img.shape) == 3:
            self.needle_gray = cv.cvtColor(self.needle_img, cv.COLOR_BGR2GRAY)
        else:
            self.needle_gray = self.needle_img

        # Initialize SIFT
        self.sift = cv.SIFT_create()
        self.needle_kp, self.needle_des = self.sift.detectAndCompute(self.needle_gray, None)
        
        # FLANN Matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv.FlannBasedMatcher(index_params, search_params)

    def exists(self, haystack_img, min_match_count=10) -> bool:
        """Kiem tra xem co the tim thay object hay khong bang SIFT."""
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
        kp, des = self.sift.detectAndCompute(norm_gray, None)

        if des is None or self.needle_des is None:
            return False
            
        if len(des) < 2 or len(self.needle_des) < 2:
            return False

        matches = self.flann.knnMatch(self.needle_des, des, k=2)

        if not matches:
            return False
            
        good_matches = 0
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.7 * n.distance:
                    good_matches += 1

        return good_matches >= min_match_count

    def find(self, haystack_img, min_match_count=10, debug_mode=None, is_color=False):
        """Tim vi tri cua ONE best match bang SIFT."""
        scale = _global_scale
        # Normalize screenshot
        if scale != 1.0:
            nw = max(4, int(haystack_img.shape[1] / scale))
            nh = max(4, int(haystack_img.shape[0] / scale))
            interp = cv.INTER_AREA if scale > 1.0 else cv.INTER_LINEAR
            norm = cv.resize(haystack_img, (nw, nh), interpolation=interp)
        else:
            norm = haystack_img

        if self.needle_w > norm.shape[1] or self.needle_h > norm.shape[0]:
            return []

        # Convert to grayscale if not already
        norm_gray = _get_gray(norm)

        kp, des = self.sift.detectAndCompute(norm_gray, None)

        if des is None or self.needle_des is None:
            return []
            
        if len(des) < 2 or len(self.needle_des) < 2:
            return []

        matches = self.flann.knnMatch(self.needle_des, des, k=2)

        # Lowe's ratio test
        good = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.7 * n.distance:
                    good.append(m)

        if debug_mode:
            _log.info(f"[SIFT] {self.needle_img_path if hasattr(self, 'needle_img_path') else 'template'} -> {len(good)}/{min_match_count} good matches.")

        if len(good) >= min_match_count:
            # Lấy tọa độ các điểm match
            src_pts = np.float32([self.needle_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # Tính toán Homography matrix để tìm vùng chứa object
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            
            if M is not None:
                if debug_mode:
                    _log.debug(f"[SIFT] Homography matrix found. Emitting point.")
                # Tọa độ 4 góc của template
                h, w = self.needle_gray.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                
                # Ánh xạ 4 góc sang viền trên hình to
                dst = cv.perspectiveTransform(pts, M)
                
                # Tính tâm của vùng được match (theo tọa độ đã scale 'norm')
                cx_norm = int(np.mean(dst[:, 0, 0]))
                cy_norm = int(np.mean(dst[:, 0, 1]))
                
                # Revert lại tọa độ original nếu có global_scale
                cx = int(cx_norm * scale)
                cy = int(cy_norm * scale)
                
                if debug_mode in ("rectangles", "points"):
                    # Vẽ viền tìm đc
                    if debug_mode == "rectangles":
                        dst_scaled = np.int32(dst * scale)
                        cv.polylines(haystack_img, [dst_scaled], True, (0, 255, 0), 3, cv.LINE_AA)
                    elif debug_mode == "points":
                        cv.drawMarker(haystack_img, (cx, cy), (255, 0, 255), cv.MARKER_CROSS, 40, 2)
                    cv.imshow("Matches", haystack_img)
                    
                return [(cx, cy)]
                
        return []