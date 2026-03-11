import cv2 as cv
import numpy as np
import logging
import threading
import time
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

        # Init SIFT and FLANN
        # BỎ giới hạn nfeatures: vì ảnh template thì nhỏ, nhưng ảnh màn hình (haystack) rất to 1080p.
        # Nếu giới hạn 500, SIFT sẽ chỉ lấy 500 điểm đặc trưng trên TOÀN MÀN HÌNH, dẫn đến việc bỏ sót hoàn toàn cái nút bé xíu!
        self.sift = cv.SIFT_create(nfeatures=1000)
        self.needle_kp, self.needle_des = self.sift.detectAndCompute(self.needle_gray, None)
        
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=30) # Trả lại 50 checks để đảm bảo tính chính xác
        self.flann = cv.FlannBasedMatcher(index_params, search_params)

    def exists(self, haystack_img, min_match_count=10) -> bool:
        """Kiem tra xem co the tim thay object hay khong bang SIFT."""
        start_t = time.time()
        
        # Không cần resize haystack_img nữa vì SIFT có tính chất Scale-Invariant.
        # Resize ảnh to tốn CPU hơn cả việc chạy SIFT.
        # Tuy nhiên, vẫn cần chuyển sang grayscale.
        if len(haystack_img.shape) == 3:
            haystack_gray = cv.cvtColor(haystack_img, cv.COLOR_BGR2GRAY)
        else:
            haystack_gray = haystack_img

        if self.needle_w > haystack_gray.shape[1] or self.needle_h > haystack_gray.shape[0]:
            return False

        kp, des = self.sift.detectAndCompute(haystack_gray, None)

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

        elapsed_ms = (time.time() - start_t) * 1000
        _log.debug(f"[SIFT exists] {self.needle_img_path if hasattr(self, 'needle_img_path') else 'template'} - Elapsed: {elapsed_ms:.1f}ms")
        return good_matches >= min_match_count

    def find(self, haystack_img, min_match_count=10, debug_mode=None, is_color=False):
        """Tim vi tri cua ONE best match bang SIFT."""
        start_t = time.time()
        
        scale = _global_scale
        # Save the color input for color matching later
        haystack_color = haystack_img
        
        # Không cần resize haystack_img nữa vì SIFT có tính chất Scale-Invariant.
        # Resize ảnh to tốn CPU hơn cả việc chạy SIFT.
        # Tuy nhiên, vẫn cần chuyển sang grayscale.
        if len(haystack_img.shape) == 3:
            haystack_gray = cv.cvtColor(haystack_img, cv.COLOR_BGR2GRAY)
        else:
            haystack_gray = haystack_img

        if self.needle_w > haystack_gray.shape[1] or self.needle_h > haystack_gray.shape[0]:
            if debug_mode: _log.debug(f"[SIFT find] {self.needle_img_path if hasattr(self, 'needle_img_path') else 'template'} - Elapsed (fast reject): {(time.time() - start_t)*1000:.1f}ms")
            return []

        kp, des = self.sift.detectAndCompute(haystack_gray, None)

        if des is None or self.needle_des is None:
            if debug_mode: _log.debug(f"[SIFT find] {self.needle_img_path if hasattr(self, 'needle_img_path') else 'template'} - Elapsed (no features): {(time.time() - start_t)*1000:.1f}ms")
            return []
            
        if len(des) < 2 or len(self.needle_des) < 2:
            if debug_mode: _log.debug(f"[SIFT find] {self.needle_img_path if hasattr(self, 'needle_img_path') else 'template'} - Elapsed (<2 features): {(time.time() - start_t)*1000:.1f}ms")
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

        points = []
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
                
                # ------ BẮT ĐẦU KIỂM TRA CHỐNG NHẬN DIỆN SAI (FALSE POSITIVE) ------
                # Do SIFT có thể nhặt điểm rác từ nhiều nơi ráp lại thành hình kỳ dị
                dst_int = np.int32(dst)
                if not cv.isContourConvex(dst_int):
                    if debug_mode: _log.debug("[SIFT] False positive rejected (Not convex)")
                    return []
                    
                area = cv.contourArea(dst_int)
                needle_area = h * w
                # Với UI Game, kích thước scale thường chỉ dao động từ 0.5x đến 2x. 
                # Diện tích bình phương nên nằm trong khoảng 0.25x -> 4x
                if area < needle_area * 0.2 or area > needle_area * 5.0:
                    if debug_mode: _log.debug(f"[SIFT] False positive rejected (Area {area} vs {needle_area})")
                    return []
                # --------------------------------------------------------------------
                
                # Tính tâm của bounding box
                center_x = int(np.mean(dst[:, 0, 0]))
                center_y = int(np.mean(dst[:, 0, 1]))
                
                # Trả về tọa độ pixel ĐÃ ĐƯỢC SCALE để bot_engine click đúng
                if scale != 1.0:
                    center_x = int(center_x * scale)
                    center_y = int(center_y * scale)
                
                points.append((center_x, center_y))
                
                if debug_mode in ("rectangles", "points"):
                    # Vẽ viền tìm đc
                    if debug_mode == "rectangles":
                        dst_scaled = np.int32(dst * scale) if scale != 1.0 else np.int32(dst)
                        cv.polylines(haystack_img, [dst_scaled], True, (0, 255, 0), 3, cv.LINE_AA)
                    elif debug_mode == "points":
                        cv.drawMarker(haystack_img, (center_x, center_y), (255, 0, 255), cv.MARKER_CROSS, 40, 2)
                    cv.imshow("Matches", haystack_img)
                    
                if is_color and len(haystack_color.shape) == 3:
                    # Trích xuất màu ở trung tâm match (trên ảnh đã scale)
                    crop_w, crop_h = 10, 10
                    rx = max(0, center_x - crop_w // 2)
                    ry = max(0, center_y - crop_h // 2)
                    crop = haystack_color[ry:ry+crop_h, rx:rx+crop_w]
                    if crop.size > 0:
                        mean_color = cv.mean(crop)[:3] # B, G, R
                        if debug_mode: _log.debug(f"[SIFT find] {self.needle_img_path if hasattr(self, 'needle_img_path') else 'template'} - Elapsed (color): {(time.time() - start_t)*1000:.1f}ms")
                        return [(center_x, center_y)], [{"region_bgr": mean_color}]
                
                if debug_mode: _log.debug(f"[SIFT find] {self.needle_img_path if hasattr(self, 'needle_img_path') else 'template'} - Elapsed (found): {(time.time() - start_t)*1000:.1f}ms")
                return [(center_x, center_y)]
                
        if debug_mode: _log.debug(f"[SIFT find] {self.needle_img_path if hasattr(self, 'needle_img_path') else 'template'} - Elapsed (not found): {(time.time() - start_t)*1000:.1f}ms")
        if is_color: return [], []
        return []