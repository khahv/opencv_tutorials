import cv2 as cv
import numpy as np
import logging
import threading
import time
import os
import random
import yaml
from pathlib import Path
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
    # ------------------------------------------------------------------ #
    # YOLO shared model (optional) — load một lần, dùng cho tất cả instance
    # ------------------------------------------------------------------ #
    _yolo_model   = None   # ultralytics YOLO instance
    _yolo_classes = {}     # {class_name: class_id}
    _auto_label   = False  # Global toggle for SIFT-to-YOLO bootstrapping

    @classmethod
    def load_yolo_model(cls, model_path: str, conf: float = 0.25):
        """Load YOLOv8 model. Gọi 1 lần trong main.py trước khi tạo vision cache."""
        try:
            from ultralytics import YOLO
            _log.info(f"[YOLO] Loading model: {model_path}")
            model = YOLO(model_path)
            model.conf = conf
            cls._yolo_model = model
            cls._yolo_classes = {v: k for k, v in model.names.items()}  # name → id
            _log.info(f"[YOLO] Model ready. Classes: {list(model.names.values())}")
        except Exception as e:
            _log.error(f"[YOLO] Failed to load model '{model_path}': {e}")

    @classmethod
    def _ensure_yolo_classes(cls):
        """Load class names from data.yaml if _yolo_classes is empty."""
        if cls._yolo_classes:
            return
        data_yaml_path = os.path.join(os.path.dirname(__file__), "yolo_dataset", "data.yaml")
        if os.path.exists(data_yaml_path):
            try:
                with open(data_yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    names = data.get("names", {})
                    # Chuyển từ {id: name} sang {name: id}
                    cls._yolo_classes = {str(v): int(k) for k, v in names.items()}
                    _log.info(f"[Auto-Label] Loaded {len(cls._yolo_classes)} classes from data.yaml")
            except Exception as e:
                _log.error(f"[Auto-Label] Failed to load data.yaml: {e}")

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

        # Init SIFT
        # nfeatures=2000: giới hạn number of features để SIFT không phải tính vô hạn
        self.sift = cv.SIFT_create()
        self.needle_kp, self.needle_des = self.sift.detectAndCompute(self.needle_gray, None)
        
        # Auto-select Matcher: BFMatcher nhanh hơn FLANN khi template nhỏ (ít keypoints)
        # vì FLANN xây KD-tree tốn chi phí khởi tạo cao hơn brute-force trực tiếp.
        # Ngưỡng 300: nếu template có <= 300 keypoints → dùng BFMatcher, ngược lại dùng FLANN.
        needle_kp_count = len(self.needle_kp) if self.needle_kp is not None else 0
        _log.debug(f"[SIFT] {self.needle_img_path} → {needle_kp_count} keypoints → using {'BFMatcher' if needle_kp_count <= 300 else 'FLANN'}")
        if needle_kp_count <= 300:
            # BFMatcher với NORM_L2 (chuẩn cho SIFT/SURF float descriptor)
            self.matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
            self._use_bf = True
        else:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=100)  # Độ chính xác cao: tăng lên 100 (mặc định 50)
            self.matcher = cv.FlannBasedMatcher(index_params, search_params)
            self._use_bf = False

    def _yolo_find(self, haystack_img, conf=None, roi=None):
        """YOLO inference: trả về [(cx, cy), ...] của tất cả detection ứng với class này."""
        model = self._yolo_model
        if model is None:
            return []
        start_t = time.time()

        # Luôn infer trên ảnh gốc (fullscreen) vì YOLO đã học tỷ lệ nút bấm dựa trên ảnh gốc.
        # Nếu crop nhỏ lại, nút sẽ bị phóng to tương đối so với ảnh, làm YOLO nhận diện sai.
        results = model(haystack_img, verbose=False, conf=conf or model.conf)[0]
        points = []
        class_name = Path(self.needle_img_path).stem  # filename without .png
        
        rx, ry, rw, rh = 0, 0, 0, 0
        if roi is not None:
            rx, ry, rw, rh = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])

        for box in results.boxes:
            cls_name = results.names[int(box.cls)]
            if cls_name == class_name:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                # Nếu có ROI, chỉ lấy các điểm nằm TRONG vùng ROI
                if roi is not None:
                    if not (rx <= cx <= rx + rw and ry <= cy <= ry + rh):
                        continue
                points.append((cx, cy))

        elapsed_ms = (time.time() - start_t) * 1000
        _log.debug(f"[YOLO find] {self.needle_img_path} → {len(points)} hit(s) - {elapsed_ms:.1f}ms")
        return points


    def _save_yolo_label(self, haystack_img, dst):
        """Saves YOLO format label for the detected object."""
        if not self._auto_label:
            return

        self._ensure_yolo_classes()
        class_name = Path(self.needle_img_path).stem
        if class_name not in self._yolo_classes:
            _log.warning(f"[Auto-Label] Class '{class_name}' not in data.yaml. Skipping.")
            return
        
        class_id = self._yolo_classes[class_name]
        img_h, img_w = haystack_img.shape[:2]

        # Calculate bounding box
        min_x, min_y = np.min(dst[:, 0, :], axis=0)
        max_x, max_y = np.max(dst[:, 0, :], axis=0)
        
        # Clamp to image bounds
        min_x, max_x = max(0, min_x), min(img_w - 1, max_x)
        min_y, max_y = max(0, min_y), min(img_h - 1, max_y)

        bw, bh = (max_x - min_x), (max_y - min_y)
        if bw < 5 or bh < 5: return # Quá nhỏ

        cx, cy = (min_x + bw/2.0) / img_w, (min_y + bh/2.0) / img_h
        nw, nh = bw / img_w, bh / img_h

        # Save files
        base_dir = os.path.join(os.path.dirname(__file__), "yolo_dataset", "auto_labeled")
        img_dir = os.path.join(base_dir, "images")
        lbl_dir = os.path.join(base_dir, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        ts = int(time.time() * 1000)
        fname = f"{class_name}_{ts}"
        
        img_path = os.path.join(img_dir, f"{fname}.jpg")
        lbl_path = os.path.join(lbl_dir, f"{fname}.txt")

        try:
            cv.imwrite(img_path, haystack_img)
            with open(lbl_path, "w") as f:
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            _log.info(f"[Auto-Label] Saved: {fname}.jpg (Class {class_id})")
        except Exception as e:
            _log.error(f"[Auto-Label] Failed to save: {e}")


    def exists(self, haystack_img, min_match_count=10, roi=None, auto_label=None) -> bool:
        """Kiem tra needle co ton tai trong haystack khong.
        auto_label: None = dung class setting, True/False = ghi de.
        """
        if auto_label is None:
            auto_label = self._auto_label

        start_t = time.time()
        
        # Crop to ROI if specified
        roi_x, roi_y = 0, 0
        if roi is not None:
            rx, ry, rw, rh = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
            img_h, img_w = haystack_img.shape[:2]
            rx = max(0, min(rx, img_w - 1))
            ry = max(0, min(ry, img_h - 1))
            rw = max(1, min(rw, img_w - rx))
            rh = max(1, min(rh, img_h - ry))
            haystack_img = haystack_img[ry:ry+rh, rx:rx+rw]
            roi_x, roi_y = rx, ry

        # Chuyển sang grayscale.
        if len(haystack_img.shape) == 3:
            haystack_gray = cv.cvtColor(haystack_img, cv.COLOR_BGR2GRAY)
        search_img_for_sift = haystack_img # Keep original for auto_label if needed
        if roi is not None:
            rx, ry, rw, rh = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
            img_h, img_w = haystack_img.shape[:2]
            rx = max(0, min(rx, img_w - 1))
            ry = max(0, min(ry, img_h - 1))
            rw = max(1, min(rw, img_w - rx))
            rh = max(1, min(rh, img_h - ry))
            search_img_for_sift = haystack_img[ry:ry+rh, rx:rx+rw]
            roi_x, roi_y = rx, ry

        # Chuyển sang grayscale.
        if len(search_img_for_sift.shape) == 3:
            haystack_gray = cv.cvtColor(search_img_for_sift, cv.COLOR_BGR2GRAY)
        else:
            haystack_gray = search_img_for_sift

        if self.needle_w > haystack_gray.shape[1] or self.needle_h > haystack_gray.shape[0]:
            return False

        kp, des = self.sift.detectAndCompute(haystack_gray, None)

        if des is None or self.needle_des is None:
            return False
            
        if len(des) < 2 or len(self.needle_des) < 2:
            return False

        matches = self.matcher.knnMatch(self.needle_des, des, k=2)

        if not matches:
            return False
            
        good = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.65 * n.distance:  # 0.65 chặt hơn chuẩn (0.7), ít false match hơn
                    good.append(m)

        good_matches_count = len(good)

        # --- AUTO LABELER LOGIC (Bootstrap cho YOLO) ---
        if auto_label and good_matches_count >= 20: # Chỉ lưu nếu match cực mạnh
            # findHomography yêu cầu TỐI THIỂU 4 cặp điểm, không thì sẽ crash
            if good_matches_count >= 4:
                src_pts = np.float32([self.needle_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 3.0)
                if M is not None:
                    h, w = self.needle_gray.shape
                    pts_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv.perspectiveTransform(pts_corners, M)
                    # Add ROI offset back to dst points before saving label
                    dst[:, 0, 0] += roi_x
                    dst[:, 0, 1] += roi_y
                    self._save_yolo_label(haystack_img, dst)

        elapsed_ms = (time.time() - start_t) * 1000
        _log.debug(f"[SIFT exists] {self.needle_img_path if hasattr(self, 'needle_img_path') else 'template'} - Elapsed: {elapsed_ms:.1f}ms")
        return good_matches_count >= min_match_count

    def find(self, haystack_img, min_match_count=10, debug_mode=None, is_color=False, roi=None, auto_label=None):
        """Tim vi tri cua ONE best match bang SIFT hoac YOLO.
        roi: (x, y, w, h) pixel coords to restrict search area. None = full image.
        auto_label: None = dung class setting, True/False = ghi de.
        """
        # --- Route qua YOLO nếu model đã được load ---
        if self._yolo_model is not None:
            pts = self._yolo_find(haystack_img, roi=roi)
            # YOLO không cần scale vì nó train trắn lên ảnh gốc
            return (pts, []) if is_color else pts

        start_t = time.time()
        
        scale = _global_scale
        # Save the color input for color matching (full-res original)
        haystack_color = haystack_img
        
        # Crop to ROI if specified
        roi_x, roi_y = 0, 0
        if roi is not None:
            rx, ry, rw, rh = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
            img_h, img_w = haystack_img.shape[:2]
            rx = max(0, min(rx, img_w - 1))
            ry = max(0, min(ry, img_h - 1))
            rw = max(1, min(rw, img_w - rx))
            rh = max(1, min(rh, img_h - ry))
            haystack_img = haystack_img[ry:ry+rh, rx:rx+rw]
            roi_x, roi_y = rx, ry

        # Chuyển sang grayscale.
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

        matches = self.matcher.knnMatch(self.needle_des, des, k=2)

        # Lowe's ratio test
        good = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.65 * n.distance:  # 0.65 chặt hơn chuẩn (0.7), ít false match hơn
                    good.append(m)

        if debug_mode:
            _log.info(f"[SIFT] {self.needle_img_path if hasattr(self, 'needle_img_path') else 'template'} -> {len(good)}/{min_match_count} good matches.")

        points = []
        dominant_colors = [] # Added for consistency with the new return
        if len(good) >= min_match_count:
            # findHomography yêu cầu TỐI THIỂU 4 cặp điểm, không thì sẽ crash
            if len(good) < 4:
                if debug_mode: _log.debug(f"[SIFT find] {self.needle_img_path if hasattr(self, 'needle_img_path') else 'template'} - not enough good matches for homography ({len(good)}<4)")
                if is_color: return points, dominant_colors
                return points
            # Lấy tọa độ các điểm match
            src_pts = np.float32([self.needle_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # RANSAC 3.0: ngưỡng chấp nhận pixel lệch, thấp hơn = chính xác hơn (mặc định 5.0)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 3.0)
            
            if M is not None:
                # Tọa độ 4 góc của template
                h, w = self.needle_gray.shape
                pts_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                
                # Ánh xạ 4 góc sang viền trên hình to
                dst = cv.perspectiveTransform(pts_corners, M)

                # --- AUTO LABELER LOGIC (Bootstrap cho YOLO) ---
                if auto_label and len(good) >= 20: # Chỉ lưu nếu match cực mạnh
                    self._save_yolo_label(haystack_img, dst)

                if debug_mode:
                    _log.debug(f"[SIFT find] {self.needle_img_path} - Homography found. Emitting point.")
                
                # Tính tâm của bounding box
                center_x = int(np.mean(dst[:, 0, 0]))
                center_y = int(np.mean(dst[:, 0, 1]))
                
                # Cộng offset ROI vào tọa độ (đưa về tọa độ ảnh gốc)
                center_x += roi_x
                center_y += roi_y
                
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