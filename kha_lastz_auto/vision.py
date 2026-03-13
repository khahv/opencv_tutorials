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
    """Decorator: log execution time of find/exists in milliseconds."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        needle = getattr(args[0], 'needle_name', '?') if args else '?'
        if elapsed_ms > 20: # Chỉ log nếu tốn hơn 20ms để màn hình log đỡ rối
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
# Global scale (kept for compatibility)
# ---------------------------------------------------------------------------
_global_scale = 1.0

def set_global_scale(scale):
    global _global_scale
    _global_scale = float(scale)

def get_global_scale():
    return _global_scale


# ---------------------------------------------------------------------------
# Vision class (AKAZE Version - Upscale Haystack logic)
# ---------------------------------------------------------------------------
class Vision:

    needle_img   = None
    needle_w     = 0
    needle_h     = 0
    method       = None
    kpts_needle  = None
    desc_needle  = None

    def __init__(self, needle_img_path, method=cv.TM_CCOEFF_NORMED):
        # We load the image
        img = cv.imread(needle_img_path, cv.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError('Khong tim thay anh: {}'.format(needle_img_path))

        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)

        self.needle_img = img
        import os
        self.needle_name = os.path.splitext(os.path.basename(needle_img_path))[0]
        self.needle_w    = img.shape[1]
        self.needle_h    = img.shape[0]
        self.method      = method # Kept for backward compatibility
        
        # Prepare Template Edges for Distance Transform Matching
        gray = cv.cvtColor(self.needle_img, cv.COLOR_BGR2GRAY) if len(self.needle_img.shape) == 3 else self.needle_img
        # Get edges of the template
        self.needle_edges = cv.Canny(gray, 50, 150)
        
        # Inchamfer Matching, we don't care about keypoints, so log success directly
        _log.info("[vision:info] [%s] Load Distance Transform Model thành công (%dx%d).", self.needle_name, self.needle_w, self.needle_h)

    @timeit
    def exists(self, haystack_img, threshold=0.5, debug_log=False) -> bool:
        """AKAZE existence check."""
        hits = self.find(haystack_img, threshold, debug_mode=None, is_color=False, debug_log=debug_log)
        return len(hits) > 0

    @timeit
    def find(self, haystack_img, threshold=0.5, debug_mode=None, is_color=False, debug_log=False,
             ratio_test=None, min_inliers=None):
        """
        Distance Transform (Chamfer) Matcher.
        Returns a list of (cx, cy, w, h).
        """
        # _global_scale = window_width / reference_width (e.g., 540/1080 = 0.5)
        # Template is always at reference_width (1080p).
        # We must UPSCALE the haystack to match the template's 1080p scale.
        scale_factor = 1.0 / _global_scale if _global_scale > 0 else 1.0

        if scale_factor != 1.0:
            haystack_target = cv.resize(haystack_img, (0,0), fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_LINEAR)
        else:
            haystack_target = haystack_img

        if is_color:
            match_result = cv.matchTemplate(haystack_target, self.needle_img, cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(match_result)
            
            if max_val < threshold:
                if debug_log:
                    _log.debug("[vision:debug] [%s] Color Match Failed: score %.2f < %.2f", self.needle_name, max_val, threshold)
                return []
                
            x, y = max_loc
            bw, bh = self.needle_w, self.needle_h
            cx = int(x + bw // 2)
            cy = int(y + bh // 2)

            if debug_log:
                _log.debug("[vision:debug] [%s] Color Match OK! Score: %.2f (Thresh: %.2f)", self.needle_name, max_val, threshold)

            if scale_factor != 1.0:
                cx = int(cx / scale_factor)
                cy = int(cy / scale_factor)
                bw = int(bw / scale_factor)
                bh = int(bh / scale_factor)

            return [(cx, cy, bw, bh)]


        # 1. Convert Haystack to Edges
        gray_hay = cv.cvtColor(haystack_target, cv.COLOR_BGR2GRAY) if len(haystack_target.shape) == 3 else haystack_target
        edges_hay = cv.Canny(gray_hay, 50, 150)
        
        # 2. Invert Edges (Edges become 0, background becomes 255) because distanceTransform calculates distance to ZERO pixels
        inverted_edges_hay = cv.bitwise_not(edges_hay)

        # 3. Compute Distance Transform Map
        # cv.DIST_L2 is Euclidean distance. cv.DIST_MASK_PRECISE returns accurate float distances.
        dist_map = cv.distanceTransform(inverted_edges_hay, cv.DIST_L2, cv.DIST_MASK_PRECISE)

        # 4. Perform Chamfer Matching via Template Matching
        # matchTemplate slides the needle over the dist_map.
        # TM_SQDIFF evaluates the sum of squared differences.
        # Wait, since our template (needle_edges) has edges as 255 and background as 0:
        # We must normalize needle_edges to 1 at edges and 0 at background to easily sum up the distances under the edges!
        needle_binary = (self.needle_edges / 255.0).astype(np.float32)

        # Slide binary template over distance map. 
        # Result pixel (x,y) contains the SUM of distances from needle edges to nearest haystack edges.
        match_result = cv.matchTemplate(dist_map, needle_binary, cv.TM_CCORR)

        # Find the location with the MINIMUM sum of distances
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(match_result)
        
        # 5. Threshold checking logic
        # min_val is the total distance sum across all edge pixels in the template.
        # Let's find the AVERAGE distance per edge pixel:
        total_edge_pixels = np.count_nonzero(needle_binary)
        if total_edge_pixels == 0:
             return []
             
        avg_dist = min_val / total_edge_pixels
        
        # Threshold Mapping:
        # User threshold goes from 0.0 (Loose) to 1.0 (Strict).
        # We translate this to max allowed average distance per pixel (in pixels).
        # Thresh 0.8 -> max_avg_dist = 2.0 pixels
        # Thresh 0.5 -> max_avg_dist = 5.0 pixels
        max_allowed_dist = (1.0 - threshold) * 10.0 
        
        if avg_dist > max_allowed_dist:
            if debug_log:
                _log.debug("[vision:debug] [%s] Khong dat Threshold (Thresh: %.2f). Avg Dist = %.2fpx (Tu choi vi > %.2fpx)", 
                           self.needle_name, threshold, avg_dist, max_allowed_dist)
            return []
            
        if debug_log:
            _log.debug("[vision:debug] [%s] Match OK! Avg Dist = %.2fpx (Cho phep <= %.2fpx)", 
                       self.needle_name, avg_dist, max_allowed_dist)

        # Found Match!
        x, y = min_loc  # In TM_CCORR, we would usually look at max_loc. BUT wait! 
                        # We want the MINIMUM distance. So min_loc is the right coordinate.
        bw, bh = self.needle_w, self.needle_h
        cx = int(x + bw // 2)
        cy = int(y + bh // 2)

        # 6. Scale coordinates back down to the smaller window resolution
        if scale_factor != 1.0:
            cx = int(cx / scale_factor)
            cy = int(cy / scale_factor)
            bw = int(bw / scale_factor)
            bh = int(bh / scale_factor)

        points = [(cx, cy, bw, bh)]

        # Drawing logic for debugging
        if debug_mode in ("rectangles", "points"):
            haystack_vis = haystack_img.copy() if len(haystack_img.shape) == 3 else cv.cvtColor(haystack_img, cv.COLOR_GRAY2BGR)
            if debug_mode == "rectangles":
                # Draw a simple rectangle since we don't have tilted corners anymore
                top_left = (int(cx - bw/2), int(cy - bh/2))
                bottom_right = (int(cx + bw/2), int(cy + bh/2))
                cv.rectangle(haystack_vis, top_left, bottom_right, (0, 255, 0), 2)
            elif debug_mode == "points":
                cv.drawMarker(haystack_vis, (cx, cy), (255, 0, 255), cv.MARKER_CROSS, 40, 2)
            cv.imshow("Matches", haystack_vis)

        return points