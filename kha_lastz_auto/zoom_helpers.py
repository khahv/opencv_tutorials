"""
Shared helpers for world_zoomout and base_zoomout.
Used by bot_engine (step execution) and connection_detector (connection check).
"""
import time
import pyautogui
from vision import get_global_scale


def _roi_crop(screenshot, vision, roi_center_x, roi_center_y, roi_padding):
    """Return (search_img, roi_offset). If ROI not set, return (screenshot, (0,0))."""
    if screenshot is None or vision is None or roi_center_x is None or roi_center_y is None:
        return screenshot, (0, 0)
    sh, sw = screenshot.shape[:2]
    scale = get_global_scale()
    nw_px = max(1, int(vision.needle_w * scale))
    nh_px = max(1, int(vision.needle_h * scale))
    cx_px = int(roi_center_x * sw)
    cy_px = int(roi_center_y * sh)
    half_w = int(nw_px * roi_padding)
    half_h = int(nh_px * roi_padding)
    rx = max(0, cx_px - half_w)
    ry = max(0, cy_px - half_h)
    rx2 = min(sw, cx_px + half_w)
    ry2 = min(sh, cy_px + half_h)
    roi_offset = (rx, ry)
    search_img = screenshot[ry:ry2, rx:rx2]
    return search_img, roi_offset


def _shift_points(pts, roi_offset):
    if not pts or roi_offset == (0, 0):
        return pts
    ox, oy = roi_offset
    return [
        ((p[0] + ox, p[1] + oy) + tuple(p[2:]) if len(p) > 2 else (p[0] + ox, p[1] + oy))
        for p in pts
    ]


def _find_kwargs(debug_log=False, match_color=False, color_tolerance=None):
    out = {}
    if debug_log:
        out["debug_mode"] = "info"
        out["debug_log"] = True
    if match_color:
        out["is_color"] = True
    if color_tolerance is not None:
        out["color_tolerance"] = color_tolerance
    return out


def _fresh_search_img(wincap, vision, roi_center_x, roi_center_y, roi_padding):
    """Get a fresh screenshot and return (search_img, roi_offset). Returns (None, (0,0)) on failure."""
    snap = wincap.get_screenshot() if hasattr(wincap, "get_screenshot") else None
    if snap is None:
        return None, (0, 0)
    search_img, roi_offset = _roi_crop(snap, vision, roi_center_x, roi_center_y, roi_padding)
    if search_img is None or search_img.size == 0:
        search_img, roi_offset = snap, (0, 0)
    return search_img, roi_offset


def do_world_zoomout(
    wincap,
    vision_cache,
    log,
    template_path,
    world_button_path,
    screenshot=None,
    threshold=0.75,
    scroll_times=5,
    scroll_interval_sec=0.1,
    roi_center_x=None,
    roi_center_y=None,
    roi_padding=3.0,
    log_prefix="",
    debug_log=False,
    match_color=False,
    color_tolerance=None,
):
    """
    Ensure we are on world view then scroll (one attempt).
    - If we already see HQ: click HQ → sleep 2 → fresh screenshot → click World → sleep 2 → zoomout (scroll).
    - If we see World (not HQ): click HQ first if visible → sleep 2 → fresh screenshot → click World → sleep 2 → fresh screenshot → if HQ visible then zoomout.
    Every match step uses a fresh screenshot.
    Returns True if scroll was done; False otherwise.
    """
    if screenshot is None and hasattr(wincap, "get_screenshot"):
        screenshot = wincap.get_screenshot()
    if screenshot is None:
        return False
    vision = vision_cache.get(template_path)
    vision_world = vision_cache.get(world_button_path) if world_button_path else None
    if not vision:
        if log_prefix:
            log.warning("%sworld_zoomout template not in cache: %s", log_prefix, template_path)
        return False

    search_img, roi_offset = _roi_crop(screenshot, vision, roi_center_x, roi_center_y, roi_padding)
    if search_img is None or search_img.size == 0:
        search_img = screenshot
        roi_offset = (0, 0)

    find_kw = _find_kwargs(debug_log=debug_log, match_color=match_color, color_tolerance=color_tolerance)

    def _scroll():
        if hasattr(wincap, "focus_window"):
            wincap.focus_window(force=True)
            time.sleep(0.05)
        cx = wincap.offset_x + wincap.w // 2
        cy = wincap.offset_y + wincap.h // 2
        pyautogui.moveTo(cx, cy)
        time.sleep(0.05)
        for _ in range(scroll_times):
            pyautogui.scroll(-3)
            time.sleep(scroll_interval_sec)
        if log_prefix:
            log.info("%sworld_zoomout scrolled x%d at center", log_prefix, scroll_times)

    # Initial state: use fresh screenshot (we already have it as `screenshot`).
    points_hq = vision.find(search_img, threshold=threshold, **find_kw)
    points_hq = _shift_points(points_hq if points_hq else [], roi_offset)
    points_w = []
    if vision_world:
        points_w = vision_world.find(search_img, threshold=threshold, **find_kw)
        points_w = _shift_points(points_w if points_w else [], roi_offset)

    if hasattr(wincap, "focus_window"):
        wincap.focus_window(force=True)
        time.sleep(0.05)

    if points_hq:
        # Already see HQ: click HQ → sleep 2 → fresh screenshot → click World → sleep 2 → zoomout
        sx, sy = wincap.get_screen_position((points_hq[0][0], points_hq[0][1]))
        pyautogui.click(sx, sy)
        time.sleep(2)
        search_img2, roi_off2 = _fresh_search_img(wincap, vision, roi_center_x, roi_center_y, roi_padding)
        if search_img2 is None:
            return False
        points_w2 = vision_world.find(search_img2, threshold=threshold, **find_kw) if vision_world else []
        points_w2 = _shift_points(points_w2 if points_w2 else [], roi_off2)
        if not points_w2:
            return False
        sx, sy = wincap.get_screen_position((points_w2[0][0], points_w2[0][1]))
        pyautogui.click(sx, sy)
        time.sleep(2)
        _scroll()
        return True

    if points_w:
        # We see World (e.g. base view). If HQ also visible: click HQ → sleep 2 → fresh screenshot.
        points_hq_same = vision.find(search_img, threshold=threshold, **find_kw)
        points_hq_same = _shift_points(points_hq_same if points_hq_same else [], roi_offset)
        if points_hq_same:
            sx, sy = wincap.get_screen_position((points_hq_same[0][0], points_hq_same[0][1]))
            pyautogui.click(sx, sy)
            time.sleep(2)
            search_img2, roi_off2 = _fresh_search_img(wincap, vision, roi_center_x, roi_center_y, roi_padding)
            if search_img2 is None:
                return False
            points_w2 = vision_world.find(search_img2, threshold=threshold, **find_kw) if vision_world else []
            points_w2 = _shift_points(points_w2 if points_w2 else [], roi_off2)
            if not points_w2:
                return False
            sx, sy = wincap.get_screen_position((points_w2[0][0], points_w2[0][1]))
            pyautogui.click(sx, sy)
            time.sleep(2)
        else:
            sx, sy = wincap.get_screen_position((points_w[0][0], points_w[0][1]))
            pyautogui.click(sx, sy)
            time.sleep(2)
        # Fresh screenshot: confirm we are on world (HQ visible) then zoomout
        search_img3, roi_off3 = _fresh_search_img(wincap, vision, roi_center_x, roi_center_y, roi_padding)
        if search_img3 is None:
            return False
        points_hq3 = vision.find(search_img3, threshold=threshold, **find_kw)
        points_hq3 = _shift_points(points_hq3 if points_hq3 else [], roi_off3)
        if points_hq3:
            _scroll()
            return True
    return False


def do_base_zoomout(
    wincap,
    vision_cache,
    log,
    template_path,
    world_button_path,
    screenshot=None,
    threshold=0.75,
    scroll_times=5,
    scroll_interval_sec=0.1,
    roi_center_x=None,
    roi_center_y=None,
    roi_padding=3.0,
    log_prefix="",
    debug_log=False,
    match_color=False,
    color_tolerance=None,
):
    """
    Zoom out from base to world view: find HQ or World, click in sequence, then scroll (one attempt).
    Returns True if scroll was performed; False otherwise.
    """
    if screenshot is None and hasattr(wincap, "get_screenshot"):
        screenshot = wincap.get_screenshot()
    if screenshot is None:
        return False
    vision = vision_cache.get(template_path)
    vision_world = vision_cache.get(world_button_path) if world_button_path else None
    if not vision or not vision_world:
        if log_prefix:
            log.warning("%sbase_zoomout: template or world_button not in cache", log_prefix)
        return False

    search_img, roi_offset = _roi_crop(screenshot, vision, roi_center_x, roi_center_y, roi_padding)
    if search_img is None or search_img.size == 0:
        search_img = screenshot
        roi_offset = (0, 0)

    find_kw = _find_kwargs(debug_log=debug_log, match_color=match_color, color_tolerance=color_tolerance)

    def _scroll():
        if hasattr(wincap, "focus_window"):
            wincap.focus_window(force=True)
            time.sleep(0.05)
        cx = wincap.offset_x + wincap.w // 2
        cy = wincap.offset_y + wincap.h // 2
        pyautogui.moveTo(cx, cy)
        time.sleep(0.05)
        for _ in range(scroll_times):
            pyautogui.scroll(-3)
            time.sleep(scroll_interval_sec)
        if log_prefix:
            log.info("%sbase_zoomout scrolled x%d at center", log_prefix, scroll_times)

    # (1) HQ visible: click HQ → then check World → scroll if World visible
    points_hq = vision.find(search_img, threshold=threshold, **find_kw)
    points_hq = _shift_points(points_hq if points_hq else [], roi_offset)
    if points_hq and len(points_hq) > 0:
        sx, sy = wincap.get_screen_position((points_hq[0][0], points_hq[0][1]))
        pyautogui.click(sx, sy)
        time.sleep(2)
        scr2 = wincap.get_screenshot() if hasattr(wincap, "get_screenshot") else None
        if scr2 is not None:
            search2, off2 = _roi_crop(scr2, vision_world, roi_center_x, roi_center_y, roi_padding)
            if search2 is None or search2.size == 0:
                search2, off2 = scr2, (0, 0)
            points_w = vision_world.find(search2, threshold=threshold, **find_kw)
            points_w = _shift_points(points_w if points_w else [], off2)
            if points_w and len(points_w) > 0:
                _scroll()
                return True
            points_hq2 = vision.find(search2, threshold=threshold, **find_kw)
            points_hq2 = _shift_points(points_hq2 if points_hq2 else [], off2)
            if points_hq2 and len(points_hq2) > 0:
                sx2, sy2 = wincap.get_screen_position((points_hq2[0][0], points_hq2[0][1]))
                pyautogui.click(sx2, sy2)
                time.sleep(2)
                scr3 = wincap.get_screenshot() if hasattr(wincap, "get_screenshot") else None
                if scr3 is not None:
                    search3, off3 = _roi_crop(scr3, vision_world, roi_center_x, roi_center_y, roi_padding)
                    if search3 is None or search3.size == 0:
                        search3, off3 = scr3, (0, 0)
                    points_w3 = vision_world.find(search3, threshold=threshold, **find_kw)
                    points_w3 = _shift_points(points_w3 if points_w3 else [], off3)
                    if points_w3 and len(points_w3) > 0:
                        _scroll()
                        return True
        return False

    # (2) World visible first: click World → click HQ → scroll
    points_w = vision_world.find(search_img, threshold=threshold, **find_kw)
    points_w = _shift_points(points_w if points_w else [], roi_offset)
    if points_w and len(points_w) > 0:
        if hasattr(wincap, "focus_window"):
            wincap.focus_window(force=True)
            time.sleep(0.05)
        sx_w, sy_w = wincap.get_screen_position((points_w[0][0], points_w[0][1]))
        pyautogui.click(sx_w, sy_w)
        time.sleep(2)
        scr_after = wincap.get_screenshot() if hasattr(wincap, "get_screenshot") else None
        if scr_after is not None:
            search_after, off_after = _roi_crop(scr_after, vision, roi_center_x, roi_center_y, roi_padding)
            if search_after is None or search_after.size == 0:
                search_after, off_after = scr_after, (0, 0)
            points_hq_after = vision.find(search_after, threshold=threshold, **find_kw)
            points_hq_after = _shift_points(points_hq_after if points_hq_after else [], off_after)
            if points_hq_after and len(points_hq_after) > 0:
                sx_hq, sy_hq = wincap.get_screen_position((points_hq_after[0][0], points_hq_after[0][1]))
                pyautogui.click(sx_hq, sy_hq)
                time.sleep(2)
        _scroll()
        return True
    return False
