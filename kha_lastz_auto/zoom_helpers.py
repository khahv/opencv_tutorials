"""
Shared helpers for world_zoomout and base_zoomout.
Used by bot_engine (step execution) and connection_detector (connection check).
"""
import time
import pyautogui
from vision import get_global_scale


def _tap_game_pixel(wincap, px: int, py: int) -> None:
    """Tap at game pixel (device / client); use ADB when available, else Win32 mouse."""
    import adb_input as _adb_mod

    adb = _adb_mod.get_adb_input()
    if adb is not None:
        adb.tap(int(px), int(py))
        return
    sx, sy = wincap.get_screen_position((int(px), int(py)))
    pyautogui.click(sx, sy)


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

    - HQ visible (world view): click HQ → sleep 2 → click World → sleep 2 →
      confirm HQ visible → scroll.
    - World button visible (base view): click World → sleep 2 →
      confirm HQ visible → scroll.
    - Neither visible: return False.

    Scroll only happens after confirming HQ is visible (world view confirmed).
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
        import adb_input as _adb_mod

        adb = _adb_mod.get_adb_input()
        if adb is not None:
            adb.wheel_zoom_out_approx(cx, cy, times=scroll_times, interval_sec=scroll_interval_sec)
        else:
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
        # Already see HQ (on world view): click HQ → sleep 2 → fresh screenshot → click World → sleep 2 →
        # confirm HQ visible (back on world view) → scroll.
        _tap_game_pixel(wincap, int(points_hq[0][0]), int(points_hq[0][1]))
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
        # Confirm we are back on world view (HQ visible) before scrolling
        search_img3, roi_off3 = _fresh_search_img(wincap, vision, roi_center_x, roi_center_y, roi_padding)
        if search_img3 is None:
            return False
        points_hq3 = vision.find(search_img3, threshold=threshold, **find_kw)
        points_hq3 = _shift_points(points_hq3 if points_hq3 else [], roi_off3)
        if points_hq3:
            _scroll()
            return True
        return False

    if points_w:
        # World button visible (on base view): click World → sleep 2 →
        # confirm HQ visible (now on world view) → scroll.
        _tap_game_pixel(wincap, int(points_w[0][0]), int(points_w[0][1]))
        time.sleep(2)
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
    Navigate to base view then scroll to zoom out the map.

    Logic (each step uses a fresh screenshot, sleep 2s before checking):
    - WorldButton visible → on base view → stop clicking → scroll → done
    - HeadquartersButton visible (World not visible) → on world view → click HQ → sleep 2s → fresh screenshot → repeat
    - Loop up to 3 attempts; scroll only once WorldButton is confirmed visible.

    Returns True if scroll was performed; False otherwise.
    """
    vision = vision_cache.get(template_path)
    vision_world = vision_cache.get(world_button_path) if world_button_path else None
    if not vision or not vision_world:
        if log_prefix:
            log.warning("%sbase_zoomout: template or world_button not in cache", log_prefix)
        return False

    find_kw = _find_kwargs(debug_log=debug_log, match_color=match_color, color_tolerance=color_tolerance)

    def _detect(scr):
        """Return (hq_points, world_points) from a screenshot using ROI crop."""
        si_hq, off_hq = _roi_crop(scr, vision, roi_center_x, roi_center_y, roi_padding)
        if si_hq is None or si_hq.size == 0:
            si_hq, off_hq = scr, (0, 0)
        hq = _shift_points(vision.find(si_hq, threshold=threshold, **find_kw) or [], off_hq)

        si_w, off_w = _roi_crop(scr, vision_world, roi_center_x, roi_center_y, roi_padding)
        if si_w is None or si_w.size == 0:
            si_w, off_w = scr, (0, 0)
        world = _shift_points(vision_world.find(si_w, threshold=threshold, **find_kw) or [], off_w)
        return hq, world

    def _scroll():
        if hasattr(wincap, "focus_window"):
            wincap.focus_window(force=True)
            time.sleep(0.05)
        cx = wincap.offset_x + wincap.w // 2
        cy = wincap.offset_y + wincap.h // 2
        import adb_input as _adb_mod

        adb = _adb_mod.get_adb_input()
        if adb is not None:
            adb.wheel_zoom_out_approx(cx, cy, times=scroll_times, interval_sec=scroll_interval_sec)
        else:
            pyautogui.moveTo(cx, cy)
            time.sleep(0.05)
            for _ in range(scroll_times):
                pyautogui.scroll(-3)
                time.sleep(scroll_interval_sec)
        if log_prefix:
            log.info("%sbase_zoomout scrolled x%d at center", log_prefix, scroll_times)

    # Always start with a fresh screenshot
    scr = wincap.get_screenshot() if hasattr(wincap, "get_screenshot") else None
    if scr is None:
        return False

    # Special case: if WorldButton already visible on entry → we're on base.
    # Click World once to go to world view first, then run the main loop below.
    hq_pts_init, world_pts_init = _detect(scr)
    if world_pts_init:
        if log_prefix:
            log.info("%sbase_zoomout WorldButton visible at start (on base), clicking World to enter world view", log_prefix)
        if hasattr(wincap, "focus_window"):
            wincap.focus_window(force=True)
            time.sleep(0.05)
        _tap_game_pixel(wincap, int(world_pts_init[0][0]), int(world_pts_init[0][1]))
        time.sleep(2)
        scr = wincap.get_screenshot() if hasattr(wincap, "get_screenshot") else None
        if scr is None:
            return False

    for attempt in range(3):
        hq_pts, world_pts = _detect(scr)

        # WorldButton visible → on base → scroll and done
        if world_pts:
            if log_prefix:
                log.info("%sbase_zoomout WorldButton visible (attempt %d), scrolling", log_prefix, attempt + 1)
            _scroll()
            return True

        # HeadquartersButton visible → on world view → click HQ to navigate to base
        if hq_pts:
            if log_prefix:
                log.info("%sbase_zoomout HQ visible (attempt %d), clicking to navigate to base", log_prefix, attempt + 1)
            if hasattr(wincap, "focus_window"):
                wincap.focus_window(force=True)
                time.sleep(0.05)
            _tap_game_pixel(wincap, int(hq_pts[0][0]), int(hq_pts[0][1]))
            time.sleep(2)
            scr = wincap.get_screenshot() if hasattr(wincap, "get_screenshot") else None
            if scr is None:
                return False
            continue

        # Neither button visible → unknown state, stop
        if log_prefix:
            log.warning("%sbase_zoomout: neither HQ nor World visible (attempt %d), aborting", log_prefix, attempt + 1)
        break

    return False
