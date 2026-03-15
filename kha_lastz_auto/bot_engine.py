"""
Engine chay Function (YAML): load functions, thuc thi tung step.
Step types: match_click, match_multi_click, match_count, sleep, send_zalo, click_position, wait_until_match, key_press, set_level, type_text, click_unless_visible, drag, close_ui, base_zoomout.
Each step returns true/false. Next step is blocked (function aborted) if previous returned false, unless run_always: true is set.
send_zalo: message (required), receiver_name (optional; ten hien thi trong danh sach chat, fallback DEFAULT_CLICK_AFTER_OPEN), repeat_interval_sec (optional; when set and trigger_active_cb provided, repeats while trigger active).
"""
import os
import re
import time
import logging
import random
import yaml
import pyautogui
import cv2 as cv
from vision import Vision, get_global_scale
from pynput.mouse import Button, Controller
from fast_clicker import FastClicker
from ocr_utils import (
    read_level_from_roi    as _read_level_from_roi,
    read_raw_text_from_roi as _read_raw_text_from_roi,
    read_region_relative,
    _parse_level,
)

log = logging.getLogger("kha_lastz")
_mouse_ctrl = Controller()

try:
    import zalo_web_clicker as _zalo_web_clicker
except ImportError:
    _zalo_web_clicker = None


def _save_debug_image(screenshot, raw_center, click_center, needle_w, needle_h, event_type, template_path,
                      truck_name=None):
    """Save a debug PNG with green rect (match area) and red circle (click/move target).
    For YellowTruckSmall: also saves a tight crop of the truck. All files include a timestamp.
    truck_name: optional string overlaid on the YellowTruckSmall crop."""
    import datetime
    ts = datetime.datetime.now().strftime("%H%M%S_%f")[:-3]  # HHMMSSmmm
    tname = os.path.splitext(os.path.basename(template_path))[0]
    os.makedirs("debug_ocr", exist_ok=True)

    # Full screenshot with rect + dot
    dbg = screenshot.copy()
    rx = raw_center[0] - needle_w // 2
    ry = raw_center[1] - needle_h // 2
    cv.rectangle(dbg, (rx, ry), (rx + needle_w, ry + needle_h), (0, 255, 0), 2)
    cv.circle(dbg, (click_center[0], click_center[1]), 12, (0, 0, 255), -1)
    out_path = os.path.join("debug_ocr", "debug_{}_{}_{}.png".format(event_type, tname, ts))
    cv.imwrite(out_path, dbg)
    log.info("[Runner] debug_click saved → {}".format(os.path.abspath(out_path)))

    # YellowTruckSmall: crop tight around the truck + overlay name
    if "YellowTruckSmall" in tname:
        pad_x = max(30, needle_w * 6)   # wide left padding to show player name
        pad_y = max(20, needle_h)
        h, w = screenshot.shape[:2]
        x1 = max(0, rx - pad_x)
        y1 = max(0, ry - pad_y)
        x2 = min(w, rx + needle_w + 20)
        y2 = min(h, ry + needle_h + pad_y)
        crop = screenshot[y1:y2, x1:x2].copy()
        lrx, lry = rx - x1, ry - y1
        cv.rectangle(crop, (lrx, lry), (lrx + needle_w, lry + needle_h), (0, 255, 0), 2)
        cx, cy = click_center[0] - x1, click_center[1] - y1
        if 0 <= cx < crop.shape[1] and 0 <= cy < crop.shape[0]:
            cv.circle(crop, (cx, cy), 10, (0, 0, 255), -1)
        if truck_name:
            label = truck_name
            font_scale = max(0.5, needle_h / 30.0)
            thickness = max(1, int(font_scale * 1.5))
            (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            tx = max(0, lrx - tw - 4)
            ty = lry + needle_h // 2 + th // 2
            cv.rectangle(crop, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
            cv.putText(crop, label, (tx, ty), cv.FONT_HERSHEY_SIMPLEX,
                       font_scale, (0, 255, 255), thickness, cv.LINE_AA)
        safe_name = (truck_name or "").replace("/", "_").replace("\\", "_").replace(" ", "_")[:20]
        crop_fname = "YellowTruck_{}_{}.png".format(safe_name, ts) if safe_name else "YellowTruck_crop_{}.png".format(ts)
        crop_path = os.path.join("debug_ocr", crop_fname)
        cv.imwrite(crop_path, crop)
        log.info("[Runner] debug_click saved (truck crop) → {}".format(os.path.abspath(crop_path)))
        return crop_path  # caller can store this to rename/overlay once player name is known
    return None


def _retitle_truck_crop(old_path, player_name):
    """Reload existing truck crop, overlay player name, save under new name including the name."""
    if not old_path or not os.path.isfile(old_path):
        return old_path
    img = cv.imread(old_path)
    if img is None:
        return old_path
    # Overlay name text at top-left
    label = player_name
    font_scale, thickness = 0.6, 2
    (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv.rectangle(img, (4, 4), (tw + 10, th + 12), (0, 0, 0), -1)
    cv.putText(img, label, (7, th + 7), cv.FONT_HERSHEY_SIMPLEX,
               font_scale, (0, 255, 255), thickness, cv.LINE_AA)
    # Build new filename: insert safe name before the timestamp suffix
    base = os.path.basename(old_path)
    root, ext = os.path.splitext(base)
    safe_name = player_name.replace("/", "_").replace("\\", "_").replace(" ", "_")[:25]
    # root is like "YellowTruck_crop_HHMMSS_mmm" — replace "crop" with the safe name
    new_root = root.replace("_crop_", "_{}_".format(safe_name), 1)
    if new_root == root:  # fallback: just append
        new_root = "{}_{}".format(root, safe_name)
    new_path = os.path.join(os.path.dirname(old_path), new_root + ext)
    cv.imwrite(new_path, img)
    try:
        os.remove(old_path)
    except OSError:
        pass
    log.info("[Runner] truck crop retitled → {}".format(os.path.abspath(new_path)))
    return new_path


def load_functions(functions_dir="functions"):
    """Load tat ca file YAML trong functions_dir. Tra ve dict: ten_function -> { description, steps }."""
    result = {}
    if not os.path.isdir(functions_dir):
        return result
    for fname in os.listdir(functions_dir):
        if not fname.endswith(".yaml") and not fname.endswith(".yml"):
            continue
        name = os.path.splitext(fname)[0]
        path = os.path.join(functions_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            result[name] = {
                "description": data.get("description", ""),
                "steps": data.get("steps", []),
            }
            # log.debug("[bot_engine] Loaded function: {}".format(name))
        except Exception as e:
            log.error("[bot_engine] Failed to load {}: {}".format(path, e))
    return result


def _crop_region_relative(screenshot, cx, cy, needle_w, needle_h,
                          x=0.0, y=0.0, w=1.0, h=1.0):
    """Crop a sub-region relative to a template match center.
    x/y/w/h are ratios of the needle (template) size — same coordinate space as read_region_relative.
    Returns the cropped BGR image, or None if out of bounds."""
    tl_x = cx - needle_w // 2
    tl_y = cy - needle_h // 2
    rx = max(0, tl_x + int(x * needle_w))
    ry = max(0, tl_y + int(y * needle_h))
    rw = max(1, int(w * needle_w))
    rh = max(1, int(h * needle_h))
    img_h, img_w = screenshot.shape[:2]
    rw = min(rw, img_w - rx)
    rh = min(rh, img_h - ry)
    if rw <= 0 or rh <= 0:
        return None
    crop = screenshot[ry: ry + rh, rx: rx + rw]
    return crop if crop.size > 0 else None


def collect_templates(functions_dict):
    """Lay danh sach duong dan template tu tat ca steps de tao vision cache."""
    templates = set()
    for fn in functions_dict.values():
        for step in fn["steps"]:
            if step.get("event_type") in ("match_click", "match_multi_click", "match_count", "match_move") and step.get("template"):
                templates.add(step["template"])
            if step.get("event_type") == "match_click" and step.get("template_array"):
                ta = step.get("template_array")
                if isinstance(ta, list):
                    for t in ta:
                        if t:
                            templates.add(t)
            if step.get("event_type") == "match_click" and step.get("refresh_template"):
                templates.add(step["refresh_template"])
            if step.get("event_type") == "wait_until_match" and step.get("template"):
                templates.add(step["template"])
            if step.get("event_type") == "click_unless_visible":
                if step.get("visible_template"):
                    templates.add(step["visible_template"])
                if step.get("click_template"):
                    templates.add(step["click_template"])
            if step.get("event_type") == "set_level":
                if step.get("plus_template"):
                    templates.add(step["plus_template"])
                if step.get("minus_template"):
                    templates.add(step["minus_template"])
                if step.get("level_anchor_template"):
                    templates.add(step["level_anchor_template"])
            if step.get("event_type") in ("base_zoomout", "close_ui"):
                if step.get("template"):
                    templates.add(step["template"])
                if step.get("world_button"):
                    templates.add(step["world_button"])
            if step.get("event_type") == "ocr_log":
                tpl = step.get("anchor_template") or step.get("template")
                if tpl:
                    templates.add(tpl)
    return list(templates)


def build_vision_cache(template_paths):
    """Tao dict template_path -> Vision(...)."""
    cache = {}
    for path in template_paths:
        try:
            cache[path] = Vision(path)
        except Exception as e:
            log.info("[bot_engine] Failed to load template {}: {}".format(path, e))
    return cache


class FunctionRunner:
    """Chay 1 function: giu state step hien tai, xu ly tung step theo type."""

    def __init__(self, vision_cache, fn_settings=None, bot_paused=None):
        self.vision_cache = vision_cache
        self.fn_settings = fn_settings if fn_settings is not None else {}
        self.bot_paused = bot_paused  # ref to {"paused": bool}; when True, send_zalo repeat counter resets on resume
        self.functions = {}
        self.state = "idle"  # idle | running
        self.function_name = None
        self.steps = []
        self.step_index = 0
        self.step_start_time = None
        self.step_click_count = 0
        self.wincap = None
        self.last_step_result = True  # True = previous step matched/succeeded
        self._step_retry_counts = {}  # {step_index: retry_count} for on_fail_goto
        self._tried_positions = []    # [(x, y)] positions already clicked in current cycle
        self._rtr_cache = {}          # require_text_in_region cache: frozenset(points) → filtered points
        self._rtr_page_logged = False # True after printing truck list for current refresh cycle
        self._fast_clicker = FastClicker()
        self._step_pos_cache = None
        self._debug_click_saved = False
        self._step_last_click_t = None

    def _fn_setting(self, key, fallback=None):
        """Return fn_settings[current_function][key], or fallback if not set."""
        return self.fn_settings.get(self.function_name or "", {}).get(key, fallback)

    def load(self, functions_dict):
        self.functions = functions_dict

    def start(self, function_name, wincap, trigger_event=None, trigger_active_cb=None):
        # Reload all functions from disk to pick up any YAML changes immediately
        try:
            new_functions = load_functions()
            if new_functions:
                self.functions = new_functions
        except Exception as e:
            log.warning("[Runner] Failed to reload functions from disk: {}".format(e))

        if function_name not in self.functions:
            log.info("[Runner] Function not found: {}".format(function_name))
            return False
        self.function_name = function_name
        self.steps = self.functions[function_name]["steps"]
        self.step_index = 0
        self.step_start_time = time.time()
        self.step_click_count = 0
        self.wincap = wincap
        self.state = "running"
        self.last_step_result = True
        self.trigger_event = trigger_event
        self.trigger_active_cb = trigger_active_cb
        self._step_retry_counts = {}
        self._tried_positions = []
        self._last_zero_refresh_t = 0
        self._last_truck_crop_path = None
        self._ocr_prev_vals = {}  # {step_index: last_read_value} for wait_for_change_region
        self._last_click_pos = None      # position of the most recent match_click (template-space)
        self._last_ocr_click_pos = None  # position that was current when ocr_log last ran
        self._tpl_array_idx = 0          # current template index for template_array steps
        for attr in ("_set_level_debug_saved", "_set_level_warned"):
            if hasattr(self, attr):
                delattr(self, attr)
        log.info("[Runner] Started function: {}".format(function_name))
        return True

    def _get_vision(self, template):
        """Get Vision object from cache, or load on-the-fly if missing/new."""
        if not template:
            return None
        v = self.vision_cache.get(template)
        if v:
            return v
        
        # On-the-fly load attempt
        if os.path.isfile(template):
            try:
                log.info("[Runner] Template {} not in cache, loading on-the-fly...".format(template))
                v = Vision(template)
                self.vision_cache[template] = v
                return v
            except Exception as e:
                log.error("[Runner] Failed to load template {} on-the-fly: {}".format(template, e))
        return None

    def stop(self):
        self._fast_clicker.stop()
        self.state = "idle"
        self.function_name = None

    def update(self, screenshot, wincap):
        """Tra ve 'running' | 'done' | 'idle'. Neu running thi xu ly step hien tai."""
        if self.state != "running" or screenshot is None or self.step_index >= len(self.steps):
            if self.state == "running" and self.step_index >= len(self.steps):
                self._fast_clicker.stop()
                self.state = "idle"
                suffix = "" if self.last_step_result else " (aborted)"
                log.info("[Runner] Finished function: {}{}".format(self.function_name, suffix))
                return "done"
            return "idle" if self.state == "idle" else "running"

        self.wincap = wincap
        step = self.steps[self.step_index]
        step_type = step.get("event_type", "")

        # Step result gate: skip this step if previous returned False,
        # UNLESS this step has run_always: true (always executes regardless).
        # Skip one step at a time so run_always steps later in the list still get reached.
        run_always = step.get("run_always", False)
        if not run_always and not self.last_step_result:
            log.info("[Runner] [skip] {}".format(self._step_label(step)))
            self._advance_step(False)
            return "running"

        now = time.time()
        if self.step_start_time is None:
            self.step_start_time = now

        if step_type == "match_click":
            template = step.get("template")
            template_array = step.get("template_array")  # list of templates to try in order
            threshold = step.get("threshold", 0.75)
            ratio_test = step.get("ratio_test")
            min_inliers = step.get("min_inliers")
            one_shot = step.get("one_shot", True)
            timeout_sec = step.get("timeout_sec") or 999
            # ── template_array: try each template in order, each gets its own timeout ───
            _tpls = template_array if isinstance(template_array, list) and template_array else None
            if _tpls:
                # Persist index across ticks so we don't re-try A from scratch every frame.
                tpl_idx = int(getattr(self, "_tpl_array_idx", 0) or 0)
                tpl_idx = max(0, tpl_idx)
                if tpl_idx >= len(_tpls):
                    log.info("[Runner] {} → false (all {} templates exhausted)".format(
                        self._step_label(step), len(_tpls)))
                    self._advance_step(False, step=step)
                    self._tpl_array_idx = 0
                    self._tpl_array_start_t = None
                    return "running"

                # Reset per-template timeout timer when we switch template
                cur_tpl = _tpls[tpl_idx]
                last_tpl = getattr(self, "_tpl_array_last_tpl", None)
                if last_tpl != cur_tpl or getattr(self, "_tpl_array_start_t", None) is None:
                    self._tpl_array_last_tpl = cur_tpl
                    self._tpl_array_start_t = time.time()
                    self._step_pos_cache = None

                template = cur_tpl  # override single-template path below
                # Use per-template timer for "timeout_sec"
                self.step_start_time = getattr(self, "_tpl_array_start_t", now)

            refresh_sleep_sec = step.get("refresh_sleep_sec", 1.0)
            max_clicks = step.get("max_clicks") or 999999
            click_interval_sec = step.get("click_interval_sec") or 0
            # fn_settings overrides (ClickTreasure and any function with these settings)
            _ov_max_clicks = self._fn_setting("max_clicks")
            if _ov_max_clicks is not None:
                try:
                    max_clicks = int(_ov_max_clicks)
                except (ValueError, TypeError):
                    pass
            _ov_interval = self._fn_setting("click_interval_sec")
            if _ov_interval is not None:
                try:
                    click_interval_sec = float(_ov_interval)
                except (ValueError, TypeError):
                    pass
            click_random_offset = step.get("click_random_offset") or 0   # legacy: pixels
            click_random_offset_x = step.get("click_random_offset_x")   # ratio of needle_w (e.g. 0.4 = ±40%)
            click_random_offset_y = step.get("click_random_offset_y")   # ratio of needle_h (e.g. 0.4 = ±40%)
            click_offset_x = step.get("click_offset_x") or 0.0
            click_offset_y = step.get("click_offset_y") or 0.0
            click_storm_sec      = step.get("click_storm_sec") or 0.0        # bypass screenshot: tight click loop for N seconds
            click_storm_max_rate  = step.get("click_storm_max_rate") or 0    # max clicks/s in storm (0 = unlimited)
            click_storm_offset_x  = int(step.get("click_storm_offset_x") or 0)   # ±px random X offset
            click_storm_offset_y  = int(step.get("click_storm_offset_y") or 0)   # ±px random Y offset
            _corner_cfg = step.get("click_storm_corner")                          # {offset_x, offset_y, every} or {x, y, every}
            if _corner_cfg:
                if "offset_x" in _corner_cfg or "offset_y" in _corner_cfg:
                    _cox = _corner_cfg.get("offset_x", 0.05)
                    _coy = _corner_cfg.get("offset_y", 0.05)
                    _cpx = int(wincap.w * _cox)
                    _cpy = int(wincap.h * _coy)
                    click_storm_corner = wincap.get_screen_position((_cpx, _cpy))
                else:
                    click_storm_corner = (int(_corner_cfg["x"]), int(_corner_cfg["y"]))
            else:
                click_storm_corner = None
            click_storm_corner_every = int((_corner_cfg or {}).get("every", 1000))

            vision = self._get_vision(template)
            if not vision:
                log.warning("[Runner] Step {} skipped: template NOT FOUND or FAILED TO LOAD ({})".format(
                    self._step_label(step), template))
                self._advance_step(True, step=step)  # skip
                return "running"
            debug_click  = step.get("debug_click", False)
            debug_log    = step.get("debug_log", False)
            match_color  = step.get("match_color", False)   # True = BGR match (color-sensitive)
            color_match_tolerance  = step.get("color_match_tolerance")   # BGR mean-color distance cap
            ocr_name_region        = step.get("ocr_name_region")          # [x,y,w,h] ratios → OCR name to the left of truck
            cache_position = step.get("cache_position", False) or step.get("cache_frames", 0) > 0  # match once, reuse forever

            # ── ROI crop: restrict matching to a region centered on the expected button position ──
            # roi_center_x / roi_center_y: ratios [0..1] of the expected button CENTER (copy from YAML ROI in mouse log).
            # The crop window is auto-sized to roi_padding × template dimensions around that center.
            # roi_padding: how many template-widths/heights to extend in each direction (default 3.0).
            # search_region: {x, y, w, h} explicit override (ratios), for advanced use.
            _roi_offset = (0, 0)
            _roi_cx = step.get("roi_center_x")
            _roi_cy = step.get("roi_center_y")
            _explicit_roi = step.get("search_region")
            if _explicit_roi:
                # Explicit {x, y, w, h} region (ratios)
                _sh, _sw = screenshot.shape[:2]
                _rx  = max(0, int(_explicit_roi.get("x", 0.0) * _sw))
                _ry  = max(0, int(_explicit_roi.get("y", 0.0) * _sh))
                _rx2 = min(_sw, _rx + max(1, int(_explicit_roi.get("w", 1.0) * _sw)))
                _ry2 = min(_sh, _ry + max(1, int(_explicit_roi.get("h", 1.0) * _sh)))
                search_img  = screenshot[_ry:_ry2, _rx:_rx2]
                _roi_offset = (_rx, _ry)
            elif _roi_cx is not None and _roi_cy is not None:
                # Smart center-based ROI: auto-compute crop from template size
                _sh, _sw = screenshot.shape[:2]
                _scale   = get_global_scale()   # window_width / reference_width
                _nw_px   = max(1, int(vision.needle_w * _scale))   # template width  in screenshot pixels
                _nh_px   = max(1, int(vision.needle_h * _scale))   # template height in screenshot pixels
                _padding = float(step.get("roi_padding", 3.0))     # 3× template size each side
                _cx_px   = int(_roi_cx * _sw)
                _cy_px   = int(_roi_cy * _sh)
                _half_w  = int(_nw_px * _padding)
                _half_h  = int(_nh_px * _padding)
                _rx      = max(0, _cx_px - _half_w)
                _ry      = max(0, _cy_px - _half_h)
                _rx2     = min(_sw, _cx_px + _half_w)
                _ry2     = min(_sh, _cy_px + _half_h)
                search_img  = screenshot[_ry:_ry2, _rx:_rx2]
                _roi_offset = (_rx, _ry)
                if debug_log:
                    log.info("[Runner] {} ROI: center=({:.2f},{:.2f}) crop=({},{})→({},{}) tpl={}×{}px".format(
                        self._step_label(step), _roi_cx, _roi_cy, _rx, _ry, _rx2, _ry2, _nw_px, _nh_px))
            else:
                search_img = screenshot

            # Run matchTemplate only on first hit, then reuse cached position for all subsequent clicks.
            _cached_pos = getattr(self, '_step_pos_cache', None)
            if cache_position and _cached_pos is not None:
                points = [_cached_pos]  # use cached position, skip matchTemplate
            else:
                dbg = 'info' if (debug_click or debug_log) else None
                meta_list = None
                if match_color:
                    result = vision.find(search_img, threshold=threshold, debug_mode=dbg,
                                        is_color=True, debug_log=debug_log,
                                        color_tolerance=color_match_tolerance,
                                        ratio_test=ratio_test, min_inliers=min_inliers)
                    points = result[0] if isinstance(result, tuple) else result
                    meta_list = result[1] if isinstance(result, tuple) and len(result) > 1 else None
                else:
                    points = vision.find(search_img, threshold=threshold, debug_mode=dbg, debug_log=debug_log, ratio_test=ratio_test, min_inliers=min_inliers)
                # Translate ROI-local coords back to full screenshot coords
                if points and _roi_offset != (0, 0):
                    ox, oy = _roi_offset
                    _shifted = []
                    for pt in points:
                        cx, cy = pt[0] + ox, pt[1] + oy
                        _shifted.append((cx, cy) + tuple(pt[2:]))
                    points = _shifted
                if points:
                    self._step_pos_cache = points[0]
                    # If this step is using template_array, stop after first successful template.
                    if _tpls:
                        self._tpl_array_idx = 0
                        self._tpl_array_start_t = None
                        self._tpl_array_last_tpl = None
                elif debug_log:
                    elapsed = now - self.step_start_time
                    log.info("[Runner] {} → not found (elapsed={:.1f}s)".format(self._step_label(step), elapsed))
                # Fallback: debug_click YellowTruckSmall but 0 passed color filter → save best match anyway (once)
                if not points and debug_click and "YellowTruckSmall" in (template or ""):
                    if not getattr(self, '_debug_yellow_fallback_saved', False):
                        _fallback = vision.find(screenshot, threshold=0.5, debug_mode=None)
                        if _fallback:
                            self._debug_yellow_fallback_saved = True
                            rc = _fallback[0]
                            mw, mh = (rc[2], rc[3]) if len(rc) >= 4 else (vision.needle_w, vision.needle_h)
                            rcenter = (rc[0] + int(click_offset_x * mw),
                                       rc[1] + int(click_offset_y * mh))
                            _save_debug_image(screenshot, rc[:2], rcenter, mw, mh,
                                             "match_click", template)
                            log.info("[Runner] {} → 0 passed color filter, saved best match → debug_ocr/".format(
                                self._step_label(step)))
            # template_array: on per-template timeout, advance to next template and reset timer
            if not points and _tpls and (now - self.step_start_time >= timeout_sec):
                tpl_idx = int(getattr(self, "_tpl_array_idx", 0) or 0)
                cur_tpl = _tpls[tpl_idx] if 0 <= tpl_idx < len(_tpls) else None
                tpl_name = os.path.splitext(os.path.basename(cur_tpl))[0] if cur_tpl else str(tpl_idx)
                log.info("[Runner] {} [{}] → not found in {}s, trying next template".format(
                    self._step_label(step), tpl_name, timeout_sec))
                self._tpl_array_idx = tpl_idx + 1
                self._tpl_array_start_t = None
                self._tpl_array_last_tpl = None
                self._step_pos_cache = None
                return "running"
            # When 0 trucks found and track_tried+refresh: click refresh to reload list (unstick)
            if not points and step.get("track_tried") and step.get("refresh_template"):
                _last_refresh = getattr(self, "_last_zero_refresh_t", 0)
                if now - _last_refresh >= 2.0:  # throttle: max 1 refresh per 2s
                    _refresh_tpl = step.get("refresh_template")
                    _v_ref = self.vision_cache.get(_refresh_tpl) if _refresh_tpl else None
                    if _v_ref:
                        _ref_pts = _v_ref.find(screenshot, threshold=step.get("threshold", 0.75), debug_mode=None)
                        if _ref_pts:
                            _rsx, _rsy = wincap.get_screen_position(tuple(_ref_pts[0]))
                            pyautogui.click(_rsx, _rsy)
                            log.info("[Runner] {} → 0 trucks found → clicked refresh, sleeping {:.1f}s".format(self._step_label(step), refresh_sleep_sec))
                            time.sleep(refresh_sleep_sec)
                            self._tried_positions = []
                            self._rtr_cache = {}
                            self._rtr_page_logged = False
                            self.step_start_time = time.time()
                            self._last_zero_refresh_t = now
                            return "running"
            if points:
                # ── OCR player name for each truck (if configured) ────────────────────
                _point_names = {}
                if ocr_name_region and len(ocr_name_region) == 4:
                    _onr_x, _onr_y, _onr_w, _onr_h = ocr_name_region
                    for _pt in points:
                        cx, cy = _pt[0], _pt[1]
                        mw, mh = (_pt[2], _pt[3]) if len(_pt) >= 4 else (vision.needle_w, vision.needle_h)
                        _name = read_region_relative(
                            screenshot, cx, cy,
                            mw, mh,
                            x=_onr_x, y=_onr_y, w=_onr_w, h=_onr_h,
                        )
                        _point_names[tuple(_pt)] = (_name or "").strip()

                # ── require_text_in_region: keep trucks where a region has text-like edges ──
                # Uses Canny edge density — text produces many edges; bare background does not.
                # No OCR needed: just count edge pixels / total pixels vs min_edge_density.
                _rtr = step.get("require_text_in_region")
                if _rtr and points:
                    _rtr_x       = _rtr.get("x", -1.0)
                    _rtr_y       = _rtr.get("y", 0.0)
                    _rtr_w       = _rtr.get("w", 2.0)
                    _rtr_h       = _rtr.get("h", 0.8)
                    _rtr_density  = _rtr.get("min_edge_density", 0.05)
                    _rtr_canny_lo = _rtr.get("canny_low", 50)
                    _rtr_canny_hi = _rtr.get("canny_high", 150)
                    _rtr_debug    = _rtr.get("debug_save", False)
                    _rtr_top_k    = _rtr.get("top_k")
                    # Cache key: round coordinates to nearest 10px to absorb per-frame jitter
                    _rtr_key = frozenset((round(_p[0] / 10) * 10, round(_p[1] / 10) * 10) for _p in points)
                    if _rtr_key in self._rtr_cache:
                        _rtr_passed, _rtr_density_map = self._rtr_cache[_rtr_key]
                        points = _rtr_passed
                    else:
                        _passed = []
                        _rtr_density_map = {}
                        _rtr_log_parts = []
                        for _pt in points:
                            cx, cy = _pt[0], _pt[1]
                            mw, mh = (_pt[2], _pt[3]) if len(_pt) >= 4 else (vision.needle_w, vision.needle_h)
                            _crop = _crop_region_relative(screenshot, cx, cy,
                                                          mw, mh,
                                                          _rtr_x, _rtr_y, _rtr_w, _rtr_h)
                            if _crop is None:
                                continue
                            _gray  = cv.cvtColor(_crop, cv.COLOR_BGR2GRAY) if len(_crop.shape) == 3 else _crop
                            _edges = cv.Canny(_gray, _rtr_canny_lo, _rtr_canny_hi)
                            _density = float(_edges.sum()) / (255.0 * max(_edges.size, 1))
                            _ok = _density >= _rtr_density
                            _rtr_density_map[tuple(_pt)] = _density
                            _rtr_log_parts.append("({},{}) {}:{:.3f}".format(
                                _pt[0], _pt[1], "PASS" if _ok else "SKIP", _density))
                            if _rtr_debug:
                                import datetime
                                _ts = datetime.datetime.now().strftime("%H%M%S_%f")[:-3]
                                _tag = "pass" if _ok else "skip"
                                _dbg_dir = os.path.join(os.path.dirname(__file__), "debug_ocr")
                                os.makedirs(_dbg_dir, exist_ok=True)
                                cv.imwrite(os.path.join(_dbg_dir,
                                    "require_text_{}_{}_{},{}.png".format(_tag, _ts, _pt[0], _pt[1])), _crop)
                                cv.imwrite(os.path.join(_dbg_dir,
                                    "require_text_edges_{}_{}_{},{}.png".format(_tag, _ts, _pt[0], _pt[1])), _edges)
                            if _ok:
                                _passed.append(_pt)
                        if _rtr_top_k:
                            # top_k: skip threshold, pick the K trucks with highest edge density
                            _all_sorted = sorted(points, key=lambda p: _rtr_density_map.get(tuple(p), 0), reverse=True)
                            _passed = _all_sorted[:_rtr_top_k]
                        elif len(_passed) == 0:
                            pass  # all filtered out by threshold
                        if not self._rtr_page_logged:
                            log.info("[Runner] {} → require_text_in_region ({} trucks, threshold={:.3f}): {}".format(
                                self._step_label(step), len(points), _rtr_density,
                                " | ".join(_rtr_log_parts)))
                        self._rtr_cache[_rtr_key] = (_passed, _rtr_density_map)
                        points = _passed
                    self._rtr_density_map_last = _rtr_density_map

                # ── require_bright_region: keep trucks where a region is bright (light square) ──
                # Used to detect avatar frames (white/light square) above trucks.
                # Checks mean grayscale brightness of a relative region vs min_mean (0-255).
                _rbr = step.get("require_bright_region")
                if _rbr and points:
                    _rbr_x   = _rbr.get("x", -0.5)
                    _rbr_y   = _rbr.get("y", -2.0)
                    _rbr_w   = _rbr.get("w", 1.0)
                    _rbr_h   = _rbr.get("h", 1.5)
                    _rbr_min = _rbr.get("min_mean", 160)
                    _passed = []
                    for _pt in points:
                        cx, cy = _pt[0], _pt[1]
                        mw, mh = (_pt[2], _pt[3]) if len(_pt) >= 4 else (vision.needle_w, vision.needle_h)
                        _crop = _crop_region_relative(screenshot, cx, cy,
                                                      mw, mh,
                                                      _rbr_x, _rbr_y, _rbr_w, _rbr_h)
                        if _crop is None:
                            continue
                        _gray = cv.cvtColor(_crop, cv.COLOR_BGR2GRAY) if len(_crop.shape) == 3 else _crop
                        _mean = float(_gray.mean())
                        if _mean >= _rbr_min:
                            _passed.append(_pt)
                        else:
                            log.info("[Runner] {} ({},{}) → require_bright_region SKIP"
                                     " mean_brightness={:.1f} < {:.1f}".format(
                                self._step_label(step), _pt[0], _pt[1], _mean, _rbr_min))
                    points = _passed

                # ── track_tried: skip positions already clicked in this cycle ──────────
                track_tried = step.get("track_tried", False)
                _rtr_density_map = getattr(self, '_rtr_density_map_last', {})
                if track_tried:
                    _tol = step.get("tried_tolerance_px", 30)
                    untried = [pt for pt in points if not any(
                        abs(pt[0] - tp[0]) < _tol and abs(pt[1] - tp[1]) < _tol
                        for tp in self._tried_positions
                    )]
                    # Only log the full truck list once per refresh cycle.
                    # Reset _rtr_page_logged when refresh is clicked.
                    if not self._rtr_page_logged:
                        if meta_list is not None and len(meta_list) == len(points):
                            parts = []
                            for p, m in zip(points, meta_list):
                                s = m.get("score")
                                extras = []
                                _pname = _point_names.get(tuple(p), "")
                                if _pname:
                                    extras.append("name={}".format(_pname))
                                if m.get("region_bgr") is not None:
                                    b, g, r = m["region_bgr"]
                                    extras.append("B={:.0f} G={:.0f} R={:.0f}".format(b, g, r))
                                if m.get("dominant_color") is not None:
                                    extras.append("color={}".format(m["dominant_color"]))
                                if m.get("color_dist") is not None:
                                    extras.append("dist={:.0f}".format(m["color_dist"]))
                                if m.get("mean_sat") is not None:
                                    extras.append("sat={:.0f}".format(m["mean_sat"]))
                                if m.get("median_hue") is not None:
                                    extras.append("hue={:.0f}".format(m["median_hue"]))
                                if m.get("in_range_frac") is not None:
                                    extras.append("frac={:.0f}%".format(m["in_range_frac"] * 100))
                                parts.append("({},{}) score={:.3f} [{}]".format(
                                    p[0], p[1], s, " ".join(extras)))
                            _pts_str = " | ".join(parts)
                        else:
                            parts = []
                            for p in points:
                                _pname = _point_names.get(tuple(p), "")
                                _d = _rtr_density_map.get(tuple(p))
                                _d_str = " density={:.3f}".format(_d) if _d is not None else ""
                                parts.append("({},{}){}{}".format(
                                    p[0], p[1], _d_str,
                                    " name={}".format(_pname) if _pname else ""))
                            _pts_str = ", ".join(parts)
                        log.info("[Runner] {} → found {} truck(s): [{}] | tried={} untried={}".format(
                            self._step_label(step), len(points), _pts_str,
                            len(self._tried_positions), len(untried)))
                        self._rtr_page_logged = True
                    if not untried:
                        _refresh_tpl = step.get("refresh_template")
                        _v_ref = self.vision_cache.get(_refresh_tpl) if _refresh_tpl else None
                        if _v_ref:
                            _ref_pts = _v_ref.find(screenshot, threshold=threshold, debug_mode=None)
                            if _ref_pts:
                                _rsx, _rsy = wincap.get_screen_position(tuple(_ref_pts[0]))
                                pyautogui.click(_rsx, _rsy)
                                log.info("[Runner] {} → all {} truck(s) tried → clicked refresh, sleeping {:.1f}s".format(
                                    self._step_label(step), len(self._tried_positions), refresh_sleep_sec))
                                time.sleep(refresh_sleep_sec)
                                self._tried_positions = []
                                self._rtr_cache = {}
                                self._rtr_page_logged = False
                                self.step_start_time = time.time()
                                return "running"
                        log.info("[Runner] {} → all tried, no refresh available → abort".format(self._step_label(step)))
                        self._advance_step(False, step=step)
                        return "running"
                    points = untried
                # ─────────────────────────────────────────────────────────────────────
                # matched mw and mh are used to scale click_offset_x/y
                cx, cy, mw, mh = (points[0][0], points[0][1], points[0][2], points[0][3]) if len(points[0]) >= 4 else (points[0][0], points[0][1], vision.needle_w, vision.needle_h)
                center = [cx, cy]
                center[0] += int(click_offset_x * mw)
                center[1] += int(click_offset_y * mh)
                if click_random_offset_x is not None:
                    rx = int(click_random_offset_x * mw)
                    if rx > 0:
                        center[0] += random.randint(-rx, rx)
                elif click_random_offset > 0:
                    center[0] += random.randint(-click_random_offset, click_random_offset)
                if click_random_offset_y is not None:
                    ry = int(click_random_offset_y * mh)
                    if ry > 0:
                        center[1] += random.randint(-ry, ry)
                elif click_random_offset > 0:
                    center[1] += random.randint(-click_random_offset, click_random_offset)
                sx, sy = wincap.get_screen_position(tuple(center))
                raw_center = (cx, cy)
                if debug_log:
                    log.info("[Runner] {} | raw_center=({},{}) needle=({}x{}) matched=({}x{}) offset=({},{}) after_offset=({},{}) screen=({},{})".format(
                        self._step_label(step), raw_center[0], raw_center[1],
                        vision.needle_w, vision.needle_h, mw, mh,
                        click_offset_x, click_offset_y,
                        center[0], center[1], sx, sy))
                # For YellowTruckSmall: save every truck click (timestamp in filename prevents overwrite)
                # For other steps: save only once per step activation to avoid spam
                if debug_click:
                    _is_yellow_truck = "YellowTruckSmall" in (template or "")
                    if _is_yellow_truck or not getattr(self, '_debug_click_saved', False):
                        if not _is_yellow_truck:
                            self._debug_click_saved = True
                        _crop_path = _save_debug_image(screenshot, raw_center, tuple(center),
                                          mw, mh, "match_click", template)
                        if _is_yellow_truck and _crop_path:
                            self._last_truck_crop_path = _crop_path
                # click_storm: start FastClicker once, then return immediately every tick.
                # No screenshot is taken while the clicker is running — zero overhead.
                if click_storm_sec > 0 and not one_shot:
                    if not self._fast_clicker.is_running:
                        # First time: start the clicker at detected position
                        self._storm_start_t = time.time()
                        self._fast_clicker.start(sx, sy,
                            rate=click_storm_max_rate or 0,
                            offset_x=click_storm_offset_x,
                            offset_y=click_storm_offset_y,
                            corner_pos=click_storm_corner,
                            corner_every=click_storm_corner_every)
                        log.info("[Runner] {} → storm started at ({},{}) rate={}".format(
                            self._step_label(step), sx, sy, click_storm_max_rate or "unlimited"))
                    else:
                        # Clicker already running: check elapsed time and click count
                        _n = self._fast_clicker.click_count
                        _elapsed = max(0.001, time.time() - getattr(self, '_storm_start_t', time.time()))
                        if _n >= max_clicks or _elapsed >= click_storm_sec:
                            self._fast_clicker.stop()
                            self.step_click_count += _n
                            _rate = round(_n / _elapsed, 0)
                            log.info("[Runner] {} | storm={} clicks in {:.1f}s (~{:.0f}/s)".format(
                                self._step_label(step), _n, _elapsed, _rate))
                            log.info("[Runner] {} → true (clicked {})".format(self._step_label(step), self.step_click_count))
                            self._advance_step(True, step=step)
                    return "running"

                # Non-storm path: move cursor with pynput then click
                try:
                    _mouse_ctrl.position = (sx, sy)
                    time.sleep(0.05) # Tang len 0.05s de chac chan game nhan ra chuot dang o tren nut
                except Exception as e:
                    log.warning("[Runner] Failed to set mouse position: {}".format(e))
                    return "running"

                if debug_log:
                    actual = _mouse_ctrl.position
                    log.info("[Runner] {} | intended=({},{}) actual=({},{}) diff=({},{})".format(
                        self._step_label(step), sx, sy, int(actual[0]), int(actual[1]),
                        int(actual[0] - sx), int(actual[1] - sy)))

                # Force focus game window right before clicking to ensure it receives input
                if hasattr(wincap, 'focus_window'):
                    wincap.focus_window()

                burst_start = time.time()
                if not one_shot and click_interval_sec == 0:
                    # Burst mode: click as many times as possible within 50ms this frame
                    burst_end = burst_start + 0.05
                    while time.time() < burst_end and self.step_click_count < max_clicks:
                        # Improved click reliability with pynput
                        _mouse_ctrl.press(Button.left)
                        time.sleep(0.03) # Tăng nhẹ thời gian hold trong burst
                        _mouse_ctrl.release(Button.left)
                        self.step_click_count += 1
                else:
                    # Improved click reliability with pynput (standard click)
                    _mouse_ctrl.press(Button.left)
                    time.sleep(0.1) # Tăng lên 0.1s để game chắc chắn nhận diện được
                    _mouse_ctrl.release(Button.left)
                    self.step_click_count += 1

                if not one_shot:
                    click_t = time.time()
                    last_t = getattr(self, '_step_last_click_t', None)
                    if debug_log and last_t is not None:
                        frame_interval = click_t - last_t
                        actual_rate = round(self.step_click_count / max(0.001, click_t - self.step_start_time), 1)
                        log.info("[Runner] {} | frame={:.3f}s total={} rate={:.1f}/s".format(
                            self._step_label(step), frame_interval,
                            self.step_click_count, actual_rate))
                    self._step_last_click_t = click_t
                    # click_interval_sec > 0: sleep the remaining time to hit target rate
                    if click_interval_sec > 0 and last_t is not None:
                        elapsed = click_t - last_t
                        remaining = click_interval_sec - elapsed
                        if remaining > 0:
                            time.sleep(remaining)
                if step.get("track_tried", False):
                    self._tried_positions.append(tuple(points[0]))
                if one_shot:
                    self._last_click_pos = tuple(center)
                    _click_truck_pt = tuple(points[0])
                    _click_density = getattr(self, '_rtr_density_map_last', {}).get(_click_truck_pt)
                    _click_density_str = " density={:.3f}".format(_click_density) if _click_density is not None else ""
                    log.info("[Runner] {} → true (clicked position ({},{}){})"  .format(
                        self._step_label(step), _click_truck_pt[0], _click_truck_pt[1], _click_density_str))
                    self._advance_step(True, step=step)
                    return "running"
                log.info("[Runner] {} clicking... (count {})".format(self._step_label(step), self.step_click_count))
                if self.step_click_count >= max_clicks:
                    log.info("[Runner] {} → true (clicked {})".format(self._step_label(step), self.step_click_count))
                    self._advance_step(True, step=step)
                    return "running"
            if now - self.step_start_time >= timeout_sec:
                log.info("[Runner] {} → false (not found in {}s)".format(self._step_label(step), timeout_sec))
                self._advance_step(False, step=step)
            return "running"

        if step_type == "match_move":
            template     = step.get("template")
            threshold    = step.get("threshold", 0.75)
            timeout_sec  = step.get("timeout_sec") or 999
            click_offset_x = step.get("click_offset_x") or 0.0
            click_offset_y = step.get("click_offset_y") or 0.0
            debug_click = step.get("debug_click", False)  # save debug image once
            debug_log   = step.get("debug_log", False)    # log click coords

            vision = self.vision_cache.get(template)
            if not vision:
                self._advance_step(True)
                return "running"
            points = vision.find(screenshot, threshold=threshold, debug_mode='info' if (debug_click or debug_log) else None)
            if points:
                rc = points[0]
                cx, cy, mw, mh = (rc[0], rc[1], rc[2], rc[3]) if len(rc) >= 4 else (rc[0], rc[1], vision.needle_w, vision.needle_h)
                raw_center = (cx, cy)
                center = [cx, cy]
                center[0] += int(click_offset_x * mw)
                center[1] += int(click_offset_y * mh)
                sx, sy = wincap.get_screen_position(tuple(center))
                if debug_log:
                    log.info("[Runner] {} | raw_center=({},{}) needle=({}x{}) matched=({}x{}) offset=({},{}) after_offset=({},{}) screen=({},{})".format(
                        self._step_label(step), raw_center[0], raw_center[1],
                        vision.needle_w, vision.needle_h, mw, mh,
                        click_offset_x, click_offset_y,
                        center[0], center[1], sx, sy))
                if debug_click and not getattr(self, '_debug_click_saved', False):
                    self._debug_click_saved = True
                    _save_debug_image(screenshot, raw_center, tuple(center),
                                      mw, mh, "match_move", template)
                pyautogui.moveTo(sx, sy)
                if debug_log:
                    actual = pyautogui.position()
                    log.info("[Runner] {} → true | intended=({},{}) actual=({},{}) diff=({},{})".format(
                        self._step_label(step), sx, sy, actual.x, actual.y,
                        actual.x - sx, actual.y - sy))
                self._advance_step(True)
                return "running"
            if now - self.step_start_time >= timeout_sec:
                log.info("[Runner] {} → false (not found in {}s)".format(self._step_label(step), timeout_sec))
                self._advance_step(False)
            return "running"

        if step_type == "match_multi_click":
            # Find ALL visible instances of template and click each one, then advance.
            # If none found within timeout_sec, advance anyway.
            template           = step.get("template")
            threshold          = step.get("threshold", 0.75)
            timeout_sec        = step.get("timeout_sec") or 10
            click_interval_sec = step.get("click_interval_sec", 0.15)

            vision = self._get_vision(template)
            if not vision:
                self._advance_step(True)
                return "running"
            points = vision.find(screenshot, threshold=threshold, debug_mode=None)
            if points:
                for pt in points:
                    sx, sy = wincap.get_screen_position((pt[0], pt[1]))
                    try:
                        _mouse_ctrl.position = (sx, sy)
                        time.sleep(0.05)
                    except: pass
                    
                    if hasattr(wincap, 'focus_window'):
                        wincap.focus_window()
                        
                    _mouse_ctrl.press(Button.left)
                    time.sleep(0.1)
                    _mouse_ctrl.release(Button.left)
                    if click_interval_sec > 0:
                        time.sleep(click_interval_sec)
                log.info("[Runner] {} → true (clicked {} match(es))".format(self._step_label(step), len(points)))
                self._advance_step(True)
                return "running"
            if now - self.step_start_time >= timeout_sec:
                log.info("[Runner] {} → false (not found in {}s)".format(self._step_label(step), timeout_sec))
                self._advance_step(False)
            return "running"

        if step_type == "sleep":
            duration = step.get("duration_sec", 0)
            if now - self.step_start_time >= duration:
                log.info("[Runner] {} → true".format(self._step_label(step)))
                self._advance_step(True)
            return "running"

        if step_type == "send_zalo":
            # UI override from fn_settings (Settings button); fallback to YAML step
            _msg_ov = self._fn_setting("send_zalo_message")
            message = (_msg_ov if _msg_ov is not None and str(_msg_ov).strip() else "") or (step.get("message") or "")
            _rcv_ov = self._fn_setting("send_zalo_receiver_name")
            receiver_name = (_rcv_ov if _rcv_ov is not None and str(_rcv_ov).strip() else None) or step.get("receiver_name")
            _int_ov = self._fn_setting("send_zalo_repeat_interval_sec")
            if _int_ov is not None:
                try:
                    repeat_interval_sec = int(_int_ov)
                except (ValueError, TypeError):
                    repeat_interval_sec = step.get("repeat_interval_sec") or 0
            else:
                repeat_interval_sec = step.get("repeat_interval_sec") or 0
            trigger_cb = getattr(self, "trigger_active_cb", None)

            def _do_send():
                if _zalo_web_clicker:
                    _zalo_web_clicker.send_zalo_message(message, receiver_name=receiver_name, logger=log)
                else:
                    log.warning("[Runner] send_zalo: zalo_web_clicker not available, skip")

            if repeat_interval_sec > 0 and trigger_cb and callable(trigger_cb):
                def _repeat_loop():
                    _do_send()
                    next_send_at = time.time() + repeat_interval_sec
                    paused_ref = getattr(self, "bot_paused", None)
                    was_paused = bool(paused_ref and paused_ref.get("paused", False))
                    while trigger_cb():
                        if paused_ref and paused_ref.get("paused", False):
                            was_paused = True
                            time.sleep(1)
                            continue
                        if was_paused:
                            # Reset counter: gui ngay lan tiep theo khi resume (trigger van dang active)
                            next_send_at = time.time()
                            was_paused = False
                            log.info("[Runner] send_zalo repeat counter reset (resumed), will send next tick")
                        if time.time() >= next_send_at:
                            _do_send()
                            next_send_at = time.time() + repeat_interval_sec
                            log.info("[Runner] send_zalo repeat (interval={}s)".format(repeat_interval_sec))
                        time.sleep(1)
                import threading
                t = threading.Thread(target=_repeat_loop, daemon=True)
                t.start()
                log.info("[Runner] {} → started (repeat every {}s while trigger active)".format(
                    self._step_label(step), repeat_interval_sec))
            else:
                import threading
                threading.Thread(target=_do_send, daemon=True).start()
                log.info("[Runner] {} → sent once".format(self._step_label(step)))
            self._advance_step(True)
            return "running"

        if step_type == "click_position":
            # Có thể lấy (x,y) từ fn_settings: position_setting_key + positions map (key -> [x,y])
            ox, oy = None, None
            setting_key = step.get("position_setting_key")
            positions_map = step.get("positions")  # e.g. {"8h": [0.75, 0.31], "24h": [0.75, 0.40], "3d": [0.75, 0.49]}
            if setting_key and positions_map and isinstance(positions_map, dict):
                val = self._fn_setting(setting_key)
                key = (str(val).strip().lower() if val is not None else "") or str(step.get("default", "")).strip().lower()
                if key in positions_map:
                    pos = positions_map[key]
                    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        ox, oy = float(pos[0]), float(pos[1])
                if ox is None and "default" in step:
                    default_key = str(step.get("default", "")).strip().lower()
                    if default_key in positions_map:
                        pos = positions_map[default_key]
                        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                            ox, oy = float(pos[0]), float(pos[1])
            if ox is None:
                ox = step.get("x", step.get("offset_x", 0.15))
                oy = step.get("y", step.get("offset_y", 0.15))
            px = int(wincap.w * ox)
            py = int(wincap.h * oy)
            sx, sy = wincap.get_screen_position((px, py))
            try:
                _mouse_ctrl.position = (sx, sy)
                time.sleep(0.05)
            except: pass
            
            if hasattr(wincap, 'focus_window'):
                wincap.focus_window()
                
            _mouse_ctrl.press(Button.left)
            time.sleep(0.1)
            _mouse_ctrl.release(Button.left)
            if setting_key and positions_map and ox is not None:
                log.info("[Runner] {} → true ({} → x={}, y={})".format(
                    self._step_label(step),
                    str(self._fn_setting(setting_key) or step.get("default", "")).strip().lower(),
                    round(ox, 2), round(oy, 2)))
            else:
                log.info("[Runner] {} → true".format(self._step_label(step)))
            self._advance_step(True)
            return "running"

        if step_type == "wait_until_match":
            template = step.get("template")
            threshold = step.get("threshold", 0.75)
            timeout_sec = step.get("timeout_sec") or 30
            vision = self._get_vision(template)
            if not vision:
                self._advance_step(True)
                return "running"
            points = vision.find(screenshot, threshold=threshold, debug_mode=None)
            if points:
                log.info("[Runner] {} → true".format(self._step_label(step)))
                self._advance_step(True)
                return "running"
            if now - self.step_start_time >= timeout_sec:
                log.info("[Runner] {} → false (not found in {}s)".format(self._step_label(step), timeout_sec))
                self._advance_step(False)
            return "running"

        if step_type == "set_level":
            # OCR-based: read "Lv.X" from screen, click Plus/Minus to reach target_level.
            target_level = step.get("target_level", 10)
            # fn_settings override: UI-configurable target level per function
            _fn_override = self.fn_settings.get(self.function_name or "", {})
            if "target_level" in _fn_override:
                try:
                    target_level = int(_fn_override["target_level"])
                except (ValueError, TypeError):
                    pass
            level_roi = step.get("level_roi")
            level_anchor_template = step.get("level_anchor_template")
            level_anchor_offset = step.get("level_anchor_offset")
            level_ocr_region  = step.get("level_ocr_region")   # {x, y, w, h} ratios of anchor template
            plus_template = step.get("plus_template")
            minus_template = step.get("minus_template")
            threshold = step.get("threshold", 0.75)
            timeout_sec = step.get("timeout_sec") or 30
            click_interval = step.get("click_interval_sec", 0.3)
            min_level      = step.get("min_level", 1)
            max_level      = step.get("max_level", 99)
            if not plus_template or not minus_template:
                self._advance_step(True)
                return "running"
            if now - self.step_start_time >= timeout_sec:
                log.info("[Runner] set_level: timeout before reaching Lv.{}".format(target_level))
                self._advance_step(False)
                return "running"
            vision_plus = self.vision_cache.get(plus_template)
            vision_minus = self.vision_cache.get(minus_template)
            if not vision_plus or not vision_minus:
                self._advance_step(True)
                return "running"
            # Resolve anchor center from template
            anchor_center = None
            anchor_needle_w = None
            anchor_needle_h = None
            if level_anchor_template:
                v_anchor = self._get_vision(level_anchor_template)
                if v_anchor:
                    pts = v_anchor.find(screenshot, threshold=threshold, debug_mode=None)
                    if pts:
                        pt = pts[0]
                        anchor_center   = (int(pt[0]), int(pt[1]))
                        anchor_needle_w = pt[2] if len(pt) >= 4 else v_anchor.needle_w
                        anchor_needle_h = pt[3] if len(pt) >= 4 else v_anchor.needle_h
            # OCR: read current level
            # Mode A (new): level_ocr_region — absolute screen-ratio coords
            #   x, y = top-left corner of OCR region (0.0~1.0 of game window)
            #   w, h = size of OCR region (0.0~1.0 of game window)
            #   No anchor template needed — just pure screen coords
            debug_save = step.get("debug_save_roi", False)
            current = None
            if level_ocr_region:
                h_img, w_img = screenshot.shape[:2]
                rx = level_ocr_region.get("x", 0.0)
                ry = level_ocr_region.get("y", 0.0)
                rw = level_ocr_region.get("w", 0.1)
                rh = level_ocr_region.get("h", 0.05)
                px = max(0, int(rx * w_img))
                py = max(0, int(ry * h_img))
                pw = max(1, min(int(rw * w_img), w_img - px))
                ph = max(1, min(int(rh * h_img), h_img - py))
                roi = screenshot[py:py + ph, px:px + pw]
                # 540x960: ROI nhỏ, EasyOCR dễ mất số. Upscale ROI theo reference (như 1080p) trước khi OCR.
                scale = get_global_scale()
                if scale and 0 < scale < 1.0:
                    up = 1.0 / scale
                    nw = max(1, int(roi.shape[1] * up))
                    nh = max(1, int(roi.shape[0] * up))
                    roi = cv.resize(roi, (nw, nh), interpolation=cv.INTER_CUBIC)
                dbg_lbl = "debug_set_level_roi" if debug_save else None
                from ocr_easyocr import read_region_easy as _ocr_easy
                raw_text = _ocr_easy(roi, digits_only=False, debug_label=dbg_lbl)
                if debug_save:
                    log.info("[Runner] set_level: ROI crop saved to debug_ocr/debug_set_level_roi_raw.png")
                if raw_text:
                    current = _parse_level(raw_text, (min_level, max_level))
                    if current is not None:
                        log.info("[Runner] set_level: Lv.{} from {!r}".format(current, raw_text))
                    else:
                        log.info("[Runner] set_level: no level match from {!r}".format(raw_text))
            else:
                # Mode B (legacy): anchor template + pixel offset or level_roi
                current = _read_level_from_roi(
                    screenshot, level_roi or [0, 0, 0.3, 0.1], wincap,
                    anchor_center, level_anchor_offset,
                    debug_save_path="debug_set_level_roi.png" if debug_save else None,
                    level_range=(min_level, max_level),
                )
                if debug_save:
                    self._set_level_debug_saved = True
                    log.info("[Runner] set_level: ROI saved to debug_set_level_roi.png")
            if current is None:
                if not getattr(self, "_set_level_warned", False):
                    self._set_level_warned = True
                    log.info("[Runner] set_level: OCR cannot read level — check level_ocr_region / level_anchor_offset in YAML")
                return "running"
            self._set_level_warned = False

            if current == target_level:
                log.info("[Runner] set_level: already at Lv.{}, done".format(target_level))
                self._advance_step(True)
                return "running"
            if current < target_level:
                pts = vision_plus.find(screenshot, threshold=threshold, debug_mode=None)
                if pts:
                    sx, sy = wincap.get_screen_position((pts[0][0], pts[0][1]))
                    _mouse_ctrl.position = (sx, sy)
                    _mouse_ctrl.press(Button.left)
                    time.sleep(0.05)
                    _mouse_ctrl.release(Button.left)
                    log.info("[Runner] set_level: Lv.{} -> click Plus (target Lv.{})".format(current, target_level))
                    time.sleep(click_interval)
                else:
                    log.info("[Runner] set_level: Plus greyed at Lv.{} (max reached), proceeding".format(current))
                    self._advance_step(True)
                return "running"
            if current > target_level:
                pts = vision_minus.find(screenshot, threshold=threshold, debug_mode=None)
                if pts:
                    sx, sy = wincap.get_screen_position((pts[0][0], pts[0][1]))
                    pyautogui.click(sx, sy)
                    log.info("[Runner] set_level: Lv.{} -> click Minus (target Lv.{})".format(current, target_level))
                    time.sleep(click_interval)
                else:
                    log.info("[Runner] set_level: Minus greyed at Lv.{} (min reached), proceeding".format(current))
                    self._advance_step(True)
                return "running"

        if step_type == "click_unless_visible":
            # If visible_template is found on screen -> skip (already on right screen).
            # If NOT found -> click click_template to navigate there, then advance.
            visible_template = step.get("visible_template")
            click_template   = step.get("click_template")
            threshold        = step.get("threshold", 0.75)
            timeout_sec      = step.get("timeout_sec", 3)
            v_check = self.vision_cache.get(visible_template) if visible_template else None
            if v_check and v_check.find(screenshot, threshold=threshold, debug_mode=None):
                log.info("[Runner] {} → true (visible, skip nav)".format(self._step_label(step)))
                self._advance_step(True)
                return "running"
            if now - self.step_start_time >= timeout_sec:
                v_nav = self.vision_cache.get(click_template) if click_template else None
                if v_nav:
                    pts = v_nav.find(screenshot, threshold=threshold, debug_mode=None)
                    if pts:
                        sx, sy = wincap.get_screen_position((pts[0][0], pts[0][1]))
                        pyautogui.click(sx, sy)
                        log.info("[Runner] {} → true (not visible, clicked nav)".format(self._step_label(step)))
                    else:
                        log.info("[Runner] {} → true (not visible, nav absent too)".format(self._step_label(step)))
                self._advance_step(True)
            return "running"

        if step_type == "key_press":
            key = step.get("key", "")
            if key:
                pyautogui.press(key)
            log.info("[Runner] {} → true".format(self._step_label(step)))
            self._advance_step(True)
            return "running"

        if step_type == "type_text":
            text = str(step.get("text", ""))
            # Resolve ${ENV_VAR} placeholders — fn_settings takes priority over os.environ
            def _resolve_var(m):
                env_key = m.group(1)
                # Map known env vars to fn_settings keys
                _env_to_setting = {"PIN_PASSWORD": "password"}
                setting_key = _env_to_setting.get(env_key)
                if setting_key:
                    from_settings = self._fn_setting(setting_key)
                    if from_settings is not None and str(from_settings).strip():
                        return str(from_settings)
                return os.environ.get(env_key, "")
            text = re.sub(r"\$\{([^}]+)\}", _resolve_var, text)
            interval = step.get("interval_sec", 0.1)
            if text:
                pyautogui.write(text, interval=interval)
                log.info("[Runner] {} → true ({} chars)".format(self._step_label(step), len(text)))
            else:
                log.info("[Runner] {} → true (empty — check .env / ${{}} var name)".format(self._step_label(step)))
            self._advance_step(True)
            return "running"

        if step_type == "match_count":
            # Returns true if template appears >= count times within timeout_sec, false otherwise.
            # Does NOT click anything.
            template    = step.get("template")
            count       = step.get("count", 1)
            threshold   = step.get("threshold", 0.75)
            timeout_sec = step.get("timeout_sec") or 10
            debug_save  = step.get("debug_save", False)
            vision = self.vision_cache.get(template)
            if not vision:
                self._advance_step(True)
                return "running"
            match_color = step.get("match_color", False)
            debug_log   = step.get("debug_log", False)
            points = vision.find(screenshot, threshold=threshold, debug_mode=None,
                                 is_color=match_color, debug_log=debug_log, multi=True)
            found = len(points) if points else 0
            if found >= count:
                log.info("[Runner] {} → true (found {}/{})".format(self._step_label(step), found, count))
                self._advance_step(True)
                return "running"
            if now - self.step_start_time >= timeout_sec:
                log.info("[Runner] {} → false (found {}/{}, timeout {}s)".format(
                    self._step_label(step), found, count, timeout_sec))
                if debug_save:
                    try:
                        os.makedirs("debug", exist_ok=True)
                        ts_str = time.strftime("%Y%m%d_%H%M%S")
                        tpl_name = os.path.splitext(os.path.basename(template))[0]
                        # Save screenshot
                        shot_path = os.path.join("debug", "match_count_{}_{}_screenshot.png".format(tpl_name, ts_str))
                        cv.imwrite(shot_path, screenshot)
                        # Save template for comparison
                        tpl_path = os.path.join("debug", "match_count_{}_{}_template.png".format(tpl_name, ts_str))
                        cv.imwrite(tpl_path, vision.needle_img)
                        log.info("[Runner] match_count debug saved: screenshot={}x{} template={}x{} -> {}".format(
                            screenshot.shape[1], screenshot.shape[0],
                            vision.needle_w, vision.needle_h,
                            shot_path))
                    except Exception as e:
                        log.info("[Runner] match_count debug save failed: {}".format(e))
                on_fail_goto = step.get("on_fail_goto")
                max_retries  = step.get("max_retries")  # None = unlimited retries
                if on_fail_goto is not None:
                    if max_retries is None:
                        # No limit — retry indefinitely
                        log.info("[Runner] {} → retry (goto step {})".format(
                            self._step_label(step), on_fail_goto))
                        self._goto_step(int(on_fail_goto))
                        return "running"
                    else:
                        max_retries = int(max_retries)
                        cur_step_idx = self.step_index
                        retry_count = self._step_retry_counts.get(cur_step_idx, 0) + 1
                        if retry_count <= max_retries:
                            self._step_retry_counts[cur_step_idx] = retry_count
                            log.info("[Runner] {} → retry {}/{} (goto step {})".format(
                                self._step_label(step), retry_count, max_retries, on_fail_goto))
                            self._goto_step(int(on_fail_goto))
                            return "running"
                        log.info("[Runner] {} → max_retries ({}) reached → abort".format(
                            self._step_label(step), max_retries))
                self._advance_step(False)
            return "running"

        if step_type == "close_ui":
            # Vong lap: neu chua thay HQ & World thi click 1 cai (click_x, click_y), chup lai, kiem tra lai; thoat khi da thay.
            template = step.get("template")
            world_button = step.get("world_button")
            threshold = step.get("threshold", 0.75)
            debug_log = step.get("debug_log", False)
            match_color = step.get("match_color", False)
            color_tol = step.get("color_match_tolerance")
            click_x = float(step.get("click_x", 0.03))
            click_y = float(step.get("click_y", 0.08))
            max_tries = int(step.get("max_tries", 10))
            vision = self.vision_cache.get(template) if template else None
            vision_world = self.vision_cache.get(world_button) if world_button else None
            dbg_mode = "info" if debug_log else None
            _roi_offset = (0, 0)
            _roi_bounds = None
            _roi_cx = step.get("roi_center_x")
            _roi_cy = step.get("roi_center_y")
            if _roi_cx is not None and _roi_cy is not None and vision:
                _sh, _sw = screenshot.shape[:2]
                _scale = get_global_scale()
                _nw_px = max(1, int(vision.needle_w * _scale))
                _nh_px = max(1, int(vision.needle_h * _scale))
                _padding = float(step.get("roi_padding", 3.0))
                _cx_px = int(_roi_cx * _sw)
                _cy_px = int(_roi_cy * _sh)
                _half_w = int(_nw_px * _padding)
                _half_h = int(_nh_px * _padding)
                _rx = max(0, _cx_px - _half_w)
                _ry = max(0, _cy_px - _half_h)
                _rx2 = min(_sw, _cx_px + _half_w)
                _ry2 = min(_sh, _cy_px + _half_h)
                _roi_offset = (_rx, _ry)
                _roi_bounds = (_rx, _ry, _rx2, _ry2)

            def _close_ui_search_img(img):
                if _roi_bounds is None:
                    return img
                rx, ry, rx2, ry2 = _roi_bounds
                return img[ry:ry2, rx:rx2]

            def _close_ui_shift_points(pts):
                if not pts or _roi_offset == (0, 0):
                    return pts
                ox, oy = _roi_offset
                return [((p[0] + ox, p[1] + oy) + tuple(p[2:]) if len(p) > 2 else (p[0] + ox, p[1] + oy)) for p in pts]

            scr = screenshot
            for _try in range(max_tries):
                _search = _close_ui_search_img(scr)
                _phq = vision.find(_search, threshold=threshold, debug_mode=dbg_mode, debug_log=debug_log,
                                   is_color=bool(match_color), color_tolerance=color_tol) if vision else []
                _phq = _close_ui_shift_points(_phq if _phq else [])
                _pw = vision_world.find(_search, threshold=threshold, debug_mode=dbg_mode, debug_log=debug_log,
                                        is_color=bool(match_color), color_tolerance=color_tol) if vision_world else []
                _pw = _close_ui_shift_points(_pw if _pw else [])
                if _phq or _pw:
                    if debug_log:
                        log.info("[Runner] close_ui → true (thay HQ/World sau {} lan click)".format(_try))
                    break
                if _try < max_tries - 1:
                    if hasattr(wincap, "focus_window"):
                        wincap.focus_window(force=True)
                        time.sleep(0.05)
                    _px = int(wincap.w * click_x)
                    _py = int(wincap.h * click_y)
                    _sx, _sy = wincap.get_screen_position((_px, _py))
                    pyautogui.click(_sx, _sy)
                    time.sleep(1)
                    _fresh = wincap.get_screenshot() if hasattr(wincap, "get_screenshot") else None
                    if _fresh is not None:
                        scr = _fresh
                    time.sleep(0.3)
            log.info("[Runner] close_ui → true")
            self._advance_step(True)
            return "running"

        if step_type == "base_zoomout":
            # Luong: (1) Thay HQ -> click HQ -> chup lai: thay World thi zoom out; thay HQ thi click HQ them 1 lan roi zoom out.
            #        (2) Vua vo da thay World -> click World -> retry (tick sau tim HQ, click HQ, roi nhu (1)).
            #        (3) Khong thay ca hai -> scroll zoom out, advance.
            template        = step.get("template")
            world_button    = step.get("world_button")
            threshold       = step.get("threshold", 0.75)
            scroll_times    = step.get("scroll_times", 5)
            scroll_interval = step.get("scroll_interval_sec", 0.1)
            timeout_sec     = step.get("timeout_sec", 5)
            debug_save      = step.get("debug_save", False)
            debug_log       = step.get("debug_log", False)
            match_color     = step.get("match_color", False)
            color_tol       = step.get("color_match_tolerance")

            def _do_zoomout_scroll():
                if hasattr(wincap, "focus_window"):
                    wincap.focus_window(force=True)
                    time.sleep(0.05)
                cx = wincap.offset_x + wincap.w // 2
                cy = wincap.offset_y + wincap.h // 2
                pyautogui.moveTo(cx, cy)
                time.sleep(0.05)
                for i in range(scroll_times):
                    pyautogui.scroll(-3)
                    time.sleep(scroll_interval)
                log.info("[Runner] base_zoomout scrolled x{} at center ({}, {})".format(scroll_times, cx, cy))

            vision = self.vision_cache.get(template) if template else None
            vision_world = self.vision_cache.get(world_button) if world_button else None
            dbg_mode = "info" if debug_log else None

            # ROI: neu step co roi_center_x, roi_center_y (va roi_padding) thi chi tim HQ/World trong vung do
            _roi_offset = (0, 0)
            _roi_bounds = None
            _roi_cx = step.get("roi_center_x")
            _roi_cy = step.get("roi_center_y")
            if _roi_cx is not None and _roi_cy is not None and vision:
                _sh, _sw = screenshot.shape[:2]
                _scale = get_global_scale()
                _nw_px = max(1, int(vision.needle_w * _scale))
                _nh_px = max(1, int(vision.needle_h * _scale))
                _padding = float(step.get("roi_padding", 3.0))
                _cx_px = int(_roi_cx * _sw)
                _cy_px = int(_roi_cy * _sh)
                _half_w = int(_nw_px * _padding)
                _half_h = int(_nh_px * _padding)
                _rx = max(0, _cx_px - _half_w)
                _ry = max(0, _cy_px - _half_h)
                _rx2 = min(_sw, _cx_px + _half_w)
                _ry2 = min(_sh, _cy_px + _half_h)
                _roi_offset = (_rx, _ry)
                _roi_bounds = (_rx, _ry, _rx2, _ry2)
                if debug_log:
                    log.info("[Runner] base_zoomout ROI: center=({:.2f},{:.2f}) crop=({},{})->({},{})".format(
                        _roi_cx, _roi_cy, _rx, _ry, _rx2, _ry2))

            def _base_zoomout_search_img(img):
                if _roi_bounds is None:
                    return img
                rx, ry, rx2, ry2 = _roi_bounds
                return img[ry:ry2, rx:rx2]

            def _base_zoomout_shift_points(pts):
                if not pts or _roi_offset == (0, 0):
                    return pts
                ox, oy = _roi_offset
                return [((p[0] + ox, p[1] + oy) + tuple(p[2:]) if len(p) > 2 else (p[0] + ox, p[1] + oy)) for p in pts]

            def _save_roi_debug(img_roi, tpl_name_suffix):
                if not debug_save or img_roi is None or img_roi.size == 0:
                    return
                try:
                    os.makedirs("debug", exist_ok=True)
                    ts_str = time.strftime("%Y%m%d_%H%M%S")
                    path = os.path.join("debug", "base_zoomout_{}_{}_roi_not_found.png".format(tpl_name_suffix, ts_str))
                    cv.imwrite(path, img_roi)
                    log.info("[Runner] base_zoomout debug_save: ROI saved → {}".format(path))
                except Exception as e:
                    log.info("[Runner] base_zoomout debug_save ROI failed: {}".format(e))

            # (1) Thay HQ -> click HQ -> sau do kiem tra World / HQ tren man hinh moi
            clicked_hq = False
            if vision:
                search_img = _base_zoomout_search_img(screenshot)
                points = vision.find(
                    search_img,
                    threshold=threshold,
                    debug_mode=dbg_mode,
                    debug_log=debug_log,
                    is_color=bool(match_color),
                    color_tolerance=color_tol,
                )
                points = _base_zoomout_shift_points(points if points else [])
                if not points:
                    tpl_name = os.path.splitext(os.path.basename(template))[0] if template else "template"
                    _save_roi_debug(search_img, tpl_name)
                if points:
                    if debug_save:
                        try:
                            os.makedirs("debug", exist_ok=True)
                            ts_str = time.strftime("%Y%m%d_%H%M%S")
                            tpl_name = os.path.splitext(os.path.basename(template))[0] if template else "template"
                            shot_path = os.path.join("debug", f"base_zoomout_{tpl_name}_{ts_str}_screenshot.png")
                            cv.imwrite(shot_path, screenshot)
                            cx0, cy0 = int(points[0][0]), int(points[0][1])
                            mw = getattr(vision, "needle_w", 80)
                            mh = getattr(vision, "needle_h", 80)
                            x0 = max(0, cx0 - mw)
                            y0 = max(0, cy0 - mh)
                            x1 = min(screenshot.shape[1], cx0 + mw)
                            y1 = min(screenshot.shape[0], cy0 + mh)
                            crop = screenshot[y0:y1, x0:x1]
                            crop_path = os.path.join("debug", f"base_zoomout_{tpl_name}_{ts_str}_crop.png")
                            if crop.size > 0:
                                cv.imwrite(crop_path, crop)
                            log.info("[Runner] base_zoomout debug saved: {} | {}".format(shot_path, crop_path))
                        except Exception as e:
                            log.info("[Runner] base_zoomout debug save failed: {}".format(e))
                    sx, sy = wincap.get_screen_position((points[0][0], points[0][1]))
                    pyautogui.click(sx, sy)
                    time.sleep(2)
                    clicked_hq = True
                elif debug_save:
                    try:
                        os.makedirs("debug", exist_ok=True)
                        ts_str = time.strftime("%Y%m%d_%H%M%S")
                        tpl_name = os.path.splitext(os.path.basename(template))[0] if template else "template"
                        shot_path = os.path.join("debug", f"base_zoomout_{tpl_name}_{ts_str}_not_found.png")
                        cv.imwrite(shot_path, screenshot)
                        log.info("[Runner] base_zoomout debug saved (not found): {}".format(shot_path))
                    except Exception as e:
                        log.info("[Runner] base_zoomout debug save failed: {}".format(e))

            if clicked_hq:
                # Screenshot2 lay sau khi da sleep 2s o tren (sau click HQ)
                screenshot2 = wincap.get_screenshot() if hasattr(wincap, "get_screenshot") else None
                if screenshot2 is not None and vision_world:
                    search_img2 = _base_zoomout_search_img(screenshot2)
                    points_w2 = vision_world.find(
                        search_img2,
                        threshold=threshold,
                        debug_mode=dbg_mode,
                        debug_log=debug_log,
                        is_color=bool(match_color),
                        color_tolerance=color_tol,
                    )
                    points_w2 = _base_zoomout_shift_points(points_w2 if points_w2 else [])
                    if not points_w2:
                        wb_name = os.path.splitext(os.path.basename(world_button))[0] if world_button else "WorldButton"
                        _save_roi_debug(search_img2, wb_name)
                    if points_w2:
                        _do_zoomout_scroll()
                        log.info("[Runner] {} → true (clicked HQ, saw World → scroll)".format(self._step_label(step)))
                        self._advance_step(True)
                        return "running"
                if screenshot2 is not None and vision:
                    points_hq2 = vision.find(
                        _base_zoomout_search_img(screenshot2),
                        threshold=threshold,
                        debug_mode=dbg_mode,
                        debug_log=debug_log,
                        is_color=bool(match_color),
                        color_tolerance=color_tol,
                    )
                    points_hq2 = _base_zoomout_shift_points(points_hq2 if points_hq2 else [])
                    if points_hq2:
                        sx2, sy2 = wincap.get_screen_position((points_hq2[0][0], points_hq2[0][1]))
                        pyautogui.click(sx2, sy2)
                        time.sleep(2)
                        log.info("[Runner] base_zoomout: saw HQ again → click HQ once more")
                _do_zoomout_scroll()
                log.info("[Runner] {} → true (clicked HQ + scroll)".format(self._step_label(step)))
                self._advance_step(True)
                return "running"

            # (2) Vua vo da thay World: click World -> retry step (luong cu)
            if world_button and vision_world:
                search_img_w = _base_zoomout_search_img(screenshot)
                points_w = vision_world.find(
                    search_img_w,
                    threshold=threshold,
                    debug_mode=dbg_mode,
                    debug_log=debug_log,
                    is_color=bool(match_color),
                    color_tolerance=color_tol,
                )
                points_w = _base_zoomout_shift_points(points_w if points_w else [])
                if not points_w:
                    wb_name = os.path.splitext(os.path.basename(world_button))[0] if world_button else "WorldButton"
                    _save_roi_debug(search_img_w, wb_name)
                if points_w:
                    if hasattr(wincap, "focus_window"):
                        wincap.focus_window(force=True)
                        time.sleep(0.05)
                    sx, sy = wincap.get_screen_position((points_w[0][0], points_w[0][1]))
                    pyautogui.click(sx, sy)
                    time.sleep(2)
                    log.info("[Runner] base_zoomout: clicked WorldButton → retry step 1 (find HQ)")
                    return "running"

            # (3) Khong thay HQ cung khong thay World -> chi scroll zoom out, advance
            _do_zoomout_scroll()
            log.info("[Runner] {} → true (HQ/World not found, scroll)".format(self._step_label(step)))
            self._advance_step(True)
            return "running"

        if step_type == "drag":
            # start_x, start_y = diem bat dau (ti le 0-1). direction_x/y = huong, magnitude nhan voi do dai keo (vd 5 -> keo dai gap 5).
            # drag_distance_ratio = ti le so voi canh nho cua so (mac dinh 0.15). drag_duration_sec = thoi gian moi lan keo.
            dir_x = float(step.get("direction_x", step.get("x", 0)))
            dir_y = float(step.get("direction_y", step.get("y", 0)))
            count = max(1, int(step.get("count", 3)))
            start_x_ratio = float(step.get("start_x", 0.5))
            start_y_ratio = float(step.get("start_y", 0.5))
            if hasattr(wincap, "focus_window"):
                wincap.focus_window(force=True)
                time.sleep(0.05)
            # Diem bat dau (client px) -> screen px
            px_start = int(wincap.w * start_x_ratio)
            py_start = int(wincap.h * start_y_ratio)
            sx, sy = wincap.get_screen_position((px_start, py_start))
            length = (dir_x * dir_x + dir_y * dir_y) ** 0.5
            if length < 1e-6:
                dx, dy = 1.0, 0.0
                length = 1.0
            else:
                dx, dy = dir_x / length, dir_y / length
            # Do dai keo = base * max(1, magnitude(direction)). direction (5,0) -> keo dai gap 5.
            ratio = float(step.get("drag_distance_ratio", 0.15))
            base_px = ratio * min(wincap.w, wincap.h)
            drag_distance_px = base_px * max(1.0, length)
            ex = int(sx + dx * drag_distance_px)
            ey = int(sy + dy * drag_distance_px)
            offset_x = ex - sx
            offset_y = ey - sy
            # Thoi gian cho ca qua trinh keo (mac dinh cham rai ~0.8s)
            duration = step.get("drag_duration_sec", 0.8)
            num_steps = max(15, int(step.get("drag_steps", 30)))

            log.info("[Runner] drag: win={}x{}, start screen=({},{}), end=({},{}), offset=({},{}), drag_px={:.0f}, steps={}, duration={}s".format(
                wincap.w, wincap.h, sx, sy, ex, ey, offset_x, offset_y, drag_distance_px, num_steps, duration))

            # Dung Windows API (SetCursorPos + mouse_event) thay pyautogui — nhieu game/emulator chi nhan input nay.
            _MOUSEEVENTF_LEFTDOWN = 0x0002
            _MOUSEEVENTF_LEFTUP = 0x0004

            def _do_drag(sx_, sy_, off_x, off_y):
                try:
                    import ctypes
                    u32 = ctypes.windll.user32
                    ex_ = sx_ + off_x
                    ey_ = sy_ + off_y
                    step_duration = duration / num_steps
                    # Buoc 1: di chuot toi diem bat dau
                    log.info("[Runner] drag step 1: SetCursorPos start ({}, {})".format(sx_, sy_))
                    u32.SetCursorPos(int(sx_), int(sy_))
                    time.sleep(0.06)
                    # Buoc 2: nhan chuot xuong (mouse_event LEFTDOWN)
                    log.info("[Runner] drag step 2: mouse_event LEFTDOWN at ({}, {})".format(sx_, sy_))
                    u32.mouse_event(_MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                    time.sleep(0.05)
                    # Buoc 3: di chuyen chuot tung buoc (van giu nut)
                    for s in range(1, num_steps + 1):
                        t = s / num_steps
                        px = int(sx_ + off_x * t)
                        py = int(sy_ + off_y * t)
                        u32.SetCursorPos(px, py)
                        time.sleep(step_duration)
                        if s == 1 or s == num_steps or s == num_steps // 2:
                            log.info("[Runner] drag step 3: move s={}/{} -> ({}, {})".format(s, num_steps, px, py))
                    time.sleep(0.04)
                    # Buoc 4: tha chuot (mouse_event LEFTUP)
                    log.info("[Runner] drag step 4: mouse_event LEFTUP at ({}, {})".format(int(ex_), int(ey_)))
                    u32.mouse_event(_MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                except Exception as e:
                    log.warning("[Runner] drag exception: {}".format(e))

            for i in range(count):
                _do_drag(sx, sy, offset_x, offset_y)
                if i < count - 1:
                    time.sleep(0.12)
            log.info("[Runner] {} → true (drag dir=({},{}) x{})".format(
                self._step_label(step), step.get("direction_x", step.get("x")), step.get("direction_y", step.get("y")), count))
            self._advance_step(True)
            return "running"

        if step_type == "ocr_log":
            # OCR text from a region, then log the result. Always advances — never blocks the flow.
            #
            # Mode A — anchor template + pixel offset (single region):
            #   anchor_template: template to find for position
            #   anchor_offset:   [ox, oy, w, h] px from anchor center to OCR region
            #
            # Mode A2 — anchor template + ratio regions (multiple regions):
            #   anchor_template: template to find for position
            #   ocr_regions:     list of {name, x, y, w, h, digits_only, pattern}
            #                    x/y/w/h are ratios of template size (0.0–1.0)
            #
            # Mode B — absolute ROI (no template needed, most robust):
            #   roi_ratios: [x, y, w, h] as fractions of screen size (0.0–1.0)
            #
            # debug_save: true → save ROI crop (and screenshot if anchor not found)
            # exit_on_true: true → when anchor IS found, end function (same semantic as match_click exit_on_true)
            anchor_template = step.get("anchor_template") or step.get("template")
            roi_ratios      = step.get("roi_ratios")       # [x, y, w, h] 0-1 fractions
            ocr_regions     = step.get("ocr_regions")      # list of ratio-based region dicts
            threshold       = step.get("threshold", 0.75)
            anchor_offset   = step.get("anchor_offset")   # [ox, oy, w, h] px from anchor center
            char_whitelist  = step.get("char_whitelist")
            label           = step.get("label", "ocr_log")
            debug_save      = step.get("debug_save", False)
            timeout_sec     = step.get("timeout_sec", 5)
            exit_on_true    = step.get("exit_on_true", False)
            _debug_key      = "_ocr_log_debug_{}".format(label.replace(" ", "_"))

            # ── Mode B: roi_ratios — run immediately, no template needed ──────
            if roi_ratios and len(roi_ratios) == 4:
                h_img, w_img = screenshot.shape[:2]
                rx, ry, rw, rh = roi_ratios
                x = max(0, int(rx * w_img))
                y = max(0, int(ry * h_img))
                w = max(1, int(rw * w_img))
                h = max(1, int(rh * h_img))
                w = min(w, w_img - x)
                h = min(h, h_img - y)
                roi = screenshot[y:y + h, x:x + w]
                debug_path = None
                if debug_save and not getattr(self, _debug_key, False):
                    debug_path = "debug_ocr_{}.png".format(label.replace(" ", "_"))
                    setattr(self, _debug_key, True)
                # anchor_center=(0,0) + offset=(x,y,w,h) → crops exactly [x:x+w, y:y+h]
                text = _read_raw_text_from_roi(
                    screenshot, (0, 0), [x, y, w, h],
                    char_whitelist=char_whitelist, debug_save_path=debug_path)
                if text:
                    log.info("[Runner] {} [{}]: {}".format(self._step_label(step), label, text))
                else:
                    log.info("[Runner] {} [{}]: (no text read)".format(self._step_label(step), label))
                if debug_path:
                    log.info("[Runner] ocr_log: ROI saved to {}".format(debug_path))
                if exit_on_true and text and str(text).strip():
                    self._advance_step(True, step=step)
                else:
                    self._advance_step(True)
                return "running"

            # ── Mode B2: ocr_regions only (no anchor) — x,y,w,h = tỉ lệ màn hình (0.0–1.0) ─
            if ocr_regions and not anchor_template:
                h_img, w_img = screenshot.shape[:2]
                any_text = False
                for region in ocr_regions:
                    rname = region.get("name", "ocr")
                    rx = float(region.get("x", 0.0))
                    ry = float(region.get("y", 0.0))
                    rw = float(region.get("w", 1.0))
                    rh = float(region.get("h", 0.2))
                    x = max(0, int(rx * w_img))
                    y = max(0, int(ry * h_img))
                    w = max(1, int(rw * w_img))
                    h = max(1, int(rh * h_img))
                    w = min(w, w_img - x)
                    h = min(h, h_img - y)
                    debug_path = None
                    if debug_save:
                        debug_path = "debug_ocr_{}_{}.png".format(label.replace(" ", "_"), rname)
                    text = _read_raw_text_from_roi(
                        screenshot, (0, 0), [x, y, w, h],
                        char_whitelist=char_whitelist, debug_save_path=debug_path)
                    if text:
                        s = str(text).strip()
                        any_text = any_text or (bool(s) and any(c.isdigit() for c in s))
                        log.info("[Runner] {} [{}]: {}".format(self._step_label(step), rname, text))
                    else:
                        log.info("[Runner] {} [{}]: (no text read)".format(self._step_label(step), rname))
                if exit_on_true and any_text:
                    self._advance_step(True, step=step)
                else:
                    self._advance_step(True)
                return "running"

            # ── Mode A / A2: anchor template ──────────────────────────────────
            if not anchor_template or (not anchor_offset and not ocr_regions):
                log.info("[Runner] {} → skip (set roi_ratios, ocr_regions only, or anchor_template + anchor_offset/ocr_regions)".format(self._step_label(step)))
                self._advance_step(True)
                return "running"

            vision = self.vision_cache.get(anchor_template)
            if not vision:
                log.info("[Runner] {} → skip (anchor_template not loaded)".format(self._step_label(step)))
                self._advance_step(True)
                return "running"

            points = vision.find(screenshot, threshold=threshold, debug_mode=None)
            if points:
                pt = points[0]
                cx, cy, mw, mh = (pt[0], pt[1], pt[2], pt[3]) if len(pt) >= 4 else (pt[0], pt[1], vision.needle_w, vision.needle_h)
                tpl_name = os.path.splitext(os.path.basename(anchor_template))[0]

                # ── require_new_click: gate OCR on a fresh truck click ────────────
                # If set, only proceed when _last_click_pos differs from the position
                # recorded on the previous OCR run — i.e. a NEW truck was clicked.
                # No extra OCR call needed; the click coordinates are the signal.
                if step.get("require_new_click", False):
                    _cur_pos  = self._last_click_pos
                    _prev_pos = self._last_ocr_click_pos
                    if _cur_pos is not None and _cur_pos == _prev_pos:
                        # Same click as before — wait until a new truck is clicked
                        if now - self.step_start_time < timeout_sec:
                            return "running"
                        # Timeout: no new click detected → treat as assertion failure
                        log.info("[Runner] {} [require_new_click]: timeout, no new click detected".format(
                            self._step_label(step)))
                        on_fail_goto = step.get("on_fail_goto")
                        max_retries  = int(step.get("max_retries", 0))
                        if on_fail_goto is not None and max_retries > 0:
                            cur_step_idx = self.step_index
                            retry_count = self._step_retry_counts.get(cur_step_idx, 0) + 1
                            if retry_count <= max_retries:
                                self._step_retry_counts[cur_step_idx] = retry_count
                                log.info("[Runner] {} [require_new_click]: retry {}/{} (goto step {})".format(
                                    self._step_label(step), retry_count, max_retries, on_fail_goto))
                                self._goto_step(int(on_fail_goto))
                                return "running"
                            log.info("[Runner] {} [require_new_click]: max_retries ({}) reached → abort".format(
                                self._step_label(step), max_retries))
                        self._advance_step(False)
                        return "running"
                    # New click position detected — record it and proceed
                    if _cur_pos != _prev_pos:
                        self._last_ocr_click_pos = _cur_pos
                        if _prev_pos is not None:
                            log.info("[Runner] {} [require_new_click]: new click {} → {}, proceeding".format(
                                self._step_label(step), _prev_pos, _cur_pos))
                # ─────────────────────────────────────────────────────────────────

                if ocr_regions:
                    # Mode A2: multiple ratio-based regions relative to template size
                    for region in ocr_regions:
                        rname   = region.get("name", "ocr")
                        dbg_lbl = "{}_{}".format(tpl_name, rname) if debug_save else None
                        text = read_region_relative(
                            screenshot, cx, cy,
                            mw, mh,
                            x           = region.get("x", 0.0),
                            y           = region.get("y", 0.0),
                            w           = region.get("w", 1.0),
                            h           = region.get("h", 1.0),
                            digits_only = region.get("digits_only", False),
                            pattern     = region.get("pattern"),
                            debug_label = dbg_lbl,
                        )
                        log.info("[Runner] {} [{}]: {}".format(self._step_label(step), rname, text or "(no text)"))
                        if dbg_lbl:
                            log.info("[Runner] ocr_log: debug crops → debug_ocr/{}_*.png".format(dbg_lbl))

                        # ── Retitle last truck crop with player name ───────────
                        if rname == "player_name" and text:
                            _crop = getattr(self, '_last_truck_crop_path', None)
                            if _crop:
                                self._last_truck_crop_path = _retitle_truck_crop(_crop, text)

                        # ── Assertions ────────────────────────────────────────
                        assert_eq  = region.get("assert_equals")   # exact string match
                        assert_in  = region.get("assert_in")        # match any value in list
                        assert_max = region.get("assert_max")       # numeric: fail if value >= max
                        assert_min = region.get("assert_min")       # numeric: fail if value < min
                        # fn_settings overrides for TruckPlunder (and any function with these keys)
                        if rname == "server":
                            _srv_ov = self._fn_setting("servers")
                            if _srv_ov is not None and str(_srv_ov).strip():
                                _srv_str = str(_srv_ov).strip()
                                if _srv_str == "*":
                                    assert_in = None   # wildcard: accept any server
                                else:
                                    assert_in = [s.strip() for s in _srv_str.split(",") if s.strip()]
                        if rname == "power":
                            _mp_ov = self._fn_setting("max_power")
                            if _mp_ov is not None:
                                try:
                                    assert_max = int(_mp_ov)
                                except (ValueError, TypeError):
                                    pass

                        _assert_fail_reason = None

                        if assert_in is not None:
                            allowed = [str(v) for v in (assert_in if isinstance(assert_in, list) else [assert_in])]
                            if text not in allowed:
                                _assert_fail_reason = "assert_in FAIL ({!r} not in {})".format(text, allowed)
                        elif assert_eq is not None and text != str(assert_eq):
                            _assert_fail_reason = "assert_equals FAIL ({!r} != {!r})".format(text, str(assert_eq))

                        if _assert_fail_reason is None and (assert_max is not None or assert_min is not None):
                            try:
                                num = int(text.replace(",", "").replace(".", ""))
                                if assert_max is not None and num >= int(assert_max):
                                    _assert_fail_reason = "assert_max FAIL ({} >= {})".format(num, assert_max)
                                elif assert_min is not None and num < int(assert_min):
                                    _assert_fail_reason = "assert_min FAIL ({} < {})".format(num, assert_min)
                            except (ValueError, AttributeError):
                                _assert_fail_reason = "assert numeric FAIL (cannot parse {!r})".format(text)

                        if _assert_fail_reason:
                            on_fail_goto = step.get("on_fail_goto")
                            max_retries  = step.get("max_retries")  # None = unlimited retries
                            cur_step_idx = self.step_index

                            if on_fail_goto is not None:
                                if max_retries is None:
                                    # No limit — retry indefinitely
                                    log.info("[Runner] {} [{}]: {} → retry (goto step {})".format(
                                        self._step_label(step), rname, _assert_fail_reason, on_fail_goto))
                                    self._goto_step(int(on_fail_goto))
                                    return "running"
                                else:
                                    max_retries = int(max_retries)
                                    retry_count = self._step_retry_counts.get(cur_step_idx, 0) + 1
                                    if retry_count <= max_retries:
                                        self._step_retry_counts[cur_step_idx] = retry_count
                                        log.info("[Runner] {} [{}]: {} → retry {}/{} (goto step {})".format(
                                            self._step_label(step), rname, _assert_fail_reason,
                                            retry_count, max_retries, on_fail_goto))
                                        self._goto_step(int(on_fail_goto))
                                        return "running"
                                    log.info("[Runner] {} [{}]: {} → max_retries ({}) reached → abort".format(
                                        self._step_label(step), rname, _assert_fail_reason, max_retries))
                            else:
                                log.info("[Runner] {} [{}]: {} → abort".format(
                                    self._step_label(step), rname, _assert_fail_reason))

                            self._advance_step(False)
                            return "running"
                else:
                    # Mode A: single pixel-offset region
                    debug_path = None
                    if debug_save and not getattr(self, _debug_key, False):
                        debug_path = "debug_ocr_{}.png".format(label.replace(" ", "_"))
                        setattr(self, _debug_key, True)
                    text = _read_raw_text_from_roi(
                        screenshot, (cx, cy), anchor_offset,
                        char_whitelist=char_whitelist,
                        debug_save_path=debug_path,
                    )
                    if text:
                        log.info("[Runner] {} [{}]: {}".format(self._step_label(step), label, text))
                    else:
                        log.info("[Runner] {} [{}]: (no text read)".format(self._step_label(step), label))
                    if debug_path:
                        log.info("[Runner] ocr_log: ROI saved to {}".format(debug_path))

                if exit_on_true:
                    self._advance_step(True, step=step)
                else:
                    self._advance_step(True)
                return "running"

            if now - self.step_start_time >= timeout_sec:
                if exit_on_true:
                    # anchor NOT found → condition not met → continue with next steps
                    log.info("[Runner] {} → continue (exit_on_true but anchor not found in {}s)".format(self._step_label(step), timeout_sec))
                else:
                    log.info("[Runner] {} → anchor not found in {}s".format(self._step_label(step), timeout_sec))
                if debug_save and not getattr(self, _debug_key + "_shot", False):
                    setattr(self, _debug_key + "_shot", True)
                    try:
                        shot_path = "debug_ocr_{}_screen.png".format(label.replace(" ", "_"))
                        cv.imwrite(shot_path, screenshot)
                        log.info("[Runner] ocr_log: anchor not found — screen saved to {}".format(shot_path))
                    except Exception as e:
                        log.info("[Runner] ocr_log: failed to save debug screen: {}".format(e))

                if not exit_on_true:
                    # Anchor not found = cannot verify conditions → treat as assertion failure
                    on_fail_goto = step.get("on_fail_goto")
                    max_retries  = int(step.get("max_retries", 0))
                    cur_step_idx = self.step_index
                    if on_fail_goto is not None and max_retries > 0:
                        retry_count = self._step_retry_counts.get(cur_step_idx, 0) + 1
                        if retry_count <= max_retries:
                            self._step_retry_counts[cur_step_idx] = retry_count
                            log.info("[Runner] {} → anchor not found → retry {}/{} (goto step {})".format(
                                self._step_label(step), retry_count, max_retries, on_fail_goto))
                            self._goto_step(int(on_fail_goto))
                            return "running"
                        else:
                            log.info("[Runner] {} → anchor not found → max_retries ({}) reached → abort".format(
                                self._step_label(step), max_retries))
                    self._advance_step(False)
                else:
                    self._advance_step(True)
            return "running"

        # unknown type -> skip (true so next step still runs)
        self._advance_step(True)
        return "running"

    def _step_label(self, step):
        """Tra ve chuoi mo ta ngan gon cho step, dung trong log."""
        stype = step.get("event_type", "?")
        tpl = step.get("template") or step.get("click_template") or ""
        tpl_name = os.path.splitext(os.path.basename(tpl))[0] if tpl else ""
        if stype == "sleep":
            return "sleep {}s".format(step.get("duration_sec", 0))
        if stype == "click_position":
            if step.get("position_setting_key"):
                return "click_position ({} from setting)".format(step.get("position_setting_key"))
            x = step.get("x", step.get("offset_x", 0))
            y = step.get("y", step.get("offset_y", 0))
            return "click_position (x={}, y={})".format(x, y)
        if stype == "type_text":
            return "type_text"
        if stype == "key_press":
            return "key_press {}".format(step.get("key", ""))
        if stype == "set_level":
            return "set_level Lv.{}".format(step.get("target_level", "?"))
        if stype == "drag":
            dx = step.get("direction_x", step.get("x", 0))
            dy = step.get("direction_y", step.get("y", 0))
            c = step.get("count", 3)
            start = step.get("start_x"), step.get("start_y")
            if start[0] is not None or start[1] is not None:
                return "drag dir=({},{}) start=({},{}) x{}".format(dx, dy, start[0] or 0.5, start[1] or 0.5, c)
            return "drag dir=({},{}) x{}".format(dx, dy, c)
        if tpl_name:
            return "{} {}".format(stype, tpl_name)
        return stype

    def _advance_step(self, result=True, step=None):
        self.step_index += 1
        self.step_start_time = time.time()
        self.step_click_count = 0
        self.last_step_result = result
        self._step_last_click_t = None
        self._debug_click_saved = False
        self._step_pos_cache = None
        if step is not None:
            if step.get("exit_always"):
                self.step_index = len(self.steps)
                log.info("[Runner] {} → exit_always → end function".format(self._step_label(step)))
            elif step.get("exit_on_true") and result:
                self.step_index = len(self.steps)
                log.info("[Runner] {} → exit_on_true (match) → end function".format(self._step_label(step)))

    def _goto_step(self, index):
        """Jump to a specific step index (used for retry loops)."""
        self.step_index = index
        self.step_start_time = time.time()
        self.step_click_count = 0
        self.last_step_result = True
        self._step_last_click_t = None
        self._debug_click_saved = False
        self._step_pos_cache = None


def load_config(config_path="config.yaml"):
    """Load config.yaml. Tra ve dict co key_bindings, schedules."""
    if not os.path.isfile(config_path):
        return {"key_bindings": {}, "schedules": []}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"key_bindings": {}, "schedules": []}
