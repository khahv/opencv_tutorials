"""
Engine chay Function (YAML): load functions, thuc thi tung step.
Step types: match_click, match_storm_click, match_multi_click, match_count, sleep, send_zalo, click_position, wait_until_match, key_press, set_level, type_text, click_unless_visible, drag, close_ui, base_zoomout, world_zoomout.
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
from window_click_guard import WindowClickGuard
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
            steps = data.get("steps", [])
            name_to_index = {
                s["name"]: i
                for i, s in enumerate(steps)
                if isinstance(s, dict) and s.get("name")
            }
            result[name] = {
                "description": data.get("description", ""),
                "steps": steps,
                "name_to_index": name_to_index,
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


def _iter_templates(step, single_key, list_key):
    """Yield all template paths from a step that supports both a single key and a list key."""
    tpl_list = step.get(list_key)
    if tpl_list and isinstance(tpl_list, list):
        for t in tpl_list:
            if t:
                yield t
    else:
        tpl = step.get(single_key)
        if tpl:
            yield tpl


def collect_templates(functions_dict):
    """Lay danh sach duong dan template tu tat ca steps de tao vision cache."""
    templates = set()
    for fn in functions_dict.values():
        for step in fn["steps"]:
            if step.get("event_type") in (
                "match_click", "match_multi_click", "match_count", "match_move",
                "match_storm_click",
            ) and step.get("template"):
                templates.add(step["template"])
            if step.get("event_type") == "match_click" and step.get("template_array"):
                ta = step.get("template_array")
                if isinstance(ta, list):
                    for t in ta:
                        path = t.get("template") if isinstance(t, dict) else t
                        if path:
                            templates.add(path)
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
            if step.get("event_type") in ("base_zoomout", "close_ui", "world_zoomout"):
                if step.get("template"):
                    templates.add(step["template"])
                if step.get("world_button"):
                    templates.add(step["world_button"])
                if step.get("back_button"):
                    templates.add(step["back_button"])
            if step.get("event_type") == "ocr_log":
                tpl = step.get("anchor_template") or step.get("template")
                if tpl:
                    templates.add(tpl)
            if step.get("event_type") == "find_truck":
                from events import event_find_truck as _ev_find_truck
                for tpl in _ev_find_truck.collect_templates(step):
                    templates.add(tpl)
            if step.get("event_type") == "arena_filter":
                from events import event_arena_filter as _ev_arena_filter
                for tpl in _ev_arena_filter.collect_templates(step):
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
        self._step_visit_counts = {}       # {step_index: visit_count} for max_tries / on_max_tries_reach_goto
        self._step_visit_start_times = {}  # {step_index: step_start_time} to detect new visits
        self._step_dedup_positions = {}    # {step_index: [(x, y)]} positions already clicked (cross-visit dedup)
        self._tried_positions = []    # [(x, y)] positions already clicked in current cycle
        self._step_name_map = {}      # {name: step_index} built from steps with a "name" field
        self._rtr_cache = {}          # require_text_in_region cache: frozenset(points) → filtered points
        self._rtr_page_logged = False # True after printing truck list for current refresh cycle
        self._fast_clicker = FastClicker()
        self._window_click_guard = WindowClickGuard()
        self._storm_clicker_active = False   # True while match_storm_click is running
        self._storm_start_t: float = 0.0
        self._storm_clicker_kwargs: dict = {}        # last start() args, used to restart after offset change
        self._storm_offset_restart_t: float | None = None  # time when offset_changed was first detected
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

        # Reload fn_settings from .env_config so UI-saved and manually-edited values are always fresh
        try:
            from config_manager import load_fn_settings
            fresh = load_fn_settings()
            self.fn_settings.clear()
            self.fn_settings.update(fresh)
            log.debug("[Runner] fn_settings reloaded: {}".format(self.fn_settings))
        except Exception as e:
            log.warning("[Runner] Failed to reload fn_settings: {}".format(e))

        if function_name not in self.functions:
            log.info("[Runner] Function not found: {}".format(function_name))
            return False
        self.function_name = function_name
        self.steps = self.functions[function_name]["steps"]
        self._step_name_map = self.functions[function_name].get("name_to_index", {})
        self.step_index = 0
        self.step_start_time = time.time()
        self.step_click_count = 0
        self.wincap = wincap
        self.state = "running"
        self.last_step_result = True
        self.trigger_event = trigger_event
        self.trigger_active_cb = trigger_active_cb
        self._step_retry_counts = {}
        self._step_visit_counts = {}
        self._step_visit_start_times = {}
        self._step_dedup_positions = {}
        self._tried_positions = []
        # Reset storm state so a re-run of the same (or a new) function starts clean.
        self._fast_clicker.stop()
        self._window_click_guard.stop()
        self._storm_clicker_active = False
        self._storm_start_t = 0.0
        self._storm_clicker_kwargs = {}
        self._storm_offset_restart_t = None
        self._last_zero_refresh_t = 0
        self._last_truck_crop_path = None
        self._ocr_prev_vals = {}  # {step_index: last_read_value} for wait_for_change_region
        self._last_click_pos = None      # position of the most recent match_click (template-space)
        self._last_ocr_click_pos = None  # position that was current when ocr_log last ran
        self._tpl_array_idx = 0          # current template index for template_array steps
        for attr in ("_set_level_debug_saved", "_set_level_warned"):
            if hasattr(self, attr):
                delattr(self, attr)
        self._world_zoomout_start = None
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
        self._window_click_guard.stop()
        self._storm_clicker_active = False
        try:
            import ctypes
            ctypes.windll.user32.BlockInput(False)
        except Exception:
            pass
        self.state = "idle"
        self.function_name = None

    def update(self, screenshot, wincap):
        """Tra ve 'running' | 'done' | 'idle'. Neu running thi xu ly step hien tai."""
        if self.state != "running" or screenshot is None or self.step_index >= len(self.steps):
            if self.state == "running" and self.step_index >= len(self.steps):
                self._fast_clicker.stop()
                self._window_click_guard.stop()
                self._storm_clicker_active = False
                try:
                    import ctypes
                    ctypes.windll.user32.BlockInput(False)
                except Exception:
                    pass
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
            from events import event_match_click as _ev_match_click
            return _ev_match_click.run(step, screenshot, wincap, self)

        if step_type == "match_move":
            from events import event_match_move as _ev_match_move
            return _ev_match_move.run(step, screenshot, wincap, self)

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
                        if not self._safe_move(sx, sy, wincap, "match_multi_click"):
                            continue
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
            from events import event_sleep as _ev_sleep
            return _ev_sleep.run(step, screenshot, wincap, self)

        if step_type == "send_zalo":
            from events import event_send_zalo as _ev_send_zalo
            return _ev_send_zalo.run(step, screenshot, wincap, self)

        if step_type == "click_position":
            from events import event_click_position as _ev_click_pos
            return _ev_click_pos.run(step, screenshot, wincap, self)

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
            from events import event_set_level as _ev_set_level
            return _ev_set_level.run(step, screenshot, wincap, self)

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
                        self._safe_click(sx, sy, wincap, "click_unless_visible")
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
                        self._goto_step(self._resolve_goto(on_fail_goto))
                        return "running"
                    else:
                        max_retries = int(max_retries)
                        cur_step_idx = self.step_index
                        retry_count = self._step_retry_counts.get(cur_step_idx, 0) + 1
                        if retry_count <= max_retries:
                            self._step_retry_counts[cur_step_idx] = retry_count
                            log.info("[Runner] {} → retry {}/{} (goto step {})".format(
                                self._step_label(step), retry_count, max_retries, on_fail_goto))
                            self._goto_step(self._resolve_goto(on_fail_goto))
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
            # back_button uses its own threshold (default higher) to avoid false positives
            back_button_threshold = float(step.get("back_button_threshold", 0.80))
            debug_log = step.get("debug_log", False)
            debug_save = step.get("debug_save", False)
            match_color = step.get("match_color", True)
            color_tol = step.get("color_match_tolerance", 80)
            click_x = float(step.get("click_x", 0.03))
            click_y = float(step.get("click_y", 0.08))
            max_tries = int(step.get("max_tries", 10))
            back_button  = step.get("back_button")
            vision = self.vision_cache.get(template) if template else None
            vision_world = self.vision_cache.get(world_button) if world_button else None
            vision_back  = self.vision_cache.get(back_button) if back_button else None
            dbg_mode = "info" if debug_log else None
            _roi_offset = (0, 0)
            _roi_bounds = None
            _roi_cx = step.get("roi_center_x", 0.93)
            _roi_cy = step.get("roi_center_y", 0.96)
            if _roi_cx is not None and _roi_cy is not None and vision:
                _sh, _sw = screenshot.shape[:2]
                _scale = get_global_scale()
                _nw_px = max(1, int(vision.needle_w * _scale))
                _nh_px = max(1, int(vision.needle_h * _scale))
                _padding = float(step.get("roi_padding", 2.0))
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
                # Stop immediately if the bot was paused/cancelled mid-loop
                if self.bot_paused and self.bot_paused.get("paused", False):
                    if debug_log:
                        log.info("[Runner] close_ui → aborted (bot paused)")
                    return "running"

                _search = _close_ui_search_img(scr)
                _phq = vision.find(_search, threshold=threshold, debug_mode=dbg_mode, debug_log=debug_log,
                                   is_color=bool(match_color), color_tolerance=color_tol) if vision else []
                _phq = _close_ui_shift_points(_phq if _phq else [])
                _pw = vision_world.find(_search, threshold=threshold, debug_mode=dbg_mode, debug_log=debug_log,
                                        is_color=bool(match_color), color_tolerance=color_tol) if vision_world else []
                _pw = _close_ui_shift_points(_pw if _pw else [])
                if debug_save:
                    try:
                        import datetime
                        _ts = datetime.datetime.now().strftime("%H%M%S_%f")[:-3]
                        _dbg_dir = "debug_close_ui"
                        os.makedirs(_dbg_dir, exist_ok=True)
                        _dbg_img = _search.copy()
                        # Draw ROI bounds info as text
                        _label = "try={} hq={} world={} color={}".format(_try, len(_phq), len(_pw), match_color)
                        cv.putText(_dbg_img, _label, (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        # Draw match rectangles if found
                        for _pt in _phq:
                            _mx, _my = (_pt[0] - _roi_offset[0], _pt[1] - _roi_offset[1])
                            if vision:
                                cv.rectangle(_dbg_img, (_mx - vision.needle_w//2, _my - vision.needle_h//2),
                                             (_mx + vision.needle_w//2, _my + vision.needle_h//2), (0, 255, 0), 2)
                        for _pt in _pw:
                            _mx, _my = (_pt[0] - _roi_offset[0], _pt[1] - _roi_offset[1])
                            if vision_world:
                                cv.rectangle(_dbg_img, (_mx - vision_world.needle_w//2, _my - vision_world.needle_h//2),
                                             (_mx + vision_world.needle_w//2, _my + vision_world.needle_h//2), (0, 0, 255), 2)
                        _fname = os.path.join(_dbg_dir, "close_ui_try{}_{}.png".format(_try, _ts))
                        cv.imwrite(_fname, _dbg_img)
                        log.info("[Runner] close_ui debug_save → {}".format(os.path.abspath(_fname)))
                    except Exception as _de:
                        log.warning("[Runner] close_ui debug_save failed: {}".format(_de))
                if _phq or _pw:
                    if debug_log:
                        log.info("[Runner] close_ui → true (thay HQ/World sau {} lan click)".format(_try))
                    break
                if _try < max_tries - 1:
                    if hasattr(wincap, "focus_window"):
                        wincap.focus_window(force=True)
                        time.sleep(0.05)
                    # Priority: click BackButton if visible; fallback to default (click_x, click_y)
                    _clicked_back = False
                    if vision_back:
                        _pb = vision_back.find(scr, threshold=back_button_threshold,
                                               debug_mode=dbg_mode, debug_log=debug_log)
                        if _pb:
                            _bx, _by = _pb[0][0], _pb[0][1]
                            _bsx, _bsy = wincap.get_screen_position((_bx, _by))
                            self._safe_click(_bsx, _bsy, wincap, "close_ui BackButton")
                            if debug_log:
                                log.info("[Runner] close_ui → clicked BackButton ({},{})".format(_bx, _by))
                            _clicked_back = True
                    if not _clicked_back:
                        _px = int(wincap.w * click_x)
                        _py = int(wincap.h * click_y)
                        _sx, _sy = wincap.get_screen_position((_px, _py))
                        self._safe_click(_sx, _sy, wincap, "close_ui default")
                    time.sleep(1)
                    _fresh = wincap.get_screenshot() if hasattr(wincap, "get_screenshot") else None
                    if _fresh is not None:
                        scr = _fresh
                    time.sleep(0.3)
            log.info("[Runner] close_ui → true")
            self._advance_step(True)
            return "running"

        if step_type == "base_zoomout":
            from events import event_base_zoomout as _ev_base_zo
            return _ev_base_zo.run(step, screenshot, wincap, self)

        if step_type == "world_zoomout":
            from events import event_world_zoomout as _ev_world_zo
            return _ev_world_zo.run(step, screenshot, wincap, self)

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

            # Windows API (SetCursorPos + mouse_event). block_input=True uses BlockInput() to ignore user mouse/keyboard during drag.
            import ctypes
            _MOUSEEVENTF_LEFTDOWN = 0x0002
            _MOUSEEVENTF_LEFTUP = 0x0004
            block_input = step.get("block_input", True)

            def _do_drag(sx_, sy_, off_x, off_y):
                try:
                    import ctypes
                    u32 = ctypes.windll.user32
                    ex_ = sx_ + off_x
                    ey_ = sy_ + off_y
                    step_duration = duration / num_steps
                    log.info("[Runner] drag step 1: SetCursorPos start ({}, {})".format(sx_, sy_))
                    u32.SetCursorPos(int(sx_), int(sy_))
                    time.sleep(0.06)
                    log.info("[Runner] drag step 2: mouse_event LEFTDOWN at ({}, {})".format(sx_, sy_))
                    u32.mouse_event(_MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                    time.sleep(0.05)
                    for s in range(1, num_steps + 1):
                        t = s / num_steps
                        px = int(sx_ + off_x * t)
                        py = int(sy_ + off_y * t)
                        u32.SetCursorPos(px, py)
                        time.sleep(step_duration)
                        if s == 1 or s == num_steps or s == num_steps // 2:
                            log.info("[Runner] drag step 3: move s={}/{} -> ({}, {})".format(s, num_steps, px, py))
                    time.sleep(0.04)
                    log.info("[Runner] drag step 4: mouse_event LEFTUP at ({}, {})".format(int(ex_), int(ey_)))
                    u32.mouse_event(_MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                except Exception as e:
                    log.warning("[Runner] drag exception: {}".format(e))

            try:
                if block_input:
                    try:
                        if ctypes.windll.user32.BlockInput(True):
                            log.info("[Runner] drag: BlockInput(True) — blocking user input during drag")
                        else:
                            log.warning("[Runner] drag: BlockInput(True) returned False (e.g. UAC) — continuing without block")
                    except Exception as e:
                        log.warning("[Runner] drag: BlockInput(True) failed: {} — continuing without block".format(e))
                for i in range(count):
                    _do_drag(sx, sy, offset_x, offset_y)
                    if i < count - 1:
                        time.sleep(0.12)
            finally:
                if block_input:
                    try:
                        ctypes.windll.user32.BlockInput(False)
                        log.info("[Runner] drag: BlockInput(False) — user input restored")
                    except Exception as e:
                        log.warning("[Runner] drag: BlockInput(False) failed: {}".format(e))
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
                                self._goto_step(self._resolve_goto(on_fail_goto))
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
                                    self._goto_step(self._resolve_goto(on_fail_goto))
                                    return "running"
                                else:
                                    max_retries = int(max_retries)
                                    retry_count = self._step_retry_counts.get(cur_step_idx, 0) + 1
                                    if retry_count <= max_retries:
                                        self._step_retry_counts[cur_step_idx] = retry_count
                                        log.info("[Runner] {} [{}]: {} → retry {}/{} (goto step {})".format(
                                            self._step_label(step), rname, _assert_fail_reason,
                                            retry_count, max_retries, on_fail_goto))
                                        self._goto_step(self._resolve_goto(on_fail_goto))
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
                            self._goto_step(self._resolve_goto(on_fail_goto))
                            return "running"
                        else:
                            log.info("[Runner] {} → anchor not found → max_retries ({}) reached → abort".format(
                                self._step_label(step), max_retries))
                    self._advance_step(False)
                else:
                    self._advance_step(True)
            return "running"

        if step_type == "find_truck":
            # Logic lives in events/event_find_truck.py — see that file for full documentation.
            from events import event_find_truck as _ev_find_truck
            return _ev_find_truck.run(step, screenshot, wincap, self)

        # ── arena_filter ─────────────────────────────────────────────────────────
        # Logic lives in events/event_arena_filter.py — see that file for full documentation.
        if step_type == "arena_filter":
            from events import event_arena_filter as _ev_arena_filter
            return _ev_arena_filter.run(step, screenshot, wincap, self)

        # ── match_storm_click ────────────────────────────────────────────────────
        # Dedicated storm-click step: find template → guard outside clicks → FastClick.
        # YAML keys:
        #   template, threshold (default 0.75)
        #   timeout_sec          — wait up to N seconds for template (default 999)
        #   storm_sec            — max storm duration once found (default 60)
        #   max_clicks           — stop after this many clicks (default 0 = unlimited)
        #   offset_x/y           — ±random click offset as fraction of window size (default 0)
        #                          e.g. 0.03 = ±3% of window width/height in pixels
        #   guard_outside        — block user clicks outside game window (default true)
        #   position_refresh_sec — re-detect template every N seconds and reposition (default 0 = off)
        #                          template gone on refresh → storm stops immediately
        #   offset_change_time   — seconds between offset re-randomizations (default 1.0)
        #                          all clicks within the window land on the same pixel
        #   close_ui_check       — every 1s, if template not visible click once at close_ui
        #                          position to dismiss any accidentally-opened UI (default true)
        #   close_ui_click_x/y   — fractional position to click when dismissing UI (default 0.03, 0.08)
        #   close_ui_back_button — optional template path for a BackButton; clicked instead of
        #                          close_ui_click_x/y when visible
        #   corner               — {offset_x, offset_y, every} to keep window focused (optional)
        if step_type == "match_storm_click":
            from events import event_match_storm_click as _ev_msc
            return _ev_msc.run(step, screenshot, wincap, self)

        # unknown type -> skip (true so next step still runs)
        self._advance_step(True)
        return "running"

    def _safe_click(self, sx: int, sy: int, wincap, label: str = "") -> bool:
        """Click at screen coords (sx, sy) only if inside game window bounds.

        Returns True when the click fires, False when skipped (out of bounds).
        All regular click operations go through this to prevent accidental
        clicks outside the game window.
        """
        _l, _t = wincap.get_screen_position((0, 0))
        _r = _l + wincap.w
        _b = _t + wincap.h
        if not (_l <= sx < _r and _t <= sy < _b):
            log.warning(
                "[Runner] safe_click skipped (%d,%d) outside game window (%d,%d)→(%d,%d)%s",
                sx, sy, _l, _t, _r, _b,
                " [{}]".format(label) if label else "",
            )
            return False
        pyautogui.click(sx, sy)
        return True

    def _safe_move(self, sx: int, sy: int, wincap, label: str = "") -> bool:
        """Move mouse to (sx, sy) only if inside game window bounds.

        Returns True when the move is applied, False when skipped.
        Caller is responsible for the subsequent press/release.
        """
        _l, _t = wincap.get_screen_position((0, 0))
        _r = _l + wincap.w
        _b = _t + wincap.h
        if not (_l <= sx < _r and _t <= sy < _b):
            log.warning(
                "[Runner] safe_move skipped (%d,%d) outside game window (%d,%d)→(%d,%d)%s",
                sx, sy, _l, _t, _r, _b,
                " [{}]".format(label) if label else "",
            )
            return False
        _mouse_ctrl.position = (sx, sy)
        return True

    def _step_label(self, step):
        """Return a short description of a step for use in log messages.

        If the step has a ``name`` field, it is prepended as ``[name]`` so named
        steps are immediately identifiable in the log.
        """
        stype = step.get("event_type", "?")
        step_name = step.get("name")
        tpl = step.get("template") or step.get("click_template") or ""
        tpl_name = os.path.splitext(os.path.basename(tpl))[0] if tpl else ""
        if stype == "sleep":
            base = "sleep {}s".format(step.get("duration_sec", 0))
        elif stype == "click_position":
            if step.get("position_setting_key"):
                base = "click_position ({} from setting)".format(step.get("position_setting_key"))
            else:
                x = step.get("x", step.get("offset_x", 0))
                y = step.get("y", step.get("offset_y", 0))
                base = "click_position (x={}, y={})".format(x, y)
        elif stype == "type_text":
            base = "type_text"
        elif stype == "key_press":
            base = "key_press {}".format(step.get("key", ""))
        elif stype == "set_level":
            base = "set_level Lv.{}".format(step.get("target_level", "?"))
        elif stype == "drag":
            dx = step.get("direction_x", step.get("x", 0))
            dy = step.get("direction_y", step.get("y", 0))
            c = step.get("count", 3)
            start = step.get("start_x"), step.get("start_y")
            if start[0] is not None or start[1] is not None:
                base = "drag dir=({},{}) start=({},{}) x{}".format(dx, dy, start[0] or 0.5, start[1] or 0.5, c)
            else:
                base = "drag dir=({},{}) x{}".format(dx, dy, c)
        elif tpl_name:
            base = "{} {}".format(stype, tpl_name)
        else:
            base = stype
        if step_name:
            return "[{}] {}".format(step_name, base)
        return base

    def _advance_step(self, result=True, step=None):
        self.step_index += 1
        self.step_start_time = time.time()
        self.step_click_count = 0
        self.last_step_result = result
        self._step_last_click_t = None
        self._debug_click_saved = False
        self._step_pos_cache = None
        self._tpl_array_idx = 0
        self._tpl_array_start_t = None
        self._tpl_array_last_tpl = None
        if step is not None:
            if step.get("exit_always"):
                self.step_index = len(self.steps)
                log.info("[Runner] {} → exit_always → end function".format(self._step_label(step)))
            elif step.get("exit_on_true") and result:
                self.step_index = len(self.steps)
                log.info("[Runner] {} → exit_on_true (match) → end function".format(self._step_label(step)))

    def _resolve_goto(self, value) -> int:
        """Resolve a goto value to a step index.

        Accepts either an integer index or a string step name defined via the ``name``
        field on any event.  Returns the resolved integer index, or the current
        step_index as a no-op fallback when the name is not found.
        """
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except (ValueError, TypeError):
            idx = self._step_name_map.get(str(value))
            if idx is None:
                log.warning("[Runner] goto name '{}' not found in current function, staying on current step".format(value))
                return self.step_index
            return idx

    def _goto_step(self, index):
        """Jump to a specific step index (used for retry loops)."""
        self.step_index = index
        self.step_start_time = time.time()
        self.step_click_count = 0
        self.last_step_result = True
        self._step_last_click_t = None
        self._debug_click_saved = False
        self._step_pos_cache = None
        # Reset template_array state so the target step starts from template 0.
        self._tpl_array_idx = 0
        self._tpl_array_start_t = None
        self._tpl_array_last_tpl = None

    def _fail_step(self, step, reason: str = "") -> None:
        """Handle a step failure: jump to on_fail_goto (with optional max_retries), or advance normally.

        This centralises the on_fail_goto logic so every event type can use it
        without duplicating the retry-counter bookkeeping.
        """
        on_fail_goto = step.get("on_fail_goto")
        if on_fail_goto is None:
            self._advance_step(False, step=step)
            return

        max_retries = step.get("max_retries")  # None = unlimited
        label = self._step_label(step)
        target = self._resolve_goto(on_fail_goto)

        if max_retries is None:
            log.info("[Runner] {} {} → goto step {} (unlimited retries)".format(
                label, reason, target))
            self._goto_step(target)
            return

        max_retries = int(max_retries)
        cur_idx = self.step_index
        retry_count = self._step_retry_counts.get(cur_idx, 0) + 1
        if retry_count <= max_retries:
            self._step_retry_counts[cur_idx] = retry_count
            log.info("[Runner] {} {} → goto step {} (retry {}/{})".format(
                label, reason, target, retry_count, max_retries))
            self._goto_step(target)
        else:
            log.info("[Runner] {} {} → max_retries ({}) reached → advance".format(
                label, reason, max_retries))
            self._advance_step(False, step=step)


def load_config(config_path="config.yaml"):
    """Load config.yaml. Tra ve dict co key_bindings, schedules."""
    if not os.path.isfile(config_path):
        return {"key_bindings": {}, "schedules": []}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"key_bindings": {}, "schedules": []}
