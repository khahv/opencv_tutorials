"""
Engine chay Function (YAML): load functions, thuc thi tung step.
Step types: match_click, match_multi_click, match_count, sleep, click_position, wait_until_match, key_press, set_level, type_text, click_unless_visible.
Each step returns true/false. Next step is blocked (function aborted) if previous returned false, unless run_always: true is set.
"""
import os
import re
import time
import logging
import random
import yaml
import pyautogui
import cv2 as cv
from vision import Vision

def _save_debug_image(screenshot, raw_center, click_center, needle_w, needle_h, event_type, template_path):
    """Save a debug PNG with green rect (match area) and red circle (click/move target)."""
    dbg = screenshot.copy()
    rx = raw_center[0] - needle_w // 2
    ry = raw_center[1] - needle_h // 2
    cv.rectangle(dbg, (rx, ry), (rx + needle_w, ry + needle_h), (0, 255, 0), 2)
    cv.circle(dbg, (click_center[0], click_center[1]), 12, (0, 0, 255), -1)
    tname = os.path.splitext(os.path.basename(template_path))[0]
    out_path = "debug_{}_{}.png".format(event_type, tname)
    cv.imwrite(out_path, dbg)
    log.info("[Runner] debug_click saved → {}".format(out_path))


try:
    import pytesseract
    _tesseract_configured = False

    def _configure_tesseract():
        global _tesseract_configured
        if _tesseract_configured:
            return
        if os.name == "nt":
            for path in [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            ]:
                if os.path.isfile(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
        _tesseract_configured = True
except ImportError:
    pytesseract = None
    _configure_tesseract = lambda: None

log = logging.getLogger("kha_lastz")
_tesseract_warned = False


def _preprocess_for_ocr(gray, debug_save_path=None):
    """Scale up 4x, threshold, add white padding. Returns list of candidate images to try."""
    # Scale up to make text big enough for OCR (Tesseract struggles with small text)
    gray = cv.resize(gray, None, fx=4, fy=4, interpolation=cv.INTER_CUBIC)

    # Otsu: works well when contrast is clear
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Ensure dark text on white (Tesseract default)
    if thresh.mean() < 128:
        thresh = 255 - thresh

    # Add white border so characters at edges aren't clipped
    padded = cv.copyMakeBorder(thresh, 20, 20, 20, 20, cv.BORDER_CONSTANT, value=255)
    inverted = 255 - padded

    if debug_save_path:
        # Save raw gray (before threshold) for easier visual inspection
        raw_big = cv.resize(gray, None, fx=1, fy=1)
        cv.imwrite(debug_save_path.replace(".png", "_raw.png"), raw_big)
        cv.imwrite(debug_save_path, padded)

    return [padded, inverted]


def _read_level_from_roi(screenshot, roi_ratios, wincap, anchor_center=None, anchor_offset=None, debug_save_path=None, level_range=(1, 99)):
    """Crop screenshot and OCR. Use anchor_center+(offset) if provided, else roi_ratios [x,y,w,h] (0-1)."""
    if pytesseract is None:
        return None
    _configure_tesseract()
    h_img, w_img = screenshot.shape[:2]
    if anchor_center is not None and anchor_offset is not None and len(anchor_offset) == 4:
        cx, cy = anchor_center
        ox, oy, rw, rh = anchor_offset
        x = max(0, int(cx + ox))
        y = max(0, int(cy + oy))
        w = max(1, int(rw))
        h = max(1, int(rh))
    else:
        x = int(roi_ratios[0] * w_img)
        y = int(roi_ratios[1] * h_img)
        w = max(1, int(roi_ratios[2] * w_img))
        h = max(1, int(roi_ratios[3] * h_img))
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    roi = screenshot[y:y + h, x:x + w]
    if roi.size == 0:
        return None
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

    candidates = _preprocess_for_ocr(gray, debug_save_path=debug_save_path)

    # psm 7 = single text line, psm 8 = single word — best for "Lv.1" label
    for psm in (7, 8, 6):
        for img in candidates:
            try:
                text = pytesseract.image_to_string(img, config="--oem 3 --psm {}".format(psm))
            except Exception as e:
                global _tesseract_warned
                if not _tesseract_warned and ("tesseract" in str(e).lower() or "TesseractNotFound" in type(e).__name__):
                    _tesseract_warned = True
                    log.info("[bot_engine] Tesseract OCR not found. Install from https://github.com/UB-Mannheim/tesseract/wiki")
                return None
            text_s = text.strip()
            if text_s:
                log.info("[OCR] psm={} raw={!r}".format(psm, text_s))
            # Strict: "Lv.3", "LV.3", etc.
            m = re.search(r"[Ll][Vv]\.?\s*(\d{1,2})", text_s)
            if not m:
                # Fallback: ".digit" — the dot separator is reliable
                m = re.search(r"[.,]\s*(\d{1,2})\b", text_s)
            if not m:
                # Last resort: any standalone 1-2 digit number
                m = re.search(r"\b(\d{1,2})\b", text_s)
            if m:
                num = int(m.group(1))
                if level_range[0] <= num <= level_range[1]:
                    log.info("[OCR] psm={} -> Lv.{} from {!r}".format(psm, num, text_s))
                    return num
    log.info("[OCR] no level match from any psm/variant")
    return None


def _read_raw_text_from_roi(screenshot, anchor_center, anchor_offset,
                             char_whitelist=None, debug_save_path=None):
    """Crop a ROI relative to anchor_center and return OCR'd raw text.

    anchor_offset: [ox, oy, w, h] in pixels.
        ox, oy = offset from anchor_center to the TOP-LEFT corner of the ROI.
        w, h   = size of the ROI in pixels.

    char_whitelist: optional string passed to Tesseract (e.g. "0123456789:")
    Returns stripped text, or None if OCR is unavailable or ROI is empty.
    """
    if pytesseract is None:
        return None
    _configure_tesseract()

    h_img, w_img = screenshot.shape[:2]
    cx, cy = anchor_center
    ox, oy, rw, rh = anchor_offset

    x = max(0, int(cx + ox))
    y = max(0, int(cy + oy))
    w = max(1, int(rw))
    h = max(1, int(rh))
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = min(w, w_img - x)
    h = min(h, h_img - y)

    roi = screenshot[y:y + h, x:x + w]
    if roi.size == 0:
        return None

    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    candidates = _preprocess_for_ocr(gray, debug_save_path=debug_save_path)

    config = "--oem 3 --psm 7"
    if char_whitelist:
        config += " -c tessedit_char_whitelist={}".format(char_whitelist)

    for img in candidates:
        try:
            text = pytesseract.image_to_string(img, config=config).strip()
        except Exception as e:
            global _tesseract_warned
            if not _tesseract_warned and ("tesseract" in str(e).lower() or "TesseractNotFound" in type(e).__name__):
                _tesseract_warned = True
                log.info("[bot_engine] Tesseract OCR not found. Install from https://github.com/UB-Mannheim/tesseract/wiki")
            return None
        if text:
            return text

    return None


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
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        result[name] = {
            "description": data.get("description", ""),
            "steps": data.get("steps", []),
        }
    return result


def collect_templates(functions_dict):
    """Lay danh sach duong dan template tu tat ca steps de tao vision cache."""
    templates = set()
    for fn in functions_dict.values():
        for step in fn["steps"]:
            if step.get("event_type") in ("match_click", "match_multi_click", "match_count", "match_move") and step.get("template"):
                templates.add(step["template"])
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
            if step.get("event_type") == "base_zoomout" and step.get("template"):
                templates.add(step["template"])
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

    def __init__(self, vision_cache):
        self.vision_cache = vision_cache
        self.functions = {}
        self.state = "idle"  # idle | running
        self.function_name = None
        self.steps = []
        self.step_index = 0
        self.step_start_time = None
        self.step_click_count = 0
        self.wincap = None
        self.last_step_result = True  # True = previous step matched/succeeded

    def load(self, functions_dict):
        self.functions = functions_dict

    def start(self, function_name, wincap):
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
        for attr in ("_set_level_debug_saved", "_set_level_warned"):
            if hasattr(self, attr):
                delattr(self, attr)
        log.info("[Runner] Started function: {}".format(function_name))
        return True

    def stop(self):
        self.state = "idle"
        self.function_name = None

    def update(self, screenshot, wincap):
        """Tra ve 'running' | 'done' | 'idle'. Neu running thi xu ly step hien tai."""
        if self.state != "running" or screenshot is None or self.step_index >= len(self.steps):
            if self.state == "running" and self.step_index >= len(self.steps):
                self.state = "idle"
                log.info("[Runner] Finished function: {}".format(self.function_name))
                return "done"
            return "idle" if self.state == "idle" else "running"

        self.wincap = wincap
        step = self.steps[self.step_index]
        step_type = step.get("event_type", "")

        # Step result gate: abort function if previous step returned False
        # unless this step sets run_always: true
        run_always = step.get("run_always", False)
        if not run_always and not self.last_step_result:
            # Log tat ca cac step con lai (tu vi tri hien tai) la [skip]
            for i in range(self.step_index, len(self.steps)):
                s = self.steps[i]
                if not s.get("run_always", False):
                    log.info("[Runner] [skip] {}".format(self._step_label(s)))
            self.state = "idle"
            log.info("[Runner] Finished function: {} (aborted)".format(self.function_name))
            return "done"

        now = time.time()
        if self.step_start_time is None:
            self.step_start_time = now

        if step_type == "match_click":
            template = step.get("template")
            threshold = step.get("threshold", 0.75)
            one_shot = step.get("one_shot", True)
            timeout_sec = step.get("timeout_sec") or 999
            max_clicks = step.get("max_clicks") or 999999
            click_interval_sec = step.get("click_interval_sec") or 0
            click_random_offset = step.get("click_random_offset") or 0
            click_offset_x = step.get("click_offset_x") or 0.0
            click_offset_y = step.get("click_offset_y") or 0.0

            vision = self.vision_cache.get(template)
            if not vision:
                self._advance_step(True)  # config skip, not a match failure
                return "running"
            debug_click = step.get("debug_click", False)
            points = vision.find(screenshot, threshold=threshold, debug_mode='info' if debug_click else None)
            if points:
                center = list(points[0])
                # click_offset_x/y are ratios of template size: 0.5 = half template width/height
                center[0] += int(click_offset_x * vision.needle_w)
                center[1] += int(click_offset_y * vision.needle_h)
                if click_random_offset > 0:
                    center[0] += random.randint(-click_random_offset, click_random_offset)
                    center[1] += random.randint(-click_random_offset, click_random_offset)
                sx, sy = wincap.get_screen_position(tuple(center))
                raw_center = points[0]
                if debug_click:
                    log.info("[Runner] {} | raw_center=({},{}) needle=({}x{}) offset=({},{}) after_offset=({},{}) screen=({},{})".format(
                        self._step_label(step), raw_center[0], raw_center[1],
                        vision.needle_w, vision.needle_h,
                        click_offset_x, click_offset_y,
                        center[0], center[1], sx, sy))
                    _save_debug_image(screenshot, raw_center, tuple(center),
                                      vision.needle_w, vision.needle_h, "match_click", template)
                pyautogui.click(sx, sy)
                if debug_click:
                    actual = pyautogui.position()
                    log.info("[Runner] {} | intended=({},{}) actual=({},{}) diff=({},{})".format(
                        self._step_label(step), sx, sy, actual.x, actual.y,
                        actual.x - sx, actual.y - sy))
                self.step_click_count += 1
                if not one_shot and click_interval_sec > 0:
                    time.sleep(click_interval_sec)
                if one_shot:
                    log.info("[Runner] {} → true (clicked)".format(self._step_label(step)))
                    self._advance_step(True)
                    return "running"
                if self.step_click_count % 10 == 0:
                    log.info("[Runner] {} clicking... (count {})".format(self._step_label(step), self.step_click_count))
                if self.step_click_count >= max_clicks:
                    log.info("[Runner] {} → true (clicked {})".format(self._step_label(step), self.step_click_count))
                    self._advance_step(True)
                    return "running"
            if now - self.step_start_time >= timeout_sec:
                log.info("[Runner] {} → false (not found in {}s)".format(self._step_label(step), timeout_sec))
                self._advance_step(False)
            return "running"

        if step_type == "match_move":
            template     = step.get("template")
            threshold    = step.get("threshold", 0.75)
            timeout_sec  = step.get("timeout_sec") or 999
            click_offset_x = step.get("click_offset_x") or 0.0
            click_offset_y = step.get("click_offset_y") or 0.0
            debug_click    = step.get("debug_click", False)

            vision = self.vision_cache.get(template)
            if not vision:
                self._advance_step(True)
                return "running"
            points = vision.find(screenshot, threshold=threshold, debug_mode='info' if debug_click else None)
            if points:
                raw_center = points[0]
                center = list(raw_center)
                center[0] += int(click_offset_x * vision.needle_w)
                center[1] += int(click_offset_y * vision.needle_h)
                sx, sy = wincap.get_screen_position(tuple(center))
                if debug_click:
                    log.info("[Runner] {} | raw_center=({},{}) needle=({}x{}) offset=({},{}) after_offset=({},{}) screen=({},{})".format(
                        self._step_label(step), raw_center[0], raw_center[1],
                        vision.needle_w, vision.needle_h,
                        click_offset_x, click_offset_y,
                        center[0], center[1], sx, sy))
                    _save_debug_image(screenshot, raw_center, tuple(center),
                                      vision.needle_w, vision.needle_h, "match_move", template)
                pyautogui.moveTo(sx, sy)
                if debug_click:
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

            vision = self.vision_cache.get(template)
            if not vision:
                self._advance_step(True)
                return "running"
            points = vision.find(screenshot, threshold=threshold, debug_mode=None)
            if points:
                for pt in points:
                    sx, sy = wincap.get_screen_position(tuple(pt))
                    pyautogui.click(sx, sy)
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

        if step_type == "click_position":
            ox = step.get("offset_x", 0.15)
            oy = step.get("offset_y", 0.15)
            px = int(wincap.w * ox)
            py = int(wincap.h * oy)
            sx, sy = wincap.get_screen_position((px, py))
            pyautogui.click(sx, sy)
            log.info("[Runner] {} → true".format(self._step_label(step)))
            self._advance_step(True)
            return "running"

        if step_type == "wait_until_match":
            template = step.get("template")
            threshold = step.get("threshold", 0.75)
            timeout_sec = step.get("timeout_sec") or 30
            vision = self.vision_cache.get(template)
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
            level_roi = step.get("level_roi")
            level_anchor_template = step.get("level_anchor_template")
            level_anchor_offset = step.get("level_anchor_offset")
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
            # Resolve anchor center from template if specified
            anchor_center = None
            if level_anchor_template:
                v_anchor = self.vision_cache.get(level_anchor_template)
                if v_anchor:
                    pts = v_anchor.find(screenshot, threshold=threshold, debug_mode=None)
                    if pts:
                        anchor_center = (int(pts[0][0]), int(pts[0][1]))
            # OCR: read current level
            debug_save = step.get("debug_save_roi") and not getattr(self, "_set_level_debug_saved", False)
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
                    log.info("[Runner] set_level: OCR cannot read level — check level_anchor_offset in YAML")
                return "running"
            self._set_level_warned = False
            if current == target_level:
                log.info("[Runner] set_level: already at Lv.{}, done".format(target_level))
                self._advance_step(True)
                return "running"
            if current < target_level:
                pts = vision_plus.find(screenshot, threshold=threshold, debug_mode=None)
                if pts:
                    sx, sy = wincap.get_screen_position(tuple(pts[0]))
                    pyautogui.click(sx, sy)
                    log.info("[Runner] set_level: Lv.{} -> click Plus (target Lv.{})".format(current, target_level))
                    time.sleep(click_interval)
                else:
                    log.info("[Runner] set_level: Plus greyed at Lv.{} (max reached), proceeding".format(current))
                    self._advance_step(True)
                return "running"
            if current > target_level:
                pts = vision_minus.find(screenshot, threshold=threshold, debug_mode=None)
                if pts:
                    sx, sy = wincap.get_screen_position(tuple(pts[0]))
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
                        sx, sy = wincap.get_screen_position(tuple(pts[0]))
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
            # Resolve ${ENV_VAR} placeholders from environment
            text = re.sub(
                r"\$\{([^}]+)\}",
                lambda m: os.environ.get(m.group(1), ""),
                text,
            )
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
            points = vision.find(screenshot, threshold=threshold, debug_mode=None)
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
                self._advance_step(False)
            return "running"

        if step_type == "base_zoomout":
            # Neu tim thay template (HeadquartersButton) -> click vao -> scroll zoom out.
            # Neu khong tim thay -> chi scroll zoom out (co the da o world map roi).
            template        = step.get("template")
            threshold       = step.get("threshold", 0.75)
            scroll_times    = step.get("scroll_times", 5)
            scroll_interval = step.get("scroll_interval_sec", 0.1)
            timeout_sec     = step.get("timeout_sec", 5)

            vision = self.vision_cache.get(template) if template else None
            clicked_hq = False
            if vision:
                points = vision.find(screenshot, threshold=threshold, debug_mode=None)
                if points:
                    sx, sy = wincap.get_screen_position(tuple(points[0]))
                    pyautogui.click(sx, sy)
                    time.sleep(0.3)
                    clicked_hq = True

            # Scroll zoom out tai tam cua so game
            cx = wincap.offset_x + wincap.w // 2
            cy = wincap.offset_y + wincap.h // 2
            pyautogui.moveTo(cx, cy)
            for _ in range(scroll_times):
                pyautogui.scroll(-3)
                time.sleep(scroll_interval)

            if clicked_hq:
                log.info("[Runner] {} → true (clicked HQ + scrolled x{})".format(self._step_label(step), scroll_times))
            else:
                log.info("[Runner] {} → true (HQ not found, scrolled x{})".format(self._step_label(step), scroll_times))
            self._advance_step(True)
            return "running"

        if step_type == "ocr_log":
            # OCR text from a region, then log the result. Always advances — never blocks the flow.
            #
            # Two modes (mirror of set_level in FightBoomer):
            #
            # Mode A — anchor template (like level_anchor_template):
            #   anchor_template: template to find for position
            #   anchor_offset:   [ox, oy, w, h] px from anchor center to OCR region
            #
            # Mode B — absolute ROI (no template needed, most robust):
            #   roi_ratios: [x, y, w, h] as fractions of screen size (0.0–1.0)
            #
            # debug_save: true → save ROI crop (and screenshot if anchor not found)
            # abort_if_found: true → advance(False)/abort if anchor IS found; advance(True)/continue if NOT found
            #   Use case: check if a condition is already met, skip remaining steps if so
            anchor_template = step.get("anchor_template") or step.get("template")
            roi_ratios      = step.get("roi_ratios")       # [x, y, w, h] 0-1 fractions
            threshold       = step.get("threshold", 0.75)
            anchor_offset   = step.get("anchor_offset")   # [ox, oy, w, h] px from anchor center
            char_whitelist  = step.get("char_whitelist")
            label           = step.get("label", "ocr_log")
            debug_save      = step.get("debug_save", False)
            timeout_sec     = step.get("timeout_sec", 5)
            abort_if_found  = step.get("abort_if_found", False)
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
                gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
                candidates = _preprocess_for_ocr(gray, debug_save_path=debug_path)
                config = "--oem 3 --psm 7"
                if char_whitelist:
                    config += " -c tessedit_char_whitelist={}".format(char_whitelist)
                text = None
                for img in candidates:
                    try:
                        t = pytesseract.image_to_string(img, config=config).strip() if pytesseract else ""
                    except Exception:
                        t = ""
                    if t:
                        text = t
                        break
                if text:
                    log.info("[Runner] {} [{}]: {}".format(self._step_label(step), label, text))
                else:
                    log.info("[Runner] {} [{}]: (no text read)".format(self._step_label(step), label))
                if debug_path:
                    log.info("[Runner] ocr_log: ROI saved to {}".format(debug_path))
                self._advance_step(True)
                return "running"

            # ── Mode A: anchor template ───────────────────────────────────────
            if not anchor_template or not anchor_offset or len(anchor_offset) != 4:
                log.info("[Runner] {} → skip (set roi_ratios or anchor_template+anchor_offset)".format(self._step_label(step)))
                self._advance_step(True)
                return "running"

            vision = self.vision_cache.get(anchor_template)
            if not vision:
                log.info("[Runner] {} → skip (anchor_template not loaded)".format(self._step_label(step)))
                self._advance_step(True)
                return "running"

            points = vision.find(screenshot, threshold=threshold, debug_mode=None)
            if points:
                anchor_center = (int(points[0][0]), int(points[0][1]))
                debug_path = None
                if debug_save and not getattr(self, _debug_key, False):
                    debug_path = "debug_ocr_{}.png".format(label.replace(" ", "_"))
                    setattr(self, _debug_key, True)
                text = _read_raw_text_from_roi(
                    screenshot, anchor_center, anchor_offset,
                    char_whitelist=char_whitelist,
                    debug_save_path=debug_path,
                )
                if text:
                    log.info("[Runner] {} [{}]: {}".format(self._step_label(step), label, text))
                else:
                    log.info("[Runner] {} [{}]: (no text read)".format(self._step_label(step), label))
                if debug_path:
                    log.info("[Runner] ocr_log: ROI saved to {}".format(debug_path))
                if abort_if_found:
                    log.info("[Runner] {} → abort (abort_if_found, anchor matched)".format(self._step_label(step)))
                    self._advance_step(False)
                else:
                    self._advance_step(True)
                return "running"

            if now - self.step_start_time >= timeout_sec:
                if abort_if_found:
                    # anchor NOT found → condition not met → continue with next steps
                    log.info("[Runner] {} → continue (abort_if_found but anchor not found in {}s)".format(self._step_label(step), timeout_sec))
                else:
                    log.info("[Runner] {} → skip (anchor not found in {}s)".format(self._step_label(step), timeout_sec))
                if debug_save and not getattr(self, _debug_key + "_shot", False):
                    setattr(self, _debug_key + "_shot", True)
                    try:
                        shot_path = "debug_ocr_{}_screen.png".format(label.replace(" ", "_"))
                        cv.imwrite(shot_path, screenshot)
                        log.info("[Runner] ocr_log: anchor not found — screen saved to {}".format(shot_path))
                    except Exception as e:
                        log.info("[Runner] ocr_log: failed to save debug screen: {}".format(e))
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
            return "click_position ({}, {})".format(step.get("offset_x", 0), step.get("offset_y", 0))
        if stype == "type_text":
            return "type_text"
        if stype == "key_press":
            return "key_press {}".format(step.get("key", ""))
        if stype == "set_level":
            return "set_level Lv.{}".format(step.get("target_level", "?"))
        if tpl_name:
            return "{} {}".format(stype, tpl_name)
        return stype

    def _advance_step(self, result=True):
        self.step_index += 1
        self.step_start_time = time.time()
        self.step_click_count = 0
        self.last_step_result = result


def load_config(config_path="config.yaml"):
    """Load config.yaml. Tra ve dict co key_bindings, schedules."""
    if not os.path.isfile(config_path):
        return {"key_bindings": {}, "schedules": []}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"key_bindings": {}, "schedules": []}
