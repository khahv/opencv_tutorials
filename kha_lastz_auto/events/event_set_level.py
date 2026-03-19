"""
event_set_level.py
------------------
Handler for the ``set_level`` event type.

Reads the current level from the screen via OCR, then clicks Plus or Minus
until ``target_level`` is reached.

OCR preprocessing
-----------------
The Boomer search dialog (and similar UIs) shows "Lv.X" as small white text
on a coloured slider bar.  OpenOCR reads raw crops of that region poorly —
the text is typically only ~32 px tall.  Two preprocessing steps are applied:

  1. Binary threshold at ``_OCR_THRESH`` — keeps only bright (white) pixels,
     stripping coloured backgrounds and slider graphics.
  2. ``_OCR_UPSCALE`` × upscale (INTER_CUBIC) — enlarges text to a size
     OpenOCR handles reliably.

Tune these two constants at the top of the file if OCR accuracy degrades
on a different UI or resolution.

YAML keys
---------
target_level : int           Level to reach (default 10). Overridable via fn_settings.
level_ocr_region : {x,y,w,h} Screen-ratio coords (0–1) of the "Lv.X" text region.
plus_template  : str         Template for the "+" button.
minus_template : str         Template for the "−" button.
threshold      : float       Template-match threshold (default 0.75).
timeout_sec    : float       Abort after this many seconds (default 30).
click_interval_sec : float   Pause between clicks (default 0.3 s).
min_level / max_level        Accepted OCR level range (default 1–99).
debug_save_roi : bool        Save OCR crop to debug_ocr/ for inspection.

Legacy keys (Mode B / anchor-based)
------------------------------------
level_roi            : [x,y,w,h]  Screen-ratio ROI (used when no anchor template).
level_anchor_template: str         Template whose match centre anchors the ROI.
level_anchor_offset  : [ox,oy,w,h] Pixel offset from anchor centre to ROI top-left.
"""

import time
import logging

import cv2 as cv
from pynput.mouse import Button, Controller

from ocr_utils import _parse_level, read_level_from_roi
from ocr_openocr import read_region_openocr

log = logging.getLogger("kha_lastz")
_mouse_ctrl = Controller()

# ── OCR constants ─────────────────────────────────────────────────────────────
# Binary threshold: pixels brighter than this are kept (white game-UI text).
_OCR_THRESH  = 180
# Upscale factor applied after thresholding to improve OpenOCR accuracy on
# small ROIs (Boomer slider text is typically only ~32 px tall).
_OCR_UPSCALE = 2
# Only these characters are valid in a level string ("Lv.10", "Lv.5", …).
# Anything else (slider decorations read as "_", "—", etc.) is stripped
# before _parse_level so noise cannot prevent a successful match.
_OCR_ALLOWED = set("LlVv.0123456789")


def _preprocess_roi(roi):
    """Apply threshold + upscale so OpenOCR reads small white text reliably."""
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    _, binary = cv.threshold(gray, _OCR_THRESH, 255, cv.THRESH_BINARY)
    bgr = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    h, w = bgr.shape[:2]
    return cv.resize(bgr, (w * _OCR_UPSCALE, h * _OCR_UPSCALE),
                     interpolation=cv.INTER_CUBIC)


def run(step: dict, screenshot, wincap, runner) -> str:
    """Execute one tick of the ``set_level`` event."""
    now = time.time()

    # ── Resolve target level (fn_settings override first) ─────────────────────
    target_level = step.get("target_level", 10)
    _fn_override = runner.fn_settings.get(runner.function_name or "", {})
    if "target_level" in _fn_override:
        try:
            target_level = int(_fn_override["target_level"])
        except (ValueError, TypeError):
            pass

    level_roi             = step.get("level_roi")
    level_anchor_template = step.get("level_anchor_template")
    level_anchor_offset   = step.get("level_anchor_offset")
    level_ocr_region      = step.get("level_ocr_region")
    plus_template         = step.get("plus_template")
    minus_template        = step.get("minus_template")
    match_threshold       = step.get("threshold", 0.75)
    timeout_sec           = step.get("timeout_sec") or 30
    click_interval        = step.get("click_interval_sec", 0.3)
    min_level             = step.get("min_level", 1)
    max_level             = step.get("max_level", 99)
    debug_save            = step.get("debug_save_roi", False)

    if not plus_template or not minus_template:
        runner._advance_step(True)
        return "running"

    if now - runner.step_start_time >= timeout_sec:
        log.info("[set_level] timeout before reaching Lv.{}".format(target_level))
        runner._advance_step(False)
        return "running"

    vision_plus  = runner.vision_cache.get(plus_template)
    vision_minus = runner.vision_cache.get(minus_template)
    if not vision_plus or not vision_minus:
        runner._advance_step(True)
        return "running"

    # ── Resolve anchor center (for Mode B) ────────────────────────────────────
    anchor_center = None
    if level_anchor_template:
        v_anchor = runner._get_vision(level_anchor_template)
        if v_anchor:
            pts = v_anchor.find(screenshot, threshold=match_threshold)
            if pts:
                pt = pts[0]
                anchor_center = (int(pt[0]), int(pt[1]))

    # ── OCR: read current level ────────────────────────────────────────────────
    current = None

    if level_ocr_region:
        # Mode A: absolute screen-ratio coords — threshold + upscale for OpenOCR
        h_img, w_img = screenshot.shape[:2]
        rx  = level_ocr_region.get("x", 0.0)
        ry  = level_ocr_region.get("y", 0.0)
        rw  = level_ocr_region.get("w", 0.1)
        rh  = level_ocr_region.get("h", 0.05)
        px  = max(0, int(rx * w_img))
        py  = max(0, int(ry * h_img))
        pw  = max(1, min(int(rw * w_img), w_img - px))
        ph  = max(1, min(int(rh * h_img), h_img - py))
        roi = screenshot[py:py + ph, px:px + pw]

        dbg_lbl  = "debug_set_level_roi" if debug_save else None
        processed = _preprocess_roi(roi)
        raw_text  = read_region_openocr(processed, digits_only=False, debug_label=dbg_lbl)

        if debug_save:
            log.info("[set_level] ROI crop saved to debug_ocr/debug_set_level_roi_raw.png")
        if raw_text:
            # Strip characters that cannot appear in "Lv.X" (slider noise, dashes, etc.)
            cleaned = "".join(c for c in raw_text if c in _OCR_ALLOWED)
            if cleaned != raw_text:
                log.info("[set_level] cleaned {!r} -> {!r}".format(raw_text, cleaned))
            raw_text = cleaned
            current = _parse_level(raw_text, (min_level, max_level))
            if current is not None:
                log.info("[set_level] Lv.{} from {!r}".format(current, raw_text))
            else:
                log.info("[set_level] no level match from {!r}".format(raw_text))
    else:
        # Mode B (legacy): anchor template + pixel offset or level_roi
        current = read_level_from_roi(
            screenshot, level_roi or [0, 0, 0.3, 0.1], wincap,
            anchor_center, level_anchor_offset,
            debug_save_path="debug_set_level_roi.png" if debug_save else None,
            level_range=(min_level, max_level),
        )
        if debug_save:
            log.info("[set_level] ROI saved to debug_set_level_roi.png")

    if current is None:
        if not getattr(runner, "_set_level_warned", False):
            runner._set_level_warned = True
            log.info("[set_level] OCR cannot read level — check level_ocr_region / level_anchor_offset in YAML")
        return "running"
    runner._set_level_warned = False

    # ── Click Plus / Minus to reach target ────────────────────────────────────
    if current == target_level:
        log.info("[set_level] already at Lv.{}, done".format(target_level))
        runner._advance_step(True)
        return "running"

    if current < target_level:
        pts = vision_plus.find(screenshot, threshold=match_threshold)
        if pts:
            sx, sy = wincap.get_screen_position((pts[0][0], pts[0][1]))
            if not runner._safe_move(sx, sy, wincap, "set_level plus"):
                return "running"
            _mouse_ctrl.press(Button.left)
            time.sleep(0.05)
            _mouse_ctrl.release(Button.left)
            log.info("[set_level] Lv.{} -> click Plus (target Lv.{})".format(current, target_level))
            time.sleep(click_interval)
        else:
            log.info("[set_level] Plus greyed at Lv.{} (max reached), proceeding".format(current))
            runner._advance_step(True)
        return "running"

    # current > target_level
    pts = vision_minus.find(screenshot, threshold=match_threshold)
    if pts:
        sx, sy = wincap.get_screen_position((pts[0][0], pts[0][1]))
        runner._safe_click(sx, sy, wincap, "set_level minus")
        log.info("[set_level] Lv.{} -> click Minus (target Lv.{})".format(current, target_level))
        time.sleep(click_interval)
    else:
        log.info("[set_level] Minus greyed at Lv.{} (min reached), proceeding".format(current))
        runner._advance_step(True)
    return "running"
