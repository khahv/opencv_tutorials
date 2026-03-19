"""
event_arena_filter.py
---------------------
Handler for the ``arena_filter`` event type.

Flow
----
1. (Optional) Sleep ``sleep_before_ocr`` seconds so the challenge screen fully renders.
2. Take a fresh screenshot, then OCR *my_power* from an absolute screen-ratio region.
3. OCR each opponent's power from their respective ratio regions.
4. Collect opponents whose power is strictly less than my_power.
5. Click the attack button of the weakest eligible opponent.
6. Sleep ``sleep_before_combat`` seconds.
7. Find and click the combat confirm button via template match.

YAML keys
---------
my_power_region : {x, y, w, h}
    Absolute screen ratios for the self-power label.
opponents : list of {x, y, w, h, attack_x, attack_y}
    x/y/w/h        → ratio region for OCR
    attack_x/y     → ratio click position for the Attack button
sleep_before_ocr : float
    Seconds to wait after the step starts before taking the OCR screenshot.
    Allows the challenge screen to finish rendering. Default: 1.5
combat_template : str
    Path to the combat confirm button template image.
combat_threshold : float
    Template match threshold (default 0.8).
sleep_before_combat : float
    Seconds to wait after clicking Attack before clicking Combat (default 2.0).
debug_save : bool
    When true, each OCR crop is saved to ``debug_ocr/arena_filter_<label>_<ts>_raw.png``.

Power OCR
---------
Uses OpenOCR (openocr-python, ONNX mobile mode) which reads game power values like
"27.46M", "109.55M", "9.5M" accurately without any image preprocessing.
The "M" suffix and any surrounding characters are stripped before parsing.
If the decimal point is missing (rare fallback), it is re-inserted two places from
the right (e.g. "2746" → 27.46).
"""

import os
import re
import time
import logging
import datetime
import tempfile

import cv2 as cv

log = logging.getLogger("kha_lastz")

_DEBUG_DIR = "debug_ocr"
_POWER_VALID_MIN = 1.0
_POWER_VALID_MAX = 999.0

# ── OpenOCR singleton ──────────────────────────────────────────────────────────

_openocr_rec = None


def _ensure_openocr():
    """Lazy-initialize the OpenOCR recognizer (ONNX mobile, downloads once)."""
    global _openocr_rec
    if _openocr_rec is None:
        from openocr import OpenOCR
        _openocr_rec = OpenOCR(task="rec", mode="mobile")
    return _openocr_rec


# ── Internal helpers ───────────────────────────────────────────────────────────

def _crop_ratio_region(screenshot, cfg: dict):
    """Crop *screenshot* at the absolute ratio region described by *cfg*.

    Keys: x, y, w, h — all as fractions of the screenshot dimensions.
    Returns the cropped numpy array, or ``None`` if the region is empty.
    """
    img_h, img_w = screenshot.shape[:2]
    rx = max(0, int(float(cfg.get("x", 0)) * img_w))
    ry = max(0, int(float(cfg.get("y", 0)) * img_h))
    rw = max(1, min(int(float(cfg.get("w", 0.1)) * img_w), img_w - rx))
    rh = max(1, min(int(float(cfg.get("h", 0.02)) * img_h), img_h - ry))
    crop = screenshot[ry:ry + rh, rx:rx + rw]
    return crop if crop.size > 0 else None


def _save_debug_crop(crop, label: str) -> None:
    """Save *crop* to ``debug_ocr/arena_filter_<label>_<timestamp>_raw.png``."""
    ts = datetime.datetime.now().strftime("%H%M%S_%f")[:-3]
    os.makedirs(_DEBUG_DIR, exist_ok=True)
    path = os.path.join(_DEBUG_DIR, "arena_filter_{}_{}_raw.png".format(label, ts))
    cv.imwrite(path, crop)
    log.info("[arena_filter] debug saved -> {}".format(path))


def _normalize_power(num_str: str):
    """Convert an OCR number string to a float power value in M units.

    Handles two cases:
    - Has decimal point (e.g. "27.46", "9.5")  → use as-is.
    - No decimal point  (e.g. "2746", "18569") → decimal was dropped by OCR;
      insert it two places from the right (e.g. "2746" → 27.46).

    Returns float or None on parse failure.
    """
    num_str = num_str.replace(",", ".")
    try:
        val = float(num_str)
    except (ValueError, TypeError):
        return None
    if "." not in num_str and val >= 100:
        val = val / 100
    return val


def _ocr_power(crop) -> tuple:
    """Run OpenOCR on *crop* and return (float_value, raw_text).

    Saves crop to a temp PNG, runs the ONNX recognizer, strips non-numeric
    characters, and parses the first plausible power value.

    Returns (None, "") on failure.
    """
    rec = _ensure_openocr()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv.imwrite(tmp_path, crop)
        results = rec(image_path=tmp_path)
        if not results:
            return None, ""
        texts = [r.get("text", "") or r.get("rec_text", "") for r in results if r]
        raw = " ".join(t for t in texts if t).strip()
        cleaned = re.sub(r"[^\d.,]", "", raw)
        m = re.search(r"(\d+[.,]\d{1,2}|\d+)", cleaned)
        if not m:
            return None, raw
        val = _normalize_power(m.group(1))
        if val is None or not (_POWER_VALID_MIN <= val <= _POWER_VALID_MAX):
            return None, raw
        return val, raw
    except Exception as exc:
        log.debug("[arena_filter] OpenOCR error: {}".format(exc))
        return None, ""
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _read_power(screenshot, cfg: dict, debug_label=None):
    """OCR a power value in M units from a ratio-region of *screenshot*.

    Parameters
    ----------
    debug_label : str or None
        When set, saves the raw crop image to debug_ocr/ for inspection.

    Returns
    -------
    (float_value, raw_ocr_text)
        ``float_value`` is ``None`` when OCR fails or text cannot be parsed.
    """
    crop = _crop_ratio_region(screenshot, cfg)
    if crop is None:
        if debug_label:
            log.info("[arena_filter] [{}] crop is empty — check x/y/w/h ratios".format(debug_label))
        return None, ""

    if debug_label:
        _save_debug_crop(crop, debug_label)

    return _ocr_power(crop)


# ── Public entry point ─────────────────────────────────────────────────────────

def run(step: dict, screenshot, wincap, runner) -> str:
    """Execute one tick of the ``arena_filter`` event.

    Parameters
    ----------
    step : dict
        Parsed YAML step (the ``arena_filter`` block).
    screenshot : numpy.ndarray
        Current game screenshot (BGR) — may be stale; a fresh one is taken
        after ``sleep_before_ocr`` to guarantee the challenge screen is ready.
    wincap : WindowCapture
        Provides ``.w``, ``.h``, ``.get_screen_position()``, ``.get_screenshot()``.
    runner : FunctionRunner
        Provides ``vision_cache``, ``_safe_click()``, ``_advance_step()``.

    Returns
    -------
    str
        Always ``"running"`` — consistent with bot_engine step conventions.
    """
    my_power_cfg        = step.get("my_power_region") or {}
    opponents_cfg       = step.get("opponents") or []
    sleep_before_ocr    = float(step.get("sleep_before_ocr", 1.5))
    combat_template     = step.get("combat_template")
    combat_threshold    = float(step.get("combat_threshold", 0.8))
    sleep_before_combat = float(step.get("sleep_before_combat", 2.0))
    debug_save          = bool(step.get("debug_save", False))

    # Wait for the challenge screen to finish rendering before OCR
    if sleep_before_ocr > 0:
        time.sleep(sleep_before_ocr)

    # Always take a fresh screenshot — the one passed in may pre-date the click
    ocr_screenshot = wincap.get_screenshot()

    # Step 1 – read my own power
    my_label = "my_power" if debug_save else None
    my_val, my_text = _read_power(ocr_screenshot, my_power_cfg, my_label)
    log.info("[arena_filter] my_power OCR={!r} -> {}".format(my_text, my_val))

    if my_val is None:
        log.info("[arena_filter] could not parse my_power -> abort")
        runner._advance_step(False)
        return "running"

    # Step 2 – read each opponent's power; keep those strictly weaker than self
    candidates = []
    for idx, opp in enumerate(opponents_cfg):
        opp_label = "opponent{}".format(idx) if debug_save else None
        opp_val, opp_text = _read_power(ocr_screenshot, opp, opp_label)
        log.info("[arena_filter] opponent[{}] OCR={!r} -> {}".format(idx, opp_text, opp_val))
        if opp_val is not None and opp_val < my_val:
            candidates.append((opp_val, idx, opp))

    if not candidates:
        log.info("[arena_filter] no opponent weaker than {} -> abort".format(my_val))
        runner._advance_step(False)
        return "running"

    # Step 3 – click attack button of the weakest eligible opponent
    candidates.sort(key=lambda t: t[0])
    best_val, best_idx, best_opp = candidates[0]
    log.info("[arena_filter] attacking opponent[{}] power={} (mine={})".format(
        best_idx, best_val, my_val))

    atk_px = int(float(best_opp.get("attack_x", 0.75)) * wincap.w)
    atk_py = int(float(best_opp.get("attack_y", 0.5)) * wincap.h)
    atk_sx, atk_sy = wincap.get_screen_position((atk_px, atk_py))
    runner._safe_click(atk_sx, atk_sy, wincap, "arena_filter attack")
    log.info("[arena_filter] clicked attack at screen ({},{})".format(atk_sx, atk_sy))

    # Step 4 – wait, then click the combat confirm button
    time.sleep(sleep_before_combat)

    if combat_template:
        v_combat = runner.vision_cache.get(combat_template)
        if v_combat:
            scr_combat = wincap.get_screenshot()
            combat_pts = v_combat.find(scr_combat, threshold=combat_threshold)
            if combat_pts:
                ccx, ccy = combat_pts[0][0], combat_pts[0][1]
                csx, csy = wincap.get_screen_position((ccx, ccy))
                runner._safe_click(csx, csy, wincap, "arena_filter combat")
                log.info("[arena_filter] clicked combat button at ({},{})".format(ccx, ccy))
            else:
                log.info("[arena_filter] combat template not found on screen")
        else:
            log.info("[arena_filter] combat template not loaded: {}".format(combat_template))

    runner._advance_step(True)
    return "running"


def collect_templates(step: dict) -> list:
    """Return template paths used by this step so they can be pre-loaded.

    Called by ``bot_engine.collect_templates`` for every ``arena_filter`` step.
    """
    templates = []
    tpl = step.get("combat_template")
    if tpl:
        templates.append(tpl)
    return templates
