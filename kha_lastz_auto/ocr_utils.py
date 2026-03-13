"""
ocr_utils.py
------------
OCR helpers used by bot_engine.py. All OCR is done via EasyOCR (ocr_easyocr.py).
Tesseract is no longer used.

Public API (imported by bot_engine):
    read_level_from_roi(screenshot, roi_ratios, wincap,
                        anchor_center=None, anchor_offset=None,
                        debug_save_path=None, level_range=(1,99))  -> int|None
    read_raw_text_from_roi(screenshot, anchor_center, anchor_offset,
                           char_whitelist=None, debug_save_path=None)  -> str|None
    read_region_relative(screenshot, cx, cy, needle_w, needle_h,
                         x, y, w, h,
                         digits_only=False, pattern=None,
                         debug_label=None)  -> str
"""

import os
import re
import logging

import cv2 as cv

from ocr_easyocr import read_region_easy

_log = logging.getLogger("kha_lastz")


# ── Level parsing helper ───────────────────────────────────────────────────────

def _parse_level(text, level_range):
    """Extract level number from OCR text. Returns int or None."""
    text_s = text.strip()
    if not text_s:
        return None
    
    # Pre-clean: common misreads for '0' in a level context
    # If we see digit + [UoO], it's very likely a '0'.
    text_s = re.sub(r"(\d)[UoO]", r"\1 0", text_s)
    # Also handle standalone [UoO] if we are expecting a number (more risky, so we use regex later)

    m = re.search(r"[Ll][Vv]\.?\s*(\d{1,2})", text_s)
    if not m:
        # Try to match after resolving possible misreads in the digits
        cleaned = text_s.replace('U', '0').replace('O', '0').replace('o', '0')
        m = re.search(r"[Ll][Vv]\.?\s*(\d{1,2})", cleaned)
    
    if not m:
        m = re.search(r"[.,]\s*(\d{1,2})\b", text_s)
    if not m:
        # Pure digits or fixed misreads
        cleaned = text_s.replace('U', '0').replace('O', '0').replace('o', '0')
        m = re.search(r"\b(\d{1,2})\b", cleaned)
        
    if m:
        try:
            num = int(m.group(1))
            if level_range[0] <= num <= level_range[1]:
                return num
        except:
            pass
    return None


# ── High-level read helpers ────────────────────────────────────────────────────

def read_level_from_roi(screenshot, roi_ratios, wincap,
                        anchor_center=None, anchor_offset=None,
                        debug_save_path=None, level_range=(1, 99)):
    """
    Crop screenshot and OCR a level number using EasyOCR.
    Mode A: anchor_center + anchor_offset  (px)
    Mode B: roi_ratios [x, y, w, h] as fractions of screen size
    Returns int level or None.
    """
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

    debug_label = os.path.splitext(os.path.basename(debug_save_path))[0] if debug_save_path else None
    text = read_region_easy(roi, digits_only=False, debug_label=debug_label)
    if not text:
        return None

    level = _parse_level(text, level_range)
    if level is not None:
        _log.info("[OCR] EasyOCR → Lv.{} from {!r}".format(level, text))
    else:
        _log.info("[OCR] EasyOCR → no level match from {!r}".format(text))
    return level


def read_raw_text_from_roi(screenshot, anchor_center, anchor_offset,
                            char_whitelist=None, debug_save_path=None):
    """
    Crop a ROI relative to anchor_center and return OCR'd raw text using EasyOCR.

    anchor_offset: [ox, oy, w, h] in pixels
        ox, oy = offset from anchor_center to the TOP-LEFT of the ROI
        w, h   = size of the ROI in pixels
    char_whitelist: optional string — keeps only these characters from EasyOCR output
                    (e.g. "0123456789:" for timer text).
    Returns stripped text or None.
    """
    h_img, w_img = screenshot.shape[:2]
    cx, cy = anchor_center
    ox, oy, rw, rh = anchor_offset

    x = max(0, min(int(cx + ox), w_img - 1))
    y = max(0, min(int(cy + oy), h_img - 1))
    w = min(max(1, int(rw)), w_img - x)
    h = min(max(1, int(rh)), h_img - y)

    roi = screenshot[y:y + h, x:x + w]
    if roi.size == 0:
        return None

    debug_label = os.path.splitext(os.path.basename(debug_save_path))[0] if debug_save_path else None
    text = read_region_easy(roi, digits_only=False, debug_label=debug_label)
    if text is None:
        return None
    if char_whitelist and text:
        text = "".join(c for c in text if c in char_whitelist)
    return text if text else None


def read_region_relative(screenshot, cx, cy, needle_w, needle_h,
                          x=0.0, y=0.0, w=1.0, h=1.0,
                          digits_only=False, pattern=None,
                          debug_label=None):
    """
    Crop a sub-region relative to a template match center, then OCR with EasyOCR.

    Parameters x, y, w, h are ratios of the template size (0.0–1.0):
        x=0, y=0  → top-left of matched template
        x=1, y=1  → bottom-right

    digits_only: keep only digits, commas, periods from result
    pattern:     regex to extract a specific group, e.g. r"(\\d{3,4})"
    debug_label: if set, saves raw crop to debug_ocr/<label>_raw.png

    Returns extracted string or "".
    """
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
        return ""

    crop = screenshot[ry: ry + rh, rx: rx + rw]
    if crop.size == 0:
        return ""

    result = read_region_easy(crop, digits_only=digits_only,
                               pattern=pattern, debug_label=debug_label)
    return result if result is not None else ""
