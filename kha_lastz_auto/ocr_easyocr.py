"""
ocr_easyocr.py
--------------
EasyOCR-based OCR helper. Replaces Tesseract for reading game UI text —
handles white-stroke outlined fonts (e.g. #549, 1,127,749) much more reliably.

Public API:
    EASYOCR_OK          → bool
    read_region_easy(crop_bgr, digits_only, pattern, debug_label) → str
"""

import os
import re
import logging

import cv2 as cv

_log = logging.getLogger("kha_lastz")

# ── Singleton reader ───────────────────────────────────────────────────────────
_reader = None
EASYOCR_OK = False
_tried = False

def preload():
    """Call at app startup to load the EasyOCR model eagerly (avoids delay on first OCR)."""
    _ensure_reader()

def _ensure_reader():
    global _reader, EASYOCR_OK, _tried
    if _tried:
        return _reader
    _tried = True
    try:
        import easyocr
        _log.info("[EasyOCR] Loading model (first time ~2s)...")
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        EASYOCR_OK = True
        _log.info("[EasyOCR] Ready.")
    except ImportError:
        _log.warning("[EasyOCR] Not installed. Run:  pip install easyocr")
    except Exception as e:
        _log.warning("[EasyOCR] Failed to load: {}".format(e))
    return _reader


# ── Main read function ─────────────────────────────────────────────────────────
def read_region_easy(crop_bgr, digits_only=False, pattern=None, debug_label=None):
    """
    OCR a BGR crop with EasyOCR.

    digits_only: strip all non-digit/comma/period chars from result
    pattern:     regex group to extract, e.g. r"(\\d{3,4})"
    debug_label: if set, saves raw crop to debug_ocr/<label>_raw.png

    Returns extracted string if EasyOCR available, or None if not installed.
    Caller can check for None to fall back to Tesseract.
    """
    reader = _ensure_reader()
    if reader is None:
        return None  # signal: unavailable, let caller fall back

    if crop_bgr is None or crop_bgr.size == 0:
        return ""

    # Scale up — EasyOCR works better with larger text
    h, w = crop_bgr.shape[:2]
    if max(h, w) < 80:
        crop_bgr = cv.resize(crop_bgr, (w * 3, h * 3), interpolation=cv.INTER_CUBIC)
    elif max(h, w) < 150:
        crop_bgr = cv.resize(crop_bgr, (w * 2, h * 2), interpolation=cv.INTER_CUBIC)

    if debug_label:
        os.makedirs("debug_ocr", exist_ok=True)
        cv.imwrite("debug_ocr/{}_raw.png".format(debug_label), crop_bgr)

    # For digits_only: threshold to isolate white/bright text (game UI numbers are
    # white with dark outline on colored backgrounds). This removes icon/background
    # noise that causes EasyOCR to hallucinate extra digits with allowlist active.
    ocr_input = crop_bgr
    if digits_only:
        gray = cv.cvtColor(crop_bgr, cv.COLOR_BGR2GRAY)
        _, mask = cv.threshold(gray, 180, 255, cv.THRESH_BINARY)
        ocr_input = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        if debug_label:
            cv.imwrite("debug_ocr/{}_thresh.png".format(debug_label), ocr_input)

    try:
        if digits_only:
            # paragraph=True merges nearby boxes → prevents "1,127" + "749" split.
            # allowlist="0123456789,." tells EasyOCR to prefer comma over digit noise.
            # Callers that have "#" in their crop should use digits_only=False + pattern.
            results = reader.readtext(ocr_input, detail=0, paragraph=True,
                                      allowlist="0123456789,.")
        else:
            results = reader.readtext(ocr_input, detail=0, paragraph=False)
        raw = " ".join(str(r) for r in results).strip()
    except Exception as e:
        _log.debug("[EasyOCR] readtext error: {}".format(e))
        return ""

    if digits_only:
        raw = re.sub(r"[^\d,.]", "", raw)

    if not raw:
        if debug_label:
            _log.info("[OCR] {} raw=(empty)".format(debug_label))
        return ""

    if debug_label:
        _log.info("[OCR] {} raw={!r}".format(debug_label, raw))

    if pattern:
        m = re.search(pattern, raw)
        if debug_label and not m:
            _log.info("[OCR] {} pattern {!r} → no match".format(debug_label, pattern))
        return m.group(1) if m else ""

    return raw
