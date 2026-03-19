"""
ocr_openocr.py
--------------
OpenOCR-based OCR helper — replaces EasyOCR for all game UI text reading.
Uses openocr-python (ONNX runtime, fast CPU inference, no GPU needed).

Public API
----------
OPENOCR_OK : bool
    True once the recognizer has been loaded successfully.

preload()
    Call at app startup to warm up the model (avoids first-call delay).

read_region_openocr(crop_bgr, digits_only, pattern, debug_label) -> str
    OCR a BGR numpy crop.  Drop-in replacement for read_region_easy().
"""

import os
import re
import logging
import tempfile
import datetime

import cv2 as cv

_log = logging.getLogger("kha_lastz")

# ── Singleton recognizer ───────────────────────────────────────────────────────

_rec       = None
OPENOCR_OK = False
_tried     = False
_loading   = False


def preload():
    """Eagerly load the OpenOCR model (call once at startup in a background thread)."""
    _ensure_rec()


def _ensure_rec():
    global _rec, OPENOCR_OK, _tried, _loading
    if _tried or _loading:
        return _rec
    _loading = True
    try:
        from openocr import OpenOCR
        _log.info("[OpenOCR] Loading model (first time)...")
        _rec = OpenOCR(task="rec", mode="mobile")
        OPENOCR_OK = True
        _log.info("[OpenOCR] Ready.")
    except ImportError:
        _log.warning("[OpenOCR] Not installed. Run:  pip install openocr-python")
    except Exception as exc:
        _log.warning("[OpenOCR] Failed to load: {}".format(exc))
    finally:
        _loading = False
        _tried = True
    return _rec


# ── Main read function ─────────────────────────────────────────────────────────

def read_region_openocr(crop_bgr, digits_only: bool = False,
                        pattern: str = None, debug_label: str = None) -> str:
    """
    OCR a BGR numpy crop with OpenOCR.

    Parameters
    ----------
    crop_bgr    : numpy ndarray (BGR)
    digits_only : strip all non-digit / comma / period chars from result
    pattern     : regex with one capture group to extract from raw text
                  (e.g. r"(\\d{3,4})" )
    debug_label : if set, saves the raw crop to debug_ocr/<label>_<ts>_raw.png

    Returns
    -------
    str — extracted text, or "" on failure / no match.
    Returns None only when the engine is unavailable (caller can fall back).
    """
    rec = _ensure_rec()
    if rec is None:
        return None  # signal: unavailable

    if crop_bgr is None or crop_bgr.size == 0:
        return ""

    # Optional debug save (raw crop, before any processing)
    if debug_label:
        _ts = datetime.datetime.now().strftime("%H%M%S_%f")[:-3]
        os.makedirs("debug_ocr", exist_ok=True)
        cv.imwrite("debug_ocr/{}_{}_raw.png".format(debug_label, _ts), crop_bgr)

    # OpenOCR needs a file path — write to a temp PNG and clean up after
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv.imwrite(tmp_path, crop_bgr)
        results = rec(image_path=tmp_path)
        if not results:
            return ""
        texts = [r.get("text", "") or r.get("rec_text", "") for r in results if r]
        raw = " ".join(t for t in texts if t).strip()
    except Exception as exc:
        _log.debug("[OpenOCR] readtext error: {}".format(exc))
        return ""
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if debug_label:
        _log.info("[OpenOCR] {} raw={!r}".format(debug_label, raw))

    if digits_only:
        raw = re.sub(r"[^\d,.]", "", raw)

    if not raw:
        return ""

    if pattern:
        raw = raw.replace(",.", ",")
        m = re.search(pattern, raw)
        if debug_label and not m:
            _log.info("[OpenOCR] {} pattern {!r} -> no match".format(debug_label, pattern))
        return m.group(1) if m else ""

    return raw
