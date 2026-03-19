"""
test_arena_surya.py
-------------------
Test Surya OCR on arena power debug crops.

Run with the venv312 interpreter:
    .venv312\\Scripts\\python.exe test_arena_surya.py

Edit CASES below to match your debug images.
"""

import re
import sys
import os
import cv2 as cv
import numpy as np
from PIL import Image

# ── Ground-truth cases ────────────────────────────────────────────────────────
CASES = [
    ("debug_ocr/arena_filter_my_power_092033_703_raw.png",   27.46),
    ("debug_ocr/arena_filter_opponent0_092035_222_raw.png",  29.52),
    ("debug_ocr/arena_filter_opponent1_092036_383_raw.png",  29.68),
    ("debug_ocr/arena_filter_opponent2_092037_252_raw.png",  25.70),
    ("debug_ocr/arena_filter_opponent3_092037_469_raw.png",  22.92),
    ("debug_ocr/arena_filter_opponent4_092038_679_raw.png",  25.62),
]

TOLERANCE  = 0.06
_VALID_MIN = 1.0
_VALID_MAX = 999.0

# ── Helpers ───────────────────────────────────────────────────────────────────

def scale_up(img, factor=2):
    h, w = img.shape[:2]
    return cv.resize(img, (w * factor, h * factor), interpolation=cv.INTER_CUBIC)


def white_thresh(img, threshold=180):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)
    return cv.cvtColor(mask, cv.COLOR_GRAY2BGR)


def otsu_thresh(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    if np.mean(mask) > 127:
        mask = cv.bitwise_not(mask)
    return cv.cvtColor(mask, cv.COLOR_GRAY2BGR)


def gray_adaptive_inv(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY_INV, 11, 2)
    return cv.cvtColor(th, cv.COLOR_GRAY2BGR)


def bgr_to_pil(img_bgr):
    """Convert OpenCV BGR image to PIL RGB image."""
    return Image.fromarray(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))


def normalize_power(num_str: str):
    s = num_str.replace(",", ".")
    try:
        val = float(s)
    except ValueError:
        return None
    if "." not in s and val >= 100:
        val = val / 100
    elif "." not in s and val >= 10:
        val = val / 10
    return val if _VALID_MIN <= val <= _VALID_MAX else None


def extract_power(raw: str):
    cleaned = re.sub(r"[^\d.,]", "", raw)
    m = re.search(r"(\d+[.,]\d{1,2}|\d+)", cleaned)
    if not m:
        return None, cleaned
    return normalize_power(m.group(1)), cleaned


# ── Preprocessing configs ─────────────────────────────────────────────────────
def _raw(img):          return img
def _raw2x(img):        return scale_up(img, 2)
def _raw3x(img):        return scale_up(img, 3)
def _w180(img):         return white_thresh(img, 180)
def _w180_2x(img):      return white_thresh(scale_up(img, 2), 180)
def _otsu(img):         return otsu_thresh(img)
def _otsu_2x(img):      return otsu_thresh(scale_up(img, 2))
def _otsu_3x(img):      return otsu_thresh(scale_up(img, 3))
def _adap_inv(img):     return gray_adaptive_inv(img)
def _adap_inv_2x(img):  return gray_adaptive_inv(scale_up(img, 2))
def _adap_inv_3x(img):  return gray_adaptive_inv(scale_up(img, 3))


PREPS = [
    ("A: raw",           _raw),
    ("B: 2x",            _raw2x),
    ("C: 3x",            _raw3x),
    ("D: white180",      _w180),
    ("E: white180+2x",   _w180_2x),
    ("F: otsu",          _otsu),
    ("G: otsu+2x",       _otsu_2x),
    ("H: otsu+3x",       _otsu_3x),
    ("I: adap_inv",      _adap_inv),
    ("J: adap_inv+2x",   _adap_inv_2x),
    ("K: adap_inv+3x",   _adap_inv_3x),
]


def surya_read(rec_predictor, img_bgr, langs=None) -> str:
    """Run Surya recognition on a single BGR image; return joined text.

    Pass a bounding box covering the full image so detection is skipped.
    """
    if langs is None:
        langs = ["en"]
    pil_img = bgr_to_pil(img_bgr)
    w, h = pil_img.size
    # bboxes shape: List[per_image: List[bbox: [x1,y1,x2,y2]]]
    full_bbox = [[0, 0, w, h]]
    predictions = rec_predictor(
        [pil_img],
        task_names=["ocr_with_boxes"],
        bboxes=[full_bbox],
    )
    if not predictions:
        return ""
    pred = predictions[0]
    if hasattr(pred, "text_lines"):
        texts = [line.text for line in pred.text_lines if line.text.strip()]
    elif hasattr(pred, "text"):
        texts = [pred.text] if pred.text.strip() else []
    else:
        texts = [str(pred)]
    return " ".join(texts)


def main():
    print("Loading Surya models...", flush=True)
    try:
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
    except ImportError:
        print("ERROR: surya-ocr not installed in this environment.")
        print("  Run: .venv312\\Scripts\\python.exe -m pip install surya-ocr")
        sys.exit(1)

    print("  Initializing FoundationPredictor (downloads weights on first run)...")
    foundation = FoundationPredictor()
    rec_predictor = RecognitionPredictor(foundation)
    print("Model loaded.\n")

    # ── load images ──────────────────────────────────────────────────────────
    images = []
    for path, _ in CASES:
        img = cv.imread(path)
        if img is None:
            print(f"WARN: cannot load {path}")
        images.append(img)

    col_w = 28

    # ── header ───────────────────────────────────────────────────────────────
    header = "{:<22}  {:>8}".format("Image", "Expected")
    for label, _ in PREPS:
        header += "  {:<{}}".format(label, col_w)
    print()
    print(header)
    print("-" * len(header))

    scores = {label: 0 for label, _ in PREPS}

    # ── per-image rows ────────────────────────────────────────────────────────
    for img_orig, (path, expected) in zip(images, CASES):
        img_name = path.split("/")[-1].replace("arena_filter_", "").replace("_raw.png", "")
        row = "{:<22}  {:>8.2f}".format(img_name[:22], expected)

        if img_orig is None:
            print(row + "  [MISSING]")
            continue

        for label, prep_fn in PREPS:
            proc = prep_fn(img_orig)
            raw  = surya_read(rec_predictor, proc)
            val, _ = extract_power(raw)
            ok = val is not None and abs(val - expected) <= TOLERANCE
            if ok:
                scores[label] += 1
            cell = "{} {:>10}->{:>7}".format(
                "OK" if ok else "--",
                (re.sub(r"[^\d.,M]", "", raw) or "''")[:10],
                "{:.2f}".format(val) if val is not None else "None",
            )
            row += "  {:<{}}".format(cell, col_w)

        print(row)

    # ── summary ──────────────────────────────────────────────────────────────
    print("-" * len(header))
    print("\nScores out of {} cases:".format(len(CASES)))
    best_score = 0
    best_label = None
    for label, _ in PREPS:
        s = scores[label]
        bar = "#" * s + "." * (len(CASES) - s)
        print("  {:<22} {}  {}/{}".format(label, bar, s, len(CASES)))
        if s > best_score:
            best_score = s
            best_label = label
    print("\nBest config: [{}]  ({}/{} correct)".format(best_label, best_score, len(CASES)))

    # ── cascade test ─────────────────────────────────────────────────────────
    print("\n--- CASCADE (first decimal wins, else fewest digits) ---")
    cascade_correct = 0
    for img_orig, (path, expected) in zip(images, CASES):
        img_name = path.split("/")[-1].replace("arena_filter_", "").replace("_raw.png", "")
        if img_orig is None:
            print("  {:22} MISSING".format(img_name))
            continue

        fallbacks     = []
        chosen_val    = None
        chosen_raw    = None
        chosen_label  = None

        for label, prep_fn in PREPS:
            proc = prep_fn(img_orig)
            raw  = surya_read(rec_predictor, proc)
            val, _ = extract_power(raw)
            if val is None:
                continue
            if "." in raw or "," in raw:
                chosen_val, chosen_raw, chosen_label = val, raw, label
                break
            digit_count = len(re.sub(r"\D", "", re.sub(r"[^\d]", "", raw)))
            fallbacks.append((digit_count, val, raw, label))

        if chosen_val is None and fallbacks:
            fallbacks.sort(key=lambda t: t[0])
            _, chosen_val, chosen_raw, chosen_label = fallbacks[0]

        ok = chosen_val is not None and abs(chosen_val - expected) <= TOLERANCE
        if ok:
            cascade_correct += 1
        print("  {:22} expected={:.2f}  got={:>7}  via={:20}  raw={!r:.30}  [{}]".format(
            img_name[:22], expected,
            "{:.2f}".format(chosen_val) if chosen_val is not None else "None",
            chosen_label or "-",
            chosen_raw or "",
            "OK" if ok else "FAIL",
        ))

    print("Cascade: {}/{} correct".format(cascade_correct, len(CASES)))


if __name__ == "__main__":
    main()
