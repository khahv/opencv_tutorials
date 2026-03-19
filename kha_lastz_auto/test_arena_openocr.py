"""
test_arena_openocr.py
---------------------
Test OpenOCR (openocr-python) on arena power debug crops.

Run:
    python test_arena_openocr.py

Uses task='rec' (recognition only) with ONNX backend — no GPU needed.
"""

import re
import os
import sys
import tempfile
import cv2 as cv
import numpy as np
from PIL import Image

# ── Ground-truth cases ────────────────────────────────────────────────────────
CASES = [
    # ── Set 2 (092033) ────────────────────────────────────────────────────────
    ("debug_ocr/arena_filter_my_power_092033_703_raw.png",   27.46),
    ("debug_ocr/arena_filter_opponent0_092035_222_raw.png",  29.52),
    ("debug_ocr/arena_filter_opponent1_092036_383_raw.png",  29.68),
    ("debug_ocr/arena_filter_opponent2_092037_252_raw.png",  25.70),
    ("debug_ocr/arena_filter_opponent3_092037_469_raw.png",  22.92),
    ("debug_ocr/arena_filter_opponent4_092038_679_raw.png",  25.62),
    # ── Set 3 (102637) — add expected values if known ─────────────────────────
    # ("debug_ocr/arena_filter_my_power_102637_495_raw.png",  ?.??),
    # ("debug_ocr/arena_filter_opponent0_102646_964_raw.png", ?.??),
    # ("debug_ocr/arena_filter_opponent1_102647_914_raw.png", ?.??),
    # ("debug_ocr/arena_filter_opponent2_102649_249_raw.png", ?.??),
    # ("debug_ocr/arena_filter_opponent3_102649_533_raw.png", ?.??),
    # ("debug_ocr/arena_filter_opponent4_102650_804_raw.png", ?.??),
    # ── Set 4 (103623) — add expected values if known ─────────────────────────
    # ("debug_ocr/arena_filter_my_power_103623_916_raw.png",  ?.??),
    # ("debug_ocr/arena_filter_opponent0_103625_348_raw.png", ?.??),
    # ("debug_ocr/arena_filter_opponent1_103626_623_raw.png", ?.??),
    # ("debug_ocr/arena_filter_opponent2_103627_663_raw.png", ?.??),
    # ("debug_ocr/arena_filter_opponent3_103628_735_raw.png", ?.??),
    # ("debug_ocr/arena_filter_opponent4_103629_082_raw.png", ?.??),
]

TOLERANCE  = 0.06
_VALID_MIN = 1.0
_VALID_MAX = 999.0

# ── Image preprocessing helpers ───────────────────────────────────────────────

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


# ── Power parsing helpers ─────────────────────────────────────────────────────

def normalize_power(num_str: str):
    """Convert raw OCR string to float, re-inserting decimal if missing."""
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
    """Extract the first plausible power value from an OCR string."""
    cleaned = re.sub(r"[^\d.,]", "", raw)
    m = re.search(r"(\d+[.,]\d{1,2}|\d+)", cleaned)
    if not m:
        return None, cleaned
    return normalize_power(m.group(1)), cleaned


# ── Preprocessing configs ─────────────────────────────────────────────────────

PREPS = [
    ("A: raw",           lambda i: i),
    ("B: 2x",            lambda i: scale_up(i, 2)),
    ("C: 3x",            lambda i: scale_up(i, 3)),
    ("D: white180",      lambda i: white_thresh(i, 180)),
    ("E: white180+2x",   lambda i: white_thresh(scale_up(i, 2), 180)),
    ("F: otsu",          lambda i: otsu_thresh(i)),
    ("G: otsu+2x",       lambda i: otsu_thresh(scale_up(i, 2))),
    ("H: otsu+3x",       lambda i: otsu_thresh(scale_up(i, 3))),
    ("I: adap_inv",      lambda i: gray_adaptive_inv(i)),
    ("J: adap_inv+2x",   lambda i: gray_adaptive_inv(scale_up(i, 2))),
    ("K: adap_inv+3x",   lambda i: gray_adaptive_inv(scale_up(i, 3))),
]

# ── OpenOCR interface ─────────────────────────────────────────────────────────

_recognizer = None

def _get_recognizer():
    global _recognizer
    if _recognizer is None:
        from openocr import OpenOCR
        # mode='mobile' uses ONNX — fast, no GPU required.
        # Switch to mode='server', backend='torch' for higher accuracy if torch is installed.
        _recognizer = OpenOCR(task="rec", mode="mobile")
    return _recognizer


def openocr_read(img_bgr: np.ndarray) -> str:
    """Save img_bgr to a temp file and run OpenOCR recognition; return text."""
    rec = _get_recognizer()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv.imwrite(tmp_path, img_bgr)
        results = rec(image_path=tmp_path)
        # results is a list of dicts: [{'text': '...', 'score': 0.99}, ...]
        if not results:
            return ""
        texts = [r.get("text", "") or r.get("rec_text", "") for r in results if r]
        return " ".join(t for t in texts if t).strip()
    except Exception as exc:
        print(f"  [openocr] error: {exc}", file=sys.stderr)
        return ""
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading OpenOCR recognizer (mobile/ONNX)...", flush=True)
    _get_recognizer()
    print("Ready.\n")

    images = []
    for path, _ in CASES:
        img = cv.imread(path)
        if img is None:
            print(f"WARN: cannot load {path}")
        images.append(img)

    col_w = 28

    header = "{:<22}  {:>8}".format("Image", "Expected")
    for label, _ in PREPS:
        header += "  {:<{}}".format(label, col_w)
    print(header)
    print("-" * len(header))

    scores = {label: 0 for label, _ in PREPS}

    for img_orig, (path, expected) in zip(images, CASES):
        img_name = os.path.basename(path).replace("arena_filter_", "").replace("_raw.png", "")
        row = "{:<22}  {:>8.2f}".format(img_name[:22], expected)

        if img_orig is None:
            print(row + "  [MISSING]")
            continue

        for label, prep_fn in PREPS:
            proc = prep_fn(img_orig)
            raw  = openocr_read(proc)
            val, _ = extract_power(raw)
            ok = val is not None and abs(val - expected) <= TOLERANCE
            if ok:
                scores[label] += 1
            short_raw = re.sub(r"[^\d.,M]", "", raw)[:10] or "''"
            cell = "{} {:>10}->{:>7}".format(
                "OK" if ok else "--",
                short_raw,
                "{:.2f}".format(val) if val is not None else "None",
            )
            row += "  {:<{}}".format(cell, col_w)

        print(row)

    print("-" * len(header))
    print("\nScores ({} cases):".format(len(CASES)))
    best_score, best_label = 0, None
    for label, _ in PREPS:
        s = scores[label]
        bar = "#" * s + "." * (len(CASES) - s)
        print("  {:<22} {}  {}/{}".format(label, bar, s, len(CASES)))
        if s > best_score:
            best_score, best_label = s, label
    print("\nBest: [{}]  ({}/{})".format(best_label, best_score, len(CASES)))

    # ── cascade ───────────────────────────────────────────────────────────────
    print("\n--- CASCADE (decimal wins; else fewest digits) ---")
    cascade_correct = 0
    for img_orig, (path, expected) in zip(images, CASES):
        img_name = os.path.basename(path).replace("arena_filter_", "").replace("_raw.png", "")
        if img_orig is None:
            print("  {:22} MISSING".format(img_name))
            continue

        fallbacks    = []
        chosen_val   = None
        chosen_raw   = None
        chosen_label = None

        for label, prep_fn in PREPS:
            proc = prep_fn(img_orig)
            raw  = openocr_read(proc)
            val, _ = extract_power(raw)
            if val is None:
                continue
            if "." in raw or "," in raw:
                chosen_val, chosen_raw, chosen_label = val, raw, label
                break
            digit_count = len(re.sub(r"\D", "", raw))
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
