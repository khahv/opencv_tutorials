"""
test_arena_rapidocr.py
----------------------
Test RapidOCR (ONNX-based PaddleOCR) on arena power debug crops.

Run from the kha_lastz_auto directory:
    python test_arena_rapidocr.py

Edit CASES below to match whichever debug images you have on disk.
"""

import re
import sys
import cv2 as cv
import numpy as np

# ── Ground-truth cases ────────────────────────────────────────────────────────
CASES = [
    ("debug_ocr/arena_filter_my_power_092033_703_raw.png",    27.46),
    ("debug_ocr/arena_filter_opponent0_092035_222_raw.png",   29.52),
    ("debug_ocr/arena_filter_opponent1_092036_383_raw.png",   29.68),
    ("debug_ocr/arena_filter_opponent2_092037_252_raw.png",   25.70),
    ("debug_ocr/arena_filter_opponent3_092037_469_raw.png",   22.92),
    ("debug_ocr/arena_filter_opponent4_092038_679_raw.png",   25.62),
]

TOLERANCE  = 0.06
_VALID_MIN = 1.0
_VALID_MAX = 999.0

SCALE      = 2          # upscale factor before OCR
SCALE_3X   = 3

# ── Charset filter ─────────────────────────────────────────────────────────────
_ALLOWED = set("0123456789.M")

def charset_filter(text: str) -> str:
    """Keep only allowed characters; collapse runs of '.' to single '.'."""
    filtered = "".join(c for c in text if c in _ALLOWED)
    # remove trailing M and any junk after decimal digits
    return filtered


# ── Preprocessing helpers ──────────────────────────────────────────────────────

def scale_up(img, factor=SCALE):
    h, w = img.shape[:2]
    return cv.resize(img, (w * factor, h * factor), interpolation=cv.INTER_CUBIC)


def _to_gray_2x(img):
    """Grayscale + 2× upscale (base for most user-requested preps)."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    return gray


def _to_gray_3x(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return cv.resize(gray, None, fx=3, fy=3, interpolation=cv.INTER_CUBIC)


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


def invert_white_thresh(img, threshold=180):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY_INV)
    return cv.cvtColor(mask, cv.COLOR_GRAY2BGR)


# ── User-requested preprocessing pipelines (return BGR for RapidOCR) ──────────

def prep_gray2x_th150(img):
    """Grayscale → 2x → threshold(150) — user's recommended pipeline."""
    g = _to_gray_2x(img)
    _, th = cv.threshold(g, 150, 255, cv.THRESH_BINARY)
    return cv.cvtColor(th, cv.COLOR_GRAY2BGR)


def prep_gray2x_th100(img):
    g = _to_gray_2x(img)
    _, th = cv.threshold(g, 100, 255, cv.THRESH_BINARY)
    return cv.cvtColor(th, cv.COLOR_GRAY2BGR)


def prep_gray2x_adaptive(img):
    """Grayscale → 2x → adaptive threshold (handles uneven backgrounds)."""
    g = _to_gray_2x(img)
    th = cv.adaptiveThreshold(g, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY, 11, 2)
    return cv.cvtColor(th, cv.COLOR_GRAY2BGR)


def prep_gray2x_adaptive_inv(img):
    """Adaptive inverted — for light text on dark background."""
    g = _to_gray_2x(img)
    th = cv.adaptiveThreshold(g, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY_INV, 11, 2)
    return cv.cvtColor(th, cv.COLOR_GRAY2BGR)


def prep_gray3x_th150(img):
    g = _to_gray_3x(img)
    _, th = cv.threshold(g, 150, 255, cv.THRESH_BINARY)
    return cv.cvtColor(th, cv.COLOR_GRAY2BGR)


def prep_gray3x_adaptive(img):
    g = _to_gray_3x(img)
    th = cv.adaptiveThreshold(g, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY, 11, 2)
    return cv.cvtColor(th, cv.COLOR_GRAY2BGR)


def prep_raw_2x(img):
    h, w = img.shape[:2]
    return cv.resize(img, (w * 2, h * 2), interpolation=cv.INTER_CUBIC)


def prep_raw_3x(img):
    h, w = img.shape[:2]
    return cv.resize(img, (w * 3, h * 3), interpolation=cv.INTER_CUBIC)


def normalize_power(num_str: str):
    """Convert raw OCR digit string to float M value."""
    s = num_str.replace(",", ".")
    try:
        val = float(s)
    except ValueError:
        return None
    if "." not in s:
        if val >= 100:
            val = val / 100
        elif val >= 10:
            val = val / 10
    return val if _VALID_MIN <= val <= _VALID_MAX else None


def extract_power(raw: str):
    """Pull first plausible number from OCR text; return (float|None, raw)."""
    cleaned = re.sub(r"[^\d.,]", "", raw)
    m = re.search(r"(\d+[.,]\d{1,2}|\d+)", cleaned)
    if not m:
        return None, raw
    val = normalize_power(m.group(1))
    return val, raw


def rapid_ocr_text(ocr, img) -> str:
    """Run RapidOCR on img; return concatenated text (all boxes joined)."""
    try:
        result, elapse = ocr(img)
        if not result:
            return ""
        return " ".join(str(r[1]) for r in result).strip()
    except Exception as exc:
        return ""


def rapid_ocr_best(ocr, img):
    """Run RapidOCR + charset filter, return best (val, raw).

    Strategy:
      1. Apply charset filter (keep only 0-9 . M) on every detected box.
      2. Try each filtered box individually.
      3. Try concatenating all filtered boxes without spaces.
      4. Prefer results with a decimal point; tie-break by fewest digits.
    Returns (float|None, str).
    """
    try:
        result, elapse = ocr(img)
    except Exception:
        return None, ""

    if not result:
        return None, ""

    # Apply charset filter to each raw box text
    texts = [charset_filter(str(r[1])) for r in result]
    texts = [t for t in texts if t]   # drop empty after filtering

    if not texts:
        return None, ""

    candidates_with_decimal = []
    candidates_no_decimal   = []

    def _add_candidate(text):
        val, _ = extract_power(text)
        if val is None:
            return
        if "." in text:
            candidates_with_decimal.append((val, text))
        else:
            digit_count = len(re.sub(r"\D", "", text))
            candidates_no_decimal.append((digit_count, val, text))

    # Check each individual filtered box
    for t in texts:
        _add_candidate(t)

    # Try joining all boxes without any separator (handles split-box problem)
    joined = "".join(texts)
    _add_candidate(joined)

    if candidates_with_decimal:
        return candidates_with_decimal[0]

    if candidates_no_decimal:
        candidates_no_decimal.sort(key=lambda t: t[0])
        _, val, raw = candidates_no_decimal[0]
        return val, raw

    return None, " | ".join(texts)


# ── Build preprocessing configs ───────────────────────────────────────────────
# Each entry: (label, preprocess_fn)

PREPS = [
    # label                  prep_fn               det_mode
    ("A: 2x+th150",          prep_gray2x_th150,    "det"),
    ("B: 2x+th100",          prep_gray2x_th100,    "det"),
    ("C: 2x+adaptive",       prep_gray2x_adaptive, "det"),
    ("D: 2x+adap_inv",       prep_gray2x_adaptive_inv, "det"),
    ("E: 3x+th150",          prep_gray3x_th150,    "det"),
    ("F: 3x+adaptive",       prep_gray3x_adaptive, "det"),
    ("G: raw2x",             prep_raw_2x,          "det"),
    ("H: raw3x",             prep_raw_3x,          "det"),
    # ── no-detection mode ────────────────────────────────────────────────────
    ("I: nd+2x+th150",       prep_gray2x_th150,    "nodet"),
    ("J: nd+2x+th100",       prep_gray2x_th100,    "nodet"),
    ("K: nd+2x+adaptive",    prep_gray2x_adaptive, "nodet"),
    ("L: nd+2x+adap_inv",    prep_gray2x_adaptive_inv, "nodet"),
    ("M: nd+3x+th150",       prep_gray3x_th150,    "nodet"),
    ("N: nd+3x+adaptive",    prep_gray3x_adaptive, "nodet"),
    ("O: nd+raw2x",          prep_raw_2x,          "nodet"),
    ("P: nd+raw3x",          prep_raw_3x,          "nodet"),
]


def main():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        print("ERROR: rapidocr-onnxruntime not installed.")
        print("  pip install --user rapidocr-onnxruntime")
        sys.exit(1)

    ocr      = RapidOCR()                    # default: detection + recognition
    ocr_nodet = RapidOCR(det_model_path=None) # recognition only (no text detection)

    # ── load images ──────────────────────────────────────────────────────────
    images = []
    for path, expected in CASES:
        img = cv.imread(path)
        if img is None:
            print("WARN: cannot load", path)
            images.append(None)
        else:
            images.append(scale_up(img, SCALE))

    labels = [c[0].split("/")[-1].replace("arena_filter_", "").split("_raw")[0]
              for c, _ in [(p, e) for p, e in CASES]]
    expected_vals = [e for _, e in CASES]

    col_w = 30
    prep_w = 16

    # ── header ───────────────────────────────────────────────────────────────
    header = "{:<22}  {:>8}".format("Image", "Expected")
    for label, _, __ in PREPS:
        header += "  {:<{}}".format(label, col_w)
    print()
    print(header)
    print("-" * len(header))

    scores = {label: 0 for label, _, __ in PREPS}

    # ── per-image rows ────────────────────────────────────────────────────────
    for img_orig, (path, expected) in zip(images, CASES):
        img_name = path.split("/")[-1].replace("arena_filter_", "").replace("_raw.png", "")
        row = "{:<22}  {:>8.2f}".format(img_name[:22], expected)

        if img_orig is None:
            row += "  [MISSING IMAGE]"
            print(row)
            continue

        for label, prep_fn, mode in PREPS:
            engine = ocr_nodet if mode == "nodet" else ocr
            proc = prep_fn(img_orig)
            val, raw = rapid_ocr_best(engine, proc)
            ok = val is not None and abs(val - expected) <= TOLERANCE
            if ok:
                scores[label] += 1
                cell = "OK   {:.2f}->  {:.2f}".format(expected, val)
            else:
                cell = "--   {!r:.10}->{}".format(
                    raw[:10] if raw else "", "{:.2f}".format(val) if val is not None else "None")
            row += "  {:<{}}".format(cell, col_w)

        print(row)

    # ── summary ──────────────────────────────────────────────────────────────
    print("-" * len(header))
    print("\nScores out of {} cases:".format(len(CASES)))
    best_score = 0
    best_label = None
    for label, _, __ in PREPS:
        s = scores[label]
        bar = "#" * s + "." * (len(CASES) - s)
        print("  {:<22} {}  {}/{}".format(label, bar, s, len(CASES)))
        if s > best_score:
            best_score = s
            best_label = label

    print("\nBest config: [{}]  ({}/{} correct)".format(best_label, best_score, len(CASES)))

    # ── per-prep cascade: each prep independently uses rapid_ocr_best ────────
    print("\n--- Per-prep BEST-BOX test (each box checked separately) ---")
    print("{:<22}  {:>8}  {}".format("Image", "Expected", "  ".join(
        "{:<18}".format(l) for l, _, __ in PREPS)))
    print("-" * 160)
    cascade_correct = 0
    # "cascade": try preps A→H, pick first result with decimal; else fewest digits
    for img_orig, (path, expected) in zip(images, CASES):
        img_name = path.split("/")[-1].replace("arena_filter_", "").replace("_raw.png", "")
        if img_orig is None:
            print("  {:22} MISSING".format(img_name))
            continue

        fallbacks = []
        chosen_val = None
        chosen_raw = None
        chosen_prep = None

        for label, prep_fn, mode in PREPS:
            engine = ocr_nodet if mode == "nodet" else ocr
            proc = prep_fn(img_orig)
            val, raw = rapid_ocr_best(engine, proc)
            if val is None:
                continue
            if "." in raw or "," in raw:
                chosen_val, chosen_raw, chosen_prep = val, raw, label
                break
            digit_count = len(re.sub(r"\D", "", raw))
            fallbacks.append((digit_count, val, raw, label))

        if chosen_val is None and fallbacks:
            fallbacks.sort(key=lambda t: t[0])
            _, chosen_val, chosen_raw, chosen_prep = fallbacks[0]

        ok = chosen_val is not None and abs(chosen_val - expected) <= TOLERANCE
        status = "OK" if ok else "FAIL"
        if ok:
            cascade_correct += 1
        print("  {:22} {:>8.2f}  got={:>7}  via={}  raw={!r:.25}  [{}]".format(
            img_name[:22], expected,
            "{:.2f}".format(chosen_val) if chosen_val is not None else "None",
            chosen_prep or "-",
            chosen_raw or "",
            status,
        ))

    print("Cascade (RapidOCR only): {}/{} correct".format(cascade_correct, len(CASES)))


if __name__ == "__main__":
    main()
