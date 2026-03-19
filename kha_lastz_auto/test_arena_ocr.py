"""
test_arena_ocr.py
-----------------
Offline test: try multiple OCR configurations on the saved debug crops and
report which configuration reads all power values correctly.

Run from the kha_lastz_auto directory:
    python test_arena_ocr.py

Edit CASES below to point at whichever debug images you have on disk.
"""

import os
import re
import sys
import cv2 as cv
import numpy as np

# ── Ground-truth cases ─────────────────────────────────────────────────────────
CASES = [
    # ── Set 1 (083528) ────────────────────────────────────────────────────────
    ("debug_ocr/arena_filter_my_power_083528_742_raw.png",   27.46),
    ("debug_ocr/arena_filter_opponent0_083529_963_raw.png",  40.90),
    ("debug_ocr/arena_filter_opponent1_083530_665_raw.png",  32.01),
    ("debug_ocr/arena_filter_opponent2_083531_365_raw.png",  26.84),
    ("debug_ocr/arena_filter_opponent3_083531_562_raw.png",  23.99),
    ("debug_ocr/arena_filter_opponent4_083531_742_raw.png",  22.94),
    # ── Set 2 (092033) ────────────────────────────────────────────────────────
    ("debug_ocr/arena_filter_my_power_092033_703_raw.png",   27.46),
    ("debug_ocr/arena_filter_opponent0_092035_222_raw.png",  29.52),
    ("debug_ocr/arena_filter_opponent1_092036_383_raw.png",  29.68),
    ("debug_ocr/arena_filter_opponent2_092037_252_raw.png",  25.70),
    ("debug_ocr/arena_filter_opponent3_092037_469_raw.png",  22.92),
    ("debug_ocr/arena_filter_opponent4_092038_679_raw.png",  25.62),
]

TOLERANCE  = 0.06    # allowed absolute error for a match to count as correct
_VALID_MIN = 1.0     # minimum plausible M power value in arena
_VALID_MAX = 999.0   # maximum plausible M power value in arena

# ── Offline unit tests for normalize_power + extract_power ────────────────────
# These do NOT need images or EasyOCR — run before the OCR section.
#
# Format: (raw_ocr_string, expected_float_or_None)
#
# Key cases:
#   - Decimal correctly read               → use as-is
#   - Decimal dropped (2-digit before .)   → /100, artifact (+M→0) excluded by range
#   - Decimal dropped (3-digit before .)   → /100, artifact (>999) excluded by range
#   - M misread as trailing digit          → handled by \d{1,2} in regex
NORMALIZE_UNIT_TESTS = [
    # ── decimal correctly read ────────────────────────────────────────────────
    ("27.46",   27.46),
    ("27.461",  27.46),   # trailing M→1 stripped by {1,2}
    ("27.460",  27.46),   # trailing M→0 stripped by {1,2}
    ("40.90",   40.90),
    ("22.94",   22.94),
    ("125.33",  125.33),
    ("125.331", 125.33),  # trailing M→1
    ("9.50",    9.50),
    # ── decimal dropped (2 digits before decimal) ─────────────────────────────
    ("2746",    27.46),
    ("4090",    40.90),
    ("2294",    22.94),
    ("3201",    32.01),
    # ── decimal dropped (3 digits before decimal) ─────────────────────────────
    ("12533",   125.33),
    # ── M→0 artifact: 2-digit before decimal → artifact IN range
    #    normalize_power alone returns these; the CASCADE's "fewest-digit" rule
    #    ensures they are NOT selected when the shorter correct form is present.
    ("22940",   229.40),  # artifact of 22.94M — parse returns 229.4 (cascade discards)
    ("27460",   274.60),  # artifact of 27.46M — same
    ("40900",   409.00),  # artifact of 40.90M (M→0) — normalize returns 409.0; cascade discards it
    ("40901",   409.01),  # artifact of 40.90M (M→1) — normalize returns 409.01; cascade discards it
    # ── M→0 artifact: 3-digit before decimal → artifact > 999, excluded by range ──
    ("125330",  None),    # 1253.3 > 999 → excluded by range filter ← safe for 100M+ values
    # ── edge / boundary ───────────────────────────────────────────────────────
    ("999",     9.99),
    ("99.99",   99.99),
    ("100",     1.00),
    ("1.00",    1.00),
    ("",        None),
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def scale_up(img, factor=None):
    """Scale small crops up so EasyOCR works better."""
    h, w = img.shape[:2]
    if factor:
        return cv.resize(img, (w * factor, h * factor), interpolation=cv.INTER_CUBIC)
    if max(h, w) < 80:
        return cv.resize(img, (w * 3, h * 3), interpolation=cv.INTER_CUBIC)
    if max(h, w) < 150:
        return cv.resize(img, (w * 2, h * 2), interpolation=cv.INTER_CUBIC)
    return img


def preprocess_white_thresh(img, threshold=180):
    """Keep only bright/white pixels — good for white text on dark backgrounds."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)
    return cv.cvtColor(mask, cv.COLOR_GRAY2BGR)


def preprocess_otsu(img):
    """Otsu binarisation — adapts to the image; works for both dark-on-light and
    light-on-dark text by checking which side has more pixels and inverting."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Make sure text is white-on-black (EasyOCR prefers this)
    if np.mean(mask) > 127:      # most pixels are white → invert (text was dark)
        mask = cv.bitwise_not(mask)
    return cv.cvtColor(mask, cv.COLOR_GRAY2BGR)


def preprocess_invert_white_thresh(img, threshold=180):
    """Threshold then invert — good for dark/coloured text on light backgrounds."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY_INV)
    return cv.cvtColor(mask, cv.COLOR_GRAY2BGR)


def normalize_power(num_str: str):
    """Convert a raw matched number string to a float in M units.

    If no decimal point is present and value >= 100, assume last 2 digits are
    decimals (OCR dropped the decimal point).
    """
    num_str = num_str.replace(",", ".")
    try:
        val = float(num_str)
    except (ValueError, TypeError):
        return None
    if "." not in num_str and val >= 100:
        val = val / 100
    return val


def extract_power(raw: str):
    """Pull the first decimal-or-integer number out of *raw* OCR text.

    Decimal part is limited to 2 digits so a trailing misread "M"→"1" or "0"
    is ignored (e.g. "27.461" → 27.46, "274460" → 2744.6 which we then fix via
    normalize_power, "27.460" → 27.46).
    """
    raw_clean = re.sub(r"[^\d,.]", "", raw)
    m = re.search(r"(\d+[.,]\d{1,2}|\d+)", raw_clean)
    if not m:
        return None, raw_clean
    return normalize_power(m.group(1)), raw_clean


# ── OCR configurations to test ─────────────────────────────────────────────────
#
# Each entry: (label, preprocess_fn, readtext_kwargs)
# preprocess_fn receives the ALREADY SCALED image.
# readtext_kwargs are passed directly to reader.readtext().

def _no_pre(img): return img
def _scale3x(img): return scale_up(img, factor=3)
def _scale4x(img): return scale_up(img, factor=4)

# ── Grayscale helpers (keep 3-channel BGR so EasyOCR accepts them) ─────────────
def _to_gray_bgr(img):
    """Convert to grayscale then back to BGR (neutral, no threshold)."""
    return cv.cvtColor(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)

def _gray_th150(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, th = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
    return cv.cvtColor(th, cv.COLOR_GRAY2BGR)

def _gray_th100(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, th = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
    return cv.cvtColor(th, cv.COLOR_GRAY2BGR)

def _gray_adaptive(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY, 11, 2)
    return cv.cvtColor(th, cv.COLOR_GRAY2BGR)

def _gray_adaptive_inv(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY_INV, 11, 2)
    return cv.cvtColor(th, cv.COLOR_GRAY2BGR)

def _3x_gray(img):       return _to_gray_bgr(_scale3x(img))
def _3x_gray_th150(img): return _gray_th150(_scale3x(img))
def _3x_gray_adaptive(img): return _gray_adaptive(_scale3x(img))


_CASCADE_STEPS = [
    # (label, preprocess_fn, readtext_kwargs)
    #
    # E — colour image, para=False, no allowlist.
    #     First because it avoids M→0 artifact AND reads light-coloured text
    #     (e.g. opponent2 pink/light-red text) more cleanly than para=True.
    # A — colour image, paragraph merge + allowlist.
    #     Merges split boxes; good for coloured text on dark backgrounds.
    # B — white thresh 180: great for white text on dark backgrounds (my_power).
    # C — otsu binarisation: adapts automatically; reads most cases correctly.
    # T — grayscale + adaptive-threshold inverted, paragraph + allowlist.
    #     Best for red text on light backgrounds that C misses (opponent1 type).
    #     Placed AFTER C so C's correct decimal takes priority when both succeed.
    # G — white thresh 160: lower threshold, last-resort fallback.
    ("E", _no_pre,                                     dict(paragraph=False)),
    ("A", _no_pre,                                     dict(paragraph=True,  allowlist="0123456789,.")),
    ("B", preprocess_white_thresh,                     dict(paragraph=True,  allowlist="0123456789,.")),
    ("C", preprocess_otsu,                             dict(paragraph=True,  allowlist="0123456789,.")),
    ("T", _gray_adaptive_inv,                          dict(paragraph=True,  allowlist="0123456789,.")),
    ("G", lambda i: preprocess_white_thresh(i, 160),   dict(paragraph=True,  allowlist="0123456789,.")),
]

_VALID_MIN = 1.0
_VALID_MAX = 999.0


def _run_step(reader, img, preprocess, kwargs):
    processed = preprocess(img)
    results = reader.readtext(processed, detail=0, **kwargs)
    # For E (no allowlist) keep everything; for others strip non-numeric
    if kwargs.get("allowlist"):
        raw = re.sub(r"[^\d,.]", "", " ".join(str(r) for r in results).strip())
    else:
        raw = " ".join(str(r) for r in results).strip()
    val, cleaned = extract_power(raw)
    return val, cleaned, raw


def _cascade_ocr(reader, img):
    """Priority cascade:

    Pass 1 — HIGH CONFIDENCE: return the first result that has a decimal point
              AND is in the valid power range (1–999 M).  A decimal means EasyOCR
              read the separator correctly (no normalization needed).

    Pass 2 — FALLBACK: when no decimal is found anywhere, collect all results in
              the valid range and return the one with the FEWEST digits.  Fewer
              digits means the 'M' suffix was not misread as an extra digit (e.g.
              "2294" is correct; "22940" has an extra '0' from M→0 artifact).
    """
    fallbacks = []   # (digit_count, val, cleaned)

    for _lbl, preprocess, kwargs in _CASCADE_STEPS:
        val, cleaned, raw = _run_step(reader, img, preprocess, kwargs)
        if val is None or not (_VALID_MIN <= val <= _VALID_MAX):
            continue
        # High-confidence: decimal present in raw OCR
        if "." in raw or "," in raw:
            return val, cleaned
        # Collect as fallback, tagged with digit count
        digit_count = len(re.sub(r"\D", "", cleaned))
        fallbacks.append((digit_count, val, cleaned))

    if fallbacks:
        # Prefer fewest digits → least likely to have extra M-artifact digit
        fallbacks.sort(key=lambda t: t[0])
        return fallbacks[0][1], fallbacks[0][2]

    return None, ""


CONFIGS = [
    # ── A: Current "no-threshold paragraph" approach ───────────────────────────
    ("A: no_thresh|para=T|allow",
     _no_pre,
     dict(paragraph=True, allowlist="0123456789,.")),

    # ── B: Original "white threshold" approach ────────────────────────────────
    ("B: wthresh180|para=T|allow",
     preprocess_white_thresh,
     dict(paragraph=True, allowlist="0123456789,.")),

    # ── C: Otsu binarisation ──────────────────────────────────────────────────
    ("C: otsu|para=T|allow",
     preprocess_otsu,
     dict(paragraph=True, allowlist="0123456789,.")),

    # ── D: Inverted white-thresh ──────────────────────────────────────────────
    ("D: inv180|para=T|allow",
     preprocess_invert_white_thresh,
     dict(paragraph=True, allowlist="0123456789,.")),

    # ── E: No threshold, no allowlist, no paragraph ───────────────────────────
    ("E: no_thresh|para=F|no_allow",
     _no_pre,
     dict(paragraph=False)),

    # ── G: White thresh 160 ───────────────────────────────────────────────────
    ("G: wthresh160|para=T|allow",
     lambda img: preprocess_white_thresh(img, 160),
     dict(paragraph=True, allowlist="0123456789,.")),

    # ── J: 3x scale + no_thresh ───────────────────────────────────────────────
    ("J: 3x+no_thresh|para=T|allow",
     _scale3x,
     dict(paragraph=True, allowlist="0123456789,.")),

    # ── K: 3x scale + white thresh ────────────────────────────────────────────
    ("K: 3x+wthresh|para=T|allow",
     lambda img: preprocess_white_thresh(_scale3x(img)),
     dict(paragraph=True, allowlist="0123456789,.")),

    # ── L: 3x scale + otsu ───────────────────────────────────────────────────
    ("L: 3x+otsu|para=T|allow",
     lambda img: preprocess_otsu(_scale3x(img)),
     dict(paragraph=True, allowlist="0123456789,.")),

    # ── M: 4x scale + no_thresh ───────────────────────────────────────────────
    ("M: 4x+no_thresh|para=T|allow",
     _scale4x,
     dict(paragraph=True, allowlist="0123456789,.")),

    # ── N: 4x scale + otsu ────────────────────────────────────────────────────
    ("N: 4x+otsu|para=T|allow",
     lambda img: preprocess_otsu(_scale4x(img)),
     dict(paragraph=True, allowlist="0123456789,.")),

    # ── Grayscale variants (new) ──────────────────────────────────────────────
    ("P: gray|para=T|allow",
     _to_gray_bgr,
     dict(paragraph=True,  allowlist="0123456789,.")),

    ("Q: gray+th150|para=T|allow",
     _gray_th150,
     dict(paragraph=True,  allowlist="0123456789,.")),

    ("R: gray+th100|para=T|allow",
     _gray_th100,
     dict(paragraph=True,  allowlist="0123456789,.")),

    ("S: gray+adaptive|para=T|allow",
     _gray_adaptive,
     dict(paragraph=True,  allowlist="0123456789,.")),

    ("T: gray+adap_inv|para=T|allow",
     _gray_adaptive_inv,
     dict(paragraph=True,  allowlist="0123456789,.")),

    ("U: gray|para=F|no_allow",
     _to_gray_bgr,
     dict(paragraph=False)),

    ("V: 3x+gray|para=T|allow",
     _3x_gray,
     dict(paragraph=True,  allowlist="0123456789,.")),

    ("W: 3x+gray+th150|para=T|allow",
     _3x_gray_th150,
     dict(paragraph=True,  allowlist="0123456789,.")),

    ("X: 3x+gray+adaptive|para=T|allow",
     _3x_gray_adaptive,
     dict(paragraph=True,  allowlist="0123456789,.")),

    # ── Z: Cascade — shown separately ────────────────────────────────────────
    ("Z: CASCADE(E->A->B->C->T->G)",
     None,   # special: handled in main()
     {}),
]


# ── Main ───────────────────────────────────────────────────────────────────────

def run_unit_tests():
    """Quick sanity-check for normalize_power + extract_power — no images needed."""
    print("\n=== Unit tests: normalize_power + extract_power ===")
    fails = 0
    for raw, expected in NORMALIZE_UNIT_TESTS:
        val, _ = extract_power(raw)
        if expected is None:
            ok = (val is None) or not (_VALID_MIN <= val <= _VALID_MAX)
        else:
            ok = (val is not None) and abs(val - expected) <= TOLERANCE
        status = "OK" if ok else "FAIL"
        if not ok:
            fails += 1
        print("  {} | raw={!r:12} expected={:>8} got={}".format(
            status, raw,
            "None" if expected is None else "{:.2f}".format(expected),
            "None" if val is None else "{:.2f}".format(val),
        ))
    print("Unit tests: {}/{} passed\n".format(len(NORMALIZE_UNIT_TESTS) - fails, len(NORMALIZE_UNIT_TESTS)))
    return fails == 0


def main():
    run_unit_tests()

    from ocr_easyocr import _ensure_reader
    reader = _ensure_reader()
    if reader is None:
        print("EasyOCR not available — install it first.")
        sys.exit(1)

    col_w = 26
    header = f"{'Image':<18} {'Expected':>8}  "
    header += "  ".join(f"{c[0][:col_w]:^{col_w}}" for c in CONFIGS)
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    scores   = [0] * len(CONFIGS)
    n_cases  = 0

    for path, expected in CASES:
        img = cv.imread(path)
        label = os.path.basename(path).replace("arena_filter_", "")
        label = re.sub(r"_\d{6}_\d{3}_raw\.png$", "", label)[:17]

        if img is None:
            print(f"  SKIP (file not found): {path}")
            continue

        n_cases += 1
        scaled = scale_up(img)
        row = f"{label:<18} {expected:>8.2f}  "

        for i, (cfg_name, preprocess, kwargs) in enumerate(CONFIGS):
            # Special cascade config
            if preprocess is None:
                val, cleaned = _cascade_ocr(reader, scaled)
                ok = (val is not None) and (abs(val - expected) <= TOLERANCE)
                if ok:
                    scores[i] += 1
                marker = "OK" if ok else "--"
                cell = f"{marker} {cleaned[:8]:>8}->{val!s:>6}"
                row += f"  {cell:^{col_w}}"
                continue

            processed = preprocess(scaled)
            try:
                results = reader.readtext(processed, detail=0, **kwargs)
                raw = " ".join(str(r) for r in results).strip()
            except Exception as exc:
                raw = ""
                print(f"    [{cfg_name}] error: {exc}")

            val, cleaned = extract_power(raw)
            ok = (val is not None) and (abs(val - expected) <= TOLERANCE)
            if ok:
                scores[i] += 1

            marker = "OK" if ok else "--"
            cell = f"{marker} {cleaned[:8]:>8}->{val!s:>6}"
            row += f"  {cell:^{col_w}}"

        print(row)

    print(sep)
    print(f"\nScores out of {n_cases} cases:")
    for i, (cfg_name, _, _) in enumerate(CONFIGS):
        bar = "#" * scores[i] + "." * (n_cases - scores[i])
        print(f"  {cfg_name:<42}  {bar}  {scores[i]}/{n_cases}")

    best = max(range(len(CONFIGS)), key=lambda i: scores[i])
    print(f"\nBest config: [{CONFIGS[best][0]}]  ({scores[best]}/{n_cases} correct)")


if __name__ == "__main__":
    main()
