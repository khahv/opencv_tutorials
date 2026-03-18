"""
Debug script: why does find_multi_with_scores give 0.583 for RadarRedDot
while raw matchTemplate gives 0.9951?

Usage:
  python kha_lastz_auto/debug_exclude_score.py [crop_path] [needle_path]

Defaults to the problematic saved crop.
"""
import sys
import os
import numpy as np
import cv2 as cv

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from vision import Vision, set_global_scale

CROP_PATH   = os.path.join(os.path.dirname(__file__),
    "debug_exclude", "exclude_kept_140319_432_462,618.png")
NEEDLE_PATH = os.path.join(os.path.dirname(__file__),
    "buttons_template", "RadarRedDot.png")

if len(sys.argv) >= 3:
    CROP_PATH, NEEDLE_PATH = sys.argv[1], sys.argv[2]
elif len(sys.argv) == 2:
    CROP_PATH = sys.argv[1]

set_global_scale(1.0)

crop   = cv.imread(CROP_PATH)
needle = cv.imread(NEEDLE_PATH, cv.IMREAD_UNCHANGED)
vision = Vision(NEEDLE_PATH)

if crop is None:
    print("[ERROR] Cannot load crop:", CROP_PATH)
    sys.exit(1)

print("=" * 60)
print("crop   shape:", crop.shape)
print("needle shape:", needle.shape, "-> after BGRA->BGR:", vision.needle_img.shape)
print()

# ── 1. Raw matchTemplate (gray) best score ────────────────────
crop_gray   = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
result_gray = cv.matchTemplate(crop_gray, vision.needle_gray, cv.TM_CCOEFF_NORMED)
_, raw_best, _, raw_loc = cv.minMaxLoc(result_gray)
print("[1] Raw gray matchTemplate")
print("    best score : {:.4f}".format(raw_best))
print("    best loc   : {}  (top-left of needle placement)".format(raw_loc))
center_x = raw_loc[0] + vision.needle_w  // 2
center_y = raw_loc[1] + vision.needle_h // 2
print("    center     : ({}, {})".format(center_x, center_y))
print()

# ── 2. How many pixels pass various thresholds? ───────────────
print("[2] Number of result pixels >= threshold")
for thr in [0.9, 0.7, 0.5, 0.3, 0.1, 0.01, 0.0]:
    count = int(np.sum(result_gray >= thr))
    print("    >= {:.2f} : {:>6} pixels".format(thr, count))
print()

# ── 3. What groupRectangles does with threshold=0.01 ─────────
print("[3] groupRectangles behaviour with threshold=0.01 (root cause)")
locs_001 = list(zip(*np.where(result_gray >= 0.01)[::-1]))
rects_001 = []
for loc in locs_001:
    r = [int(loc[0]), int(loc[1]), vision.needle_w, vision.needle_h]
    rects_001.append(r)
    rects_001.append(r)
print("    raw rects fed to groupRectangles : {}".format(len(rects_001) // 2))
grouped_001, _ = cv.groupRectangles(rects_001, groupThreshold=1, eps=0.5)
print("    output groups                    : {}".format(len(grouped_001)))
for g in grouped_001:
    x, y, w, h = g
    cx, cy = x + w // 2, y + h // 2
    r_w = min(w, result_gray.shape[1] - x)
    r_h = min(h, result_gray.shape[0] - y)
    score = float(np.max(result_gray[y:y+r_h, x:x+r_w])) if r_w > 0 and r_h > 0 else 0.0
    print("    group rect ({:>3},{:>3} {:>2}x{:>2}) -> center=({},{})  max_score_in_rect={:.4f}".format(
        x, y, w, h, cx, cy, score))
print()

# ── 4. find_multi_with_scores result ─────────────────────────
print("[4] find_multi_with_scores(threshold=0.01, is_color=False)")
pts = vision.find_multi_with_scores(crop, threshold=0.01, is_color=False)
print("    returned:", pts)
print()

# ── 5. Correct approach: match_score() ───────────────────────
correct_score = vision.match_score(crop)
print("[5] vision.match_score(crop)  =  {:.4f}  <- this is the correct value".format(correct_score))
print()

# ── 6. What groupRectangles does with a SANE threshold ────────
print("[6] groupRectangles with threshold=0.5 (sane)")
locs_05 = list(zip(*np.where(result_gray >= 0.5)[::-1]))
rects_05 = []
for loc in locs_05:
    r = [int(loc[0]), int(loc[1]), vision.needle_w, vision.needle_h]
    rects_05.append(r)
    rects_05.append(r)
print("    raw rects fed to groupRectangles : {}".format(len(rects_05) // 2))
grouped_05, _ = cv.groupRectangles(rects_05, groupThreshold=1, eps=0.5)
print("    output groups                    : {}".format(len(grouped_05)))
for g in grouped_05:
    x, y, w, h = g
    cx, cy = x + w // 2, y + h // 2
    r_w = min(w, result_gray.shape[1] - x)
    r_h = min(h, result_gray.shape[0] - y)
    score = float(np.max(result_gray[y:y+r_h, x:x+r_w])) if r_w > 0 and r_h > 0 else 0.0
    print("    group rect ({:>3},{:>3} {:>2}x{:>2}) -> center=({},{})  max_score_in_rect={:.4f}".format(
        x, y, w, h, cx, cy, score))
print()

# ── 7. Save annotated debug image ─────────────────────────────
out_path = os.path.join(os.path.dirname(CROP_PATH),
    os.path.splitext(os.path.basename(CROP_PATH))[0] + "_debug_annotated.png")
vis = crop.copy()
# Mark raw best match (green)
cv.rectangle(vis, raw_loc, (raw_loc[0]+vision.needle_w, raw_loc[1]+vision.needle_h), (0,255,0), 1)
cv.putText(vis, "{:.2f}".format(raw_best), (raw_loc[0], raw_loc[1]-2),
           cv.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)
# Mark find_multi_with_scores results (red)
for p in pts:
    rx = p[0] - p[2] // 2
    ry = p[1] - p[3] // 2
    cv.rectangle(vis, (rx, ry), (rx+p[2], ry+p[3]), (0,0,255), 1)
    cv.putText(vis, "{:.2f}".format(p[4]), (rx, ry-2),
               cv.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 1)
cv.imwrite(out_path, vis)
print("[7] Annotated image saved ->", out_path)
print("    GREEN = raw best match (correct) | RED = find_multi_with_scores (wrong)")
print()
print("=" * 60)
print("CONCLUSION")
print("  find_multi_with_scores uses groupRectangles with threshold=0.01.")
print("  With threshold=0.01, almost ALL result pixels pass -> thousands of")
print("  near-identical rects fed to groupRectangles -> they merge into an")
print("  averaged position that doesn't correspond to the true best match.")
print("  The correct fix: use match_score() for the exclude check, which")
print("  directly returns minMaxLoc best score without groupRectangles.")
