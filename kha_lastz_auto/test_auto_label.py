import cv2 as cv
import os
from vision import Vision

# Initialize Vision for Mail
mail_vision = Vision("buttons_template/Mail.png")
mail_vision._auto_label = True  # Enable auto-labeling

# The 3 test images
images = [
    "yolo_dataset/raw_screenshots/screen_20260312_110207_405.png",
    "yolo_dataset/raw_screenshots/screen_20260312_110218_719.png",
    "yolo_dataset/raw_screenshots/screen_20260312_110222_022.png"
]

print("Starting auto-annotation for test images...")
for img_path in images:
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        continue
        
    img = cv.imread(img_path)
    if img is None:
        print(f"Failed to load: {img_path}")
        continue
        
    print(f"Processing: {img_path}")
    # Run SIFT to find Mail. Since _auto_label is True, this will also save the YOLO format .txt
    # and a copy of the image to yolo_dataset/auto_labeled/
    # We use min_match_count=5 just to be safe it finds it
    result = mail_vision.find(img, min_match_count=5, debug_mode=None, auto_label=True)
    if result:
        print(f" -> Found Mail at {result[0]}")
    else:
        print(" -> Mail not found in this image using SIFT.")

print("Done. Check yolo_dataset/auto_labeled/ for results.")
