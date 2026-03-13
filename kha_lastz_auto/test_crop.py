import cv2
import easyocr
import os

img = cv2.imread('debug_set_level_roi.png', cv2.IMREAD_GRAYSCALE)
reader = easyocr.Reader(['en'], gpu=False)

def test_offset(crop_x, crop_y, crop_w, crop_h, scale=2.0):
    h, w = img.shape
    x1 = max(0, crop_x)
    y1 = max(0, crop_y)
    x2 = min(w, crop_x + crop_w)
    y2 = min(h, crop_y + crop_h)
    
    crop = img[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    cv2.imwrite(f'crop_x{crop_x}_w{crop_w}.png', crop_resized)
    res = reader.readtext(crop_resized, detail=0)
    print(f"Crop x:{crop_x:2} w:{crop_w:2} scale:{scale} -> {res}")

print("Testing full size:")
test_offset(0, 0, 70, 35, 1.0)
test_offset(0, 0, 70, 35, 2.0)
test_offset(0, 0, 70, 35, 3.0)

print("\r\nTesting crops:")
test_offset(25, 0, 45, 35, 1.0)
test_offset(25, 0, 45, 35, 2.0)
test_offset(25, 0, 45, 35, 3.0)
test_offset(30, 0, 40, 25, 1.0)
test_offset(30, 0, 40, 25, 2.0)
test_offset(30, 0, 40, 25, 3.0)
test_offset(32, 0, 30, 24, 2.0)
