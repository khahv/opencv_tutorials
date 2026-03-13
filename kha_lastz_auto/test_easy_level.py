import cv2
import easyocr
import re

img = cv2.imread('debug_set_level_roi.png')
r = easyocr.Reader(['en'], gpu=False, verbose=False)

def test_ocr(image, note=""):
    # resize up
    h, w = image.shape[:2]
    resized = cv2.resize(image, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
    
    # Try digits_only mode (from ocr_easyocr)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    res_thresh = r.readtext(cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR), detail=0, allowlist="0123456789,.", paragraph=True)
    
    print(f"{note} threshold: {res_thresh}")

test_ocr(img, "Full ROI")

# crop middle to right
h, w = img.shape[:2]
crop = img[0:h, 25:w]
test_ocr(crop, "Cropped")
