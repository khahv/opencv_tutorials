import cv2
import easyocr

img = cv2.imread('debug_set_level_roi.png')
r = easyocr.Reader(['en'], gpu=False, verbose=False)

def test_ocr(crop_x, crop_w):
    h = img.shape[0]
    crop = img[0:h, crop_x:min(img.shape[1], crop_x+crop_w)]
    
    # Save crop to look at it
    cv2.imwrite(f'crop_x{crop_x}.png', crop)
    
    resized = cv2.resize(crop, (crop.shape[1]*3, h*3), interpolation=cv2.INTER_CUBIC)
    
    # Try digits_only mode (from ocr_easyocr)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    res_thresh = r.readtext(cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR), detail=0, allowlist="0123456789,.", paragraph=True)
    
    print(f"Crop x:{crop_x} w:{crop_w} threshold: {res_thresh}")

test_ocr(32, 50)
test_ocr(35, 50)
test_ocr(40, 50)
test_ocr(45, 50)
test_ocr(48, 50)
test_ocr(50, 50)
test_ocr(55, 50)
