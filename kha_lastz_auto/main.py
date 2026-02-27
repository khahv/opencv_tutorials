import cv2 as cv
import numpy as np
import os
import pyautogui
from windowcapture import WindowCapture
from vision import Vision

# Change the working directory to the folder this script is in.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# log list all windows currently running
print('=== Cac cua so dang chay (handle, ten) ===')
WindowCapture.list_window_names()
print('==========================================')

# initialize the WindowCapture class
wincap = WindowCapture('LastZ')

# template nut Mail
vision_mail = Vision('buttons_template/Mail.png')

mail_clicked = False
MAIL_THRESHOLD = 0.75

while True:

    screenshot = wincap.get_screenshot()

    # tim nut Mail
    points = vision_mail.find(screenshot, threshold=MAIL_THRESHOLD, debug_mode=None)
    if points and not mail_clicked:
        center = points[0]
        screen_x, screen_y = wincap.get_screen_position(center)
        print('[Mail] Vi tri trong anh: {}, Toa do man hinh: ({}, {})'.format(center, screen_x, screen_y))
        pyautogui.click(screen_x, screen_y)
        mail_clicked = True
        print('[Mail] Da click.')
    if points:
        # ve khung quanh nut Mail (vi tri dau tien)
        cx, cy = points[0]
        w, h = vision_mail.needle_w, vision_mail.needle_h
        x1, y1 = cx - w // 2, cy - h // 2
        cv.rectangle(screenshot, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

    cv.imshow('LastZ Capture', screenshot)

    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

print('Done.')
