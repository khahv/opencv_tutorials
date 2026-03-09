"""
Zalo Clicker — click theo template trong cua so "Zalo (32 bit)" dung OpenCV.
Chay: python zalo_clicker.py [duong_dan_template.png]
     Neu khong truyen template: dung template mac dinh trong thu muc zalo_templates/
Phim F8: tim template va click 1 lan (khi dang chay interactive).
"""
import os
import sys
import time
import logging
import cv2 as cv

# Thu muc chua script (dung cho duong dan template khi goi tu module khac)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Khi chay rieng, chdir de import va duong dan tuong doi hoat dong
if __name__ == "__main__":
    os.chdir(SCRIPT_DIR)

from windowcapture import WindowCapture
from vision import Vision
import pyautogui

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("zalo_clicker")

ZALO_WINDOW_NAME = "Zalo (32 bit)"
DEFAULT_TEMPLATES_DIR = "zalo_templates"
THRESHOLD = 0.7


def run_zalo_click(template_path=None, threshold=THRESHOLD, click=True, logger=None):
    """
    Tim template trong cua so Zalo va click (neu click=True).
    Co the goi tu module khac (vd. attack_detector khi phat hien bi tan cong).

    Args:
        template_path: duong dan anh mau. None = lay file dau tien trong zalo_templates/
        threshold: nguong match (0..1)
        click: True thi click, False chi tra ve toa do
        logger: logger de ghi log (neu None dung log mac dinh)

    Returns:
        (sx, sy) neu tim thay (va click neu click=True), None neu khong thay hoac loi.
    """
    _log = logger or log
    base_dir = SCRIPT_DIR
    if template_path is None:
        templates_dir = os.path.join(base_dir, DEFAULT_TEMPLATES_DIR)
        if not os.path.isdir(templates_dir):
            _log.warning("[ZaloClicker] Thu muc %s khong ton tai.", templates_dir)
            return None
        templates = [
            os.path.join(templates_dir, f)
            for f in os.listdir(templates_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not templates:
            _log.warning("[ZaloClicker] Khong co template trong %s.", templates_dir)
            return None
        template_path = templates[0]
    else:
        if not os.path.isabs(template_path):
            template_path = os.path.join(base_dir, template_path)
        if not os.path.isfile(template_path):
            _log.warning("[ZaloClicker] File khong ton tai: %s", template_path)
            return None

    try:
        wincap = WindowCapture(ZALO_WINDOW_NAME)
    except Exception as e:
        _log.debug("[ZaloClicker] Cua so Zalo khong tim thay: %s", e)
        return None

    vision = Vision(template_path)
    return find_and_click(wincap, vision, threshold=threshold, click=click)


def find_and_click(wincap, vision, threshold=THRESHOLD, click=True):
    """
    Chup man hinh Zalo, tim template, neu thay thi click vao vi tri dau tien.
    Tra ve (sx, sy) neu tim thay va da click (hoac click=False thi van tra ve toa do), None neu khong thay.
    """
    wincap.focus_window()
    time.sleep(0.05)
    img = wincap.get_screenshot()
    if img is None:
        log.warning("Khong chup duoc man hinh Zalo.")
        return None
    points = vision.find(img, threshold=threshold)
    if not points:
        return None
    cx, cy = points[0]
    sx, sy = wincap.get_screen_position((cx, cy))
    if click:
        pyautogui.click(sx, sy)
        log.info("Click tai ({}, {}) screen".format(sx, sy))
    return (sx, sy)


def main():
    # Template: tu tham so dong lenh hoac thu muc mac dinh
    if len(sys.argv) >= 2:
        template_path = sys.argv[1]
    else:
        os.makedirs(DEFAULT_TEMPLATES_DIR, exist_ok=True)
        # Lay file .png hoac .jpg dau tien trong zalo_templates
        templates = [
            os.path.join(DEFAULT_TEMPLATES_DIR, f)
            for f in os.listdir(DEFAULT_TEMPLATES_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not templates:
            log.error(
                "Khong co template. Tao thu muc '{}' va dat anh mau vao, hoac chay: python zalo_clicker.py <path_template.png>".format(
                    DEFAULT_TEMPLATES_DIR
                )
            )
            sys.exit(1)
        template_path = templates[0]
        log.info("Dung template: {}".format(template_path))

    if not os.path.isfile(template_path):
        log.error("Khong tim thay file: {}".format(template_path))
        sys.exit(1)

    try:
        wincap = WindowCapture(ZALO_WINDOW_NAME)
    except Exception as e:
        log.error("Khong tim thay cua so '{}': {}".format(ZALO_WINDOW_NAME, e))
        log.info("Goi WindowCapture.list_window_names() de xem ten cua so dang mo.")
        sys.exit(1)

    vision = Vision(template_path)

    # Che do 1 lan: tim va click xong thoat
    if len(sys.argv) >= 2:
        result = find_and_click(wincap, vision)
        if result:
            log.info("Da click thanh cong.")
        else:
            log.warning("Khong tim thay template tren man hinh.")
        return

    # Interactive: F8 = find + click, Esc = thoat
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.f8:
                find_and_click(wincap, vision)
            if key == keyboard.Key.esc:
                return False  # stop listener
        except Exception as e:
            log.exception(e)

    log.info("Zalo Clicker: mo cua so Zalo, nhan F8 de tim template va click, Esc de thoat.")
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == "__main__":
    main()
