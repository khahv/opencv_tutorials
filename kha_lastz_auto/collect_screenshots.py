"""
Screenshot collector for YOLO training data.

Cách dùng:
1. Chạy script này SONG SONG với main.py bot
2. Nhấn phím 'S' để lưu screenshot hiện tại
3. Ảnh sẽ được lưu vào yolo_dataset/raw_screenshots/
4. Sau đó dùng LabelImg hoặc Roboflow để gán nhãn

python collect_screenshots.py
"""

import cv2 as cv
import os
import time
import datetime
from pathlib import Path
from pynput import keyboard
from windowcapture import WindowCapture

OUTPUT_DIR = Path("yolo_dataset/raw_screenshots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_NAME = "LastZ"   # Tên cửa sổ game

print("=== Screenshot Collector for YOLO Training ===")
print(f"Lưu vào: {OUTPUT_DIR.absolute()}")
print("Nhấn S = lưu screenshot | Nhấn ESC = thoát")
print("NOTE: Chụp nhiều màn hình khác nhau, càng nhiều button xuất hiện cùng lúc càng tốt")

try:
    wincap = WindowCapture(WINDOW_NAME)
except Exception as e:
    print(f"Không tìm thấy cửa sổ '{WINDOW_NAME}': {e}")
    exit(1)

save_triggered = False
quit_triggered = False

def on_press(key):
    global save_triggered, quit_triggered
    if key == keyboard.Key.esc:
        quit_triggered = True
        return False
    try:
        if key.char and key.char.lower() == 's':
            save_triggered = True
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()

count = 0
print("\nĐang chạy... (nhấn S để chụp, ESC để thoát)")

while not quit_triggered:
    screenshot = wincap.get_screenshot()
    if screenshot is None:
        time.sleep(0.5)
        continue

    time.sleep(0.1)  # Giảm tải CPU

    if save_triggered:
        save_triggered = False
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        filename = OUTPUT_DIR / f"screen_{ts}.png"
        cv.imwrite(str(filename), screenshot)
        count += 1
        print(f"[{count}] Saved: {filename.name}")

listener.stop()
print(f"\nDone! Saved {count} screenshots to {OUTPUT_DIR.absolute()}")
print("Bước tiếp theo: Upload ảnh lên Roboflow để gán nhãn bounding box")
