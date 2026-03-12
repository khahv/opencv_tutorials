"""
Train YOLOv8 trên dataset game buttons.

Trước khi chạy:
1. Hoàn tất gán nhãn và có folder yolo_dataset/train/ + yolo_dataset/valid/
2. pip install ultralytics

Chạy:
    python train_yolo.py

Kết quả tại: runs/detect/game_buttons_v1/weights/best.pt
"""

from ultralytics import YOLO
import torch

# ---- Config ----
DATA_YAML    = "yolo_dataset/data.yaml"
MODEL        = "yolov8n.pt"   # nano = nhanh nhất; dùng yolov8s.pt nếu cần chính xác hơn
EPOCHS       = 100
IMG_SIZE     = 960            # Đúng với độ phân giải game 540x960 hoặc 1080x1920
BATCH        = 16
WORKERS      = 4
PROJECT_NAME = "game_buttons_v1"

# Tự động chọn device:
# - Mac M4 → MPS
# - NVIDIA  → CUDA
# - Khác   → CPU
if torch.backends.mps.is_available():
    device = "mps"
    print("[Train] Sử dụng Apple MPS (Mac M4/M-series)")
elif torch.cuda.is_available():
    device = 0
    print(f"[Train] Sử dụng CUDA: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("[Train] Sử dụng CPU (sẽ chậm hơn GPU)")

print(f"[Train] Model: {MODEL} | Epochs: {EPOCHS} | ImgSize: {IMG_SIZE}")
print(f"[Train] Dataset: {DATA_YAML}")

model = YOLO(MODEL)

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    workers=WORKERS,
    device=device,
    name=PROJECT_NAME,
    patience=20,         # Early stopping nếu không cải thiện sau 20 epochs
    save_period=10,      # Lưu checkpoint mỗi 10 epochs
    verbose=True,
)

print("\n=== Training Done! ===")
print(f"Model lưu tại: runs/detect/{PROJECT_NAME}/weights/best.pt")
print("Bước tiếp theo: Copy file .pt vào project và cập nhật MODEL_PATH trong vision.py")
