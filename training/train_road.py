#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_road.py
-------------
Training script to fine-tune a YOLO11n-seg model for Road Segmentation.
It employs a safe, stable training configuration (workers=0, cache='disk', batch=2)
to prevent multiprocessing or shared memory (/dev/shm) crash issues commonly
encountered in Docker/WSL environments.
"""

import os
from ultralytics import YOLO

# Trọng số nền phân đoạn YOLO11
BASE_MODEL = "yolo11n-seg.pt"

# Đường dẫn mặc định trong Docker và tương đối cục bộ
DOCKER_CONFIG = "/workspace/unet/road_seg.yaml"
LOCAL_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "road_seg.yaml")


def main():
    # 1. Tự động xác định đường dẫn cấu hình phù hợp
    if os.path.exists(DOCKER_CONFIG):
        config_path = DOCKER_CONFIG
    elif os.path.exists(LOCAL_CONFIG):
        config_path = LOCAL_CONFIG
    else:
        config_path = "configs/road_seg.yaml"

    print("=========================================================================")
    print("🏁 Khởi chạy chương trình Huấn luyện Mô hình Phân đoạn Làn đường Road Seg 🏁")
    print(f"  - Mô hình gốc: {BASE_MODEL}")
    print(f"  - Cấu hình dữ liệu: {config_path}")
    print("=========================================================================")

    # 2. Khởi tạo mô hình YOLO11n-seg
    print(f"Đang tải mô hình nền {BASE_MODEL}...")
    model = YOLO(BASE_MODEL)

    # 3. Tiến hành huấn luyện mô hình với cấu hình siêu ổn định
    print("Bắt đầu huấn luyện...")
    results = model.train(
        data=config_path,           # Đường dẫn file yaml
        epochs=50,                  # Số epochs khuyến nghị tối ưu (50 epochs)
        imgsz=640,                  # Độ phân giải ảnh đầu vào (640px cho độ nét phân đoạn)
        batch=2,                    # Giữ batch=2 để chống tràn VRAM/RAM GPU
        workers=0,                  # BẮT BUỘC đặt 0 trong Docker để tránh lỗi DataLoader worker exited
        cache='disk',               # Sử dụng bộ đệm đĩa cứng chống crash phân vùng bộ nhớ
        device=0,                   # Chỉ định chạy trên GPU số 0
        patience=15,                # Dừng sớm nếu sau 15 epochs chỉ số mAP không cải thiện
        optimizer="AdamW",          # Thuật toán tối ưu hóa tốt nhất cho YOLO phân đoạn
        lr0=1e-3,                   # Tốc độ học ban đầu
        lrf=1e-5,                   # Tốc độ học tối thiểu ở epoch cuối
        project="runs/segment",     # Thư mục lưu kết quả train
        name="train_road",          # Tên thư mục huấn luyện cụ thể
    )

    print("\n[Thành công] Quá trình huấn luyện đã hoàn tất!")
    print(f"Kết quả lưu trữ tại: runs/segment/train_road/")
    print(f"Trọng số tốt nhất đã sẵn sàng tại: runs/segment/train_road/weights/best.pt")


if __name__ == "__main__":
    main()
