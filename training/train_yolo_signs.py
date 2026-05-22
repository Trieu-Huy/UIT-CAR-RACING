#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_yolo_signs.py
-------------------
Training script to train a custom YOLO11n-seg model for Traffic Sign Segmentation.
It uses custom parameters such as imgsz=320, copy_paste augmentation, and crucially
sets fliplr=0.0 (horizontal flip disabled) to preserve directional meaning of traffic signs
(e.g., turn left vs. turn right, and do not turn left vs. do not turn right).
"""

import os
from ultralytics import YOLO

# Trọng số nền phân đoạn YOLO11
BASE_MODEL = "yolo11n-seg.pt"

# Đường dẫn mặc định trong Docker và tương đối cục bộ
DOCKER_CONFIG = "/workspace/signtrain/traffic_sign_seg.yaml"
LOCAL_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "traffic_sign_seg.yaml")


def main():
    # 1. Tự động xác định đường dẫn cấu hình phù hợp
    if os.path.exists(DOCKER_CONFIG):
        config_path = DOCKER_CONFIG
    elif os.path.exists(LOCAL_CONFIG):
        config_path = LOCAL_CONFIG
    else:
        config_path = "configs/traffic_sign_seg.yaml"

    print("=========================================================================")
    print("🚦 Khởi chạy chương trình Huấn luyện Mô hình Phân đoạn Biển báo Traffic Sign 🚦")
    print(f"  - Mô hình gốc: {BASE_MODEL}")
    print(f"  - Cấu hình dữ liệu: {config_path}")
    print("=========================================================================")

    # 2. Khởi tạo mô hình YOLO11n-seg
    print(f"Đang tải mô hình nền {BASE_MODEL}...")
    model = YOLO(BASE_MODEL)

    # 3. Tiến hành huấn luyện mô hình với cấu hình chuyên biệt
    print("Bắt đầu huấn luyện...")
    results = model.train(
        data=config_path,           # Đường dẫn file yaml
        epochs=100,                 # Tập biển báo nhỏ nên cần train nhiều epochs hơn (100)
        imgsz=320,                  # Độ phân giải ảnh đầu vào (320px giúp nhận diện biển báo nhỏ & tăng FPS)
        batch=4,                    # Giữ batch=4 vừa phải
        workers=0,                  # Đặt workers=0 để tránh crash luồng Docker shared memory
        cache='disk',               # Sử dụng cache disk chống tràn RAM
        device=0,                   # Chỉ định chạy trên GPU số 0
        optimizer="AdamW",          # Thuật toán tối ưu hóa tốt nhất
        lr0=1e-3,                   # Tốc độ học ban đầu
        lrf=1e-5,                   # Tốc độ học tối thiểu
        fliplr=0.0,                 # BẮT BUỘC bằng 0.0 để TẮT tự động lật ngang (cấm biển rẽ trái lật thành rẽ phải)
        flipud=0.0,                 # Tắt lật dọc (biển lật ngược là không hợp lệ)
        mosaic=0.3,                 # Giảm mosaic xuống 0.3 để tránh biến dạng kích thước biển báo quá đà
        copy_paste=0.1,             # Kích hoạt copy-paste nhân tạo đối tượng giúp phong phú dữ liệu
        project="runs/segment",     # Thư mục lưu kết quả train
        name="train_signs",         # Tên thư mục huấn luyện cụ thể
    )

    print("\n[Thành công] Quá trình huấn luyện đã hoàn tất!")
    print(f"Kết quả lưu trữ tại: runs/segment/train_signs/")
    print(f"Trọng số tốt nhất đã sẵn sàng tại: runs/segment/train_signs/weights/best.pt")


if __name__ == "__main__":
    main()
