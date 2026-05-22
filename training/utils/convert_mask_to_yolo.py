#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
convert_mask_to_yolo.py
-----------------------
Utility script to convert segmentation mask images into YOLO polygon annotation format (.txt).
It supports color filtering (with tolerance) to extract road pixels and automatically splits
the dataset into train/validation sets.
"""

import os
import cv2
import numpy as np
import shutil
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Cấu hình màu sắc mặt đường mặc định từ tài liệu dự án
DEFAULT_ROAD_COLORS = [
    (255, 8, 187),  # Màu hồng đậm BGR/RGB
    (255, 20, 180), # Màu hồng tím BGR/RGB
]
DEFAULT_TOLERANCE = 15


def extract_polygon_from_mask(mask_path, road_colors=None, tolerance=DEFAULT_TOLERANCE):
    """
    Đọc ảnh mask, lọc màu đường (nếu có chỉ định màu) hoặc ngưỡng nhị phân
    để trích xuất các đường bao polygon (contours).
    """
    if road_colors is None:
        road_colors = DEFAULT_ROAD_COLORS

    # Đọc ảnh ở chế độ màu BGR
    img = cv2.imread(mask_path)
    if img is None:
        print(f"[Warning] Không thể đọc ảnh mask: {mask_path}")
        return []

    height, width, _ = img.shape
    total_mask = np.zeros((height, width), dtype=np.uint8)

    # Nếu ảnh mask có màu, thực hiện lọc theo màu chỉ định (BGR)
    has_color_matching = False
    for color in road_colors:
        # Nhớ rằng OpenCV đọc ảnh là BGR
        # Giả sử màu đầu vào là BGR (hoặc RGB, ta thử cả hai để đảm bảo tương thích)
        for order in [color, color[::-1]]:
            lower_bound = np.array([max(0, c - tolerance) for c in order], dtype=np.uint8)
            upper_bound = np.array([min(255, c + tolerance) for c in order], dtype=np.uint8)
            
            mask = cv2.inRange(img, lower_bound, upper_bound)
            if np.sum(mask) > 0:
                total_mask = cv2.bitwise_or(total_mask, mask)
                has_color_matching = True

    # Nếu không phát hiện bất kỳ màu cụ thể nào khớp, ta coi đây là ảnh đen trắng và thực hiện threshold thông thường
    if not has_color_matching:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, total_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Áp dụng một số toán tử hình thái học nhỏ để làm mượt mask đường
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    total_mask = cv2.morphologyEx(total_mask, cv2.MORPH_CLOSE, kernel)

    # Tìm contours bằng OpenCV (chỉ lấy contour bên ngoài)
    contours, _ = cv2.findContours(total_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    polygons = []
    for contour in contours:
        # Loại bỏ các contour nhiễu quá nhỏ (ví dụ diện tích < 50 pixel)
        if cv2.contourArea(contour) < 50:
            continue

        # Đơn giản hóa đa giác (để tránh ghi quá nhiều điểm tọa độ vào txt)
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Chuyển đổi và chuẩn hóa tọa độ về khoảng [0, 1]
        polygon_points = []
        for point in approx:
            x, y = point[0]
            norm_x = x / width
            norm_y = y / height
            polygon_points.append(f"{norm_x:.6f} {norm_y:.6f}")

        # YOLO yêu cầu tối thiểu 3 đỉnh (tức 6 tọa độ) để thành một đa giác phân đoạn
        if len(polygon_points) >= 3:
            polygons.append(" ".join(polygon_points))

    return polygons


def process_dataset(raw_dir, mask_dir, output_dir, split_ratio=0.2, seed=42):
    """
    Chuyển đổi toàn bộ mặt nạ trong thư mục mask_dir thành định dạng nhãn YOLO
    và chia tập dữ liệu thành các tập train/validation tương ứng.
    """
    print(f"Bắt đầu chuyển đổi dữ liệu phân đoạn...")
    print(f"  - Thư mục ảnh gốc: {raw_dir}")
    print(f"  - Thư mục ảnh mask: {mask_dir}")
    print(f"  - Thư mục đầu ra YOLO: {output_dir}")

    # Tạo cấu trúc thư mục YOLO tiêu chuẩn
    dirs = {
        'train_img': os.path.join(output_dir, 'images', 'train'),
        'val_img': os.path.join(output_dir, 'images', 'val'),
        'train_lbl': os.path.join(output_dir, 'labels', 'train'),
        'val_lbl': os.path.join(output_dir, 'labels', 'val')
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Lấy danh sách ảnh mask
    mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not mask_files:
        print("[Error] Không tìm thấy ảnh mask nào trong thư mục chỉ định!")
        return

    # Tìm ảnh gốc tương ứng (có cùng tên file cơ bản)
    valid_pairs = []
    for mask_file in mask_files:
        mask_name, ext = os.path.splitext(mask_file)
        
        # Thử tìm ảnh gốc trùng tên với các đuôi khác nhau
        raw_file = None
        for raw_ext in ['.png', '.jpg', '.jpeg', ext]:
            temp_path = os.path.join(raw_dir, mask_name + raw_ext)
            if os.path.exists(temp_path):
                raw_file = mask_name + raw_ext
                break
                
        if raw_file:
            valid_pairs.append((raw_file, mask_file))
        else:
            print(f"[Warning] Không tìm thấy ảnh gốc cho mask: {mask_file}")

    print(f"Tìm thấy {len(valid_pairs)} cặp (Ảnh gốc, Ảnh Mask) hợp lệ.")
    if not valid_pairs:
        return

    # Chia tập train/val
    train_pairs, val_pairs = train_test_split(valid_pairs, test_size=split_ratio, random_state=seed)
    print(f"  - Số lượng mẫu Train: {len(train_pairs)}")
    print(f"  - Số lượng mẫu Val (Validation): {len(val_pairs)}")

    def convert_and_save(pairs, split_name):
        img_dest_dir = dirs[f'{split_name}_img']
        lbl_dest_dir = dirs[f'{split_name}_lbl']

        for raw_file, mask_file in tqdm(pairs, desc=f"Xử lý tập {split_name}"):
            raw_path = os.path.join(raw_dir, raw_file)
            mask_path = os.path.join(mask_dir, mask_file)

            # 1. Trích xuất đa giác phân đoạn từ ảnh mask
            polygons = extract_polygon_from_mask(mask_path)
            
            # Tên file label cơ sở
            base_name, _ = os.path.splitext(raw_file)
            lbl_file = base_name + '.txt'
            lbl_path = os.path.join(lbl_dest_dir, lbl_file)

            # 2. Ghi nhãn YOLO (.txt)
            # Nếu không tìm thấy polygon nào, ta vẫn ghi file rỗng để YOLO học background (Empty Label)
            with open(lbl_path, 'w', encoding='utf-8') as f:
                for poly in polygons:
                    # Lớp mặc định cho đường là 0
                    f.write(f"0 {poly}\n")

            # 3. Copy ảnh gốc sang thư mục YOLO tương ứng
            shutil.copy(raw_path, os.path.join(img_dest_dir, raw_file))

    convert_and_save(train_pairs, 'train')
    convert_and_save(val_pairs, 'val')

    # Sinh file config yaml mẫu
    yaml_path = os.path.join(output_dir, 'road_seg.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n")
        f.write("  0: road\n")

    print(f"\n[Thành công] Đã hoàn thành sinh dataset phân đoạn làn đường tại: {output_dir}")
    print(f"File cấu hình dataset được tạo tại: {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert segmentation masks to YOLO format polygon coordinates.")
    parser.add_argument("--raw_dir", type=str, default="/workspace/og/raw", help="Path to raw image directory.")
    parser.add_argument("--mask_dir", type=str, default="/workspace/og/seg", help="Path to mask image directory.")
    parser.add_argument("--output_dir", type=str, default="/workspace/unet", help="Path to save the generated YOLO dataset.")
    parser.add_argument("--split", type=float, default=0.2, help="Validation split ratio (default: 0.2).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data split.")
    
    args = parser.parse_args()
    
    # Thực thi chạy nếu các thư mục tồn tại (hoặc báo lỗi chi tiết)
    if not os.path.exists(args.raw_dir) or not os.path.exists(args.mask_dir):
        print("[Error] Thư mục đầu vào không tồn tại! Vui lòng kiểm tra lại '--raw_dir' và '--mask_dir'.")
        print(f"  - raw_dir hiện tại: {args.raw_dir}")
        print(f"  - mask_dir hiện tại: {args.mask_dir}")
        print("\nBạn có thể chạy kiểm tra thử trên máy tính của mình bằng cách truyền tham số thủ công.")
    else:
        process_dataset(args.raw_dir, args.mask_dir, args.output_dir, args.split, args.seed)
