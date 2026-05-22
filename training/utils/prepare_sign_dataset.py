#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
prepare_sign_dataset.py
-----------------------
Utility script to automatically build a custom Traffic Sign Segmentation dataset.
It uses Segment Anything Model (SAM) for mask generation, filters masks using
circularity detection (formula: 4 * pi * area / perimeter^2 >= 0.8), automatically
assigns classes based on image index ranges, handles background learning with empty
labels, and performs train/val dataset splits.

It includes a robust OpenCV-based fallback in case SAM library or checkpoint is missing.
"""

import os
import cv2
import numpy as np
import shutil
import re
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Định nghĩa các lớp và khoảng chỉ số ảnh tương ứng
CLASS_MAPPING = [
    {"range": range(0, 141), "id": 0, "name": "di_thang"},
    {"range": range(141, 251), "id": 1, "name": "re_trai"},
    {"range": range(251, 351), "id": 2, "name": "re_phai"},
    {"range": range(351, 444), "id": 3, "name": "cam_re_trai"},
    {"range": range(444, 531), "id": 4, "name": "cam_re_phai"},
]


def get_class_by_index(img_index):
    """Xác định Class ID dựa vào số thứ tự của ảnh."""
    for item in CLASS_MAPPING:
        if img_index in item["range"]:
            return item["id"]
    return -1  # Không thuộc nhãn nào (hợp lệ làm background empty label)


def extract_index_from_filename(filename):
    """Trích xuất số nguyên từ tên tệp tin (ví dụ: '0123.png' -> 123)."""
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return -1


def calculate_circularity(contour):
    """Tính toán độ tròn hệ số (Circularity): 4 * pi * Area / Perimeter^2"""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return circularity


def get_mask_opencv_fallback(img):
    """
    Hàm fallback khi không có SAM. Sử dụng lọc màu đỏ/xanh đặc trưng của biển báo
    kết hợp dò đường tròn để tìm mặt nạ biển báo tốt nhất.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Làm mờ để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Thử kỹ thuật phân ngưỡng Otsu kết hợp Canny Edge
    edged = cv2.Canny(blurred, 30, 150)
    
    # Phình to biên dạng để nối các nét đứt
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(edged, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_contour = None
    max_circ = 0.0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:  # Bỏ qua các đối tượng quá nhỏ
            continue
            
        circ = calculate_circularity(contour)
        if circ >= 0.8 and circ > max_circ:
            max_circ = circ
            best_contour = contour
            
    if best_contour is not None:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [best_contour], -1, 255, -1)
        return mask, best_contour
        
    return None, None


def run_sam_segmentation(img, sam_checkpoint, model_type="vit_b"):
    """Sinh mặt nạ bằng Segment Anything Model (SAM)"""
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        import torch
    except ImportError:
        print("[Info] Không tìm thấy thư viện 'segment_anything'. Sử dụng OpenCV Fallback...")
        return None, None

    if not os.path.exists(sam_checkpoint):
        print(f"[Info] Không thấy checkpoint SAM tại {sam_checkpoint}. Sử dụng OpenCV Fallback...")
        return None, None

    # Load mô hình SAM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    # Cấu hình sinh mặt nạ tự động
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masks_dict = mask_generator.generate(rgb_img)
    
    best_mask = None
    best_contour = None
    max_circ = 0.0
    
    # Quét qua các mặt nạ để tìm mặt nạ tròn trịa nhất đại diện cho biển báo
    for item in masks_dict:
        m = item['segmentation'].astype(np.uint8) * 255
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            circ = calculate_circularity(contour)
            # Biển báo hình tròn có Circularity >= 0.8
            if circ >= 0.8 and circ > max_circ:
                max_circ = circ
                best_mask = m
                best_contour = contour
                
    return best_mask, best_contour


def prepare_dataset(raw_dir, output_mask_dir, train_dest_dir, sam_checkpoint, split_ratio=0.2, seed=42):
    """Tiến hành tiền xử lý, gán nhãn tự động và phân loại dataset biển báo."""
    print("Bắt đầu quy trình tiền xử lý và gán nhãn tập dữ liệu biển báo...")
    
    os.makedirs(output_mask_dir, exist_ok=True)
    
    dirs = {
        'train_img': os.path.join(train_dest_dir, 'images', 'train'),
        'val_img': os.path.join(train_dest_dir, 'images', 'val'),
        'train_lbl': os.path.join(train_dest_dir, 'labels', 'train'),
        'val_lbl': os.path.join(train_dest_dir, 'labels', 'val')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    img_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not img_files:
        print(f"[Error] Không tìm thấy hình ảnh nào trong thư mục raw: {raw_dir}")
        return

    print(f"Tổng số ảnh thô tìm thấy: {len(img_files)}")
    
    dataset_records = []

    for img_file in tqdm(img_files, desc="Phân tích ảnh & Tạo nhãn"):
        img_path = os.path.join(raw_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_idx = extract_index_from_filename(img_file)
        class_id = get_class_by_index(img_idx)
        
        # Biến lưu tọa độ YOLO polygon
        polygon_str = ""
        
        # Nếu chỉ số ảnh nằm trong khoảng các lớp biển báo giao thông hợp lệ
        if class_id != -1:
            # Thử dùng SAM trước
            mask, contour = run_sam_segmentation(img, sam_checkpoint)
            
            # Nếu SAM thất bại hoặc không khả dụng, dùng OpenCV Fallback
            if mask is None:
                mask, contour = get_mask_opencv_fallback(img)
                
            if mask is not None and contour is not None:
                # Lưu ảnh mask để phục vụ lưu trữ / đối chứng
                mask_save_path = os.path.join(output_mask_dir, img_file)
                cv2.imwrite(mask_save_path, mask)
                
                # Biến đổi contour sang định dạng polygon YOLO normalized
                height, width = img.shape[:2]
                epsilon = 0.002 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                points = []
                for pt in approx:
                    x, y = pt[0]
                    points.append(f"{x / width:.6f} {y / height:.6f}")
                    
                if len(points) >= 3:
                    polygon_str = " ".join(points)
        
        # Lưu bản ghi để chia train/val
        # Ghi chú: Nếu class_id == -1 hoặc không tìm thấy contour hợp lệ, polygon_str = "" (Empty Label)
        dataset_records.append({
            'img_file': img_file,
            'img_path': img_path,
            'class_id': class_id,
            'polygon': polygon_str
        })

    # Chia tập train/val
    train_records, val_records = train_test_split(dataset_records, test_size=split_ratio, random_state=seed)
    print(f"\nPhân bổ hoàn tất:")
    print(f"  - Tập Train: {len(train_records)} ảnh")
    print(f"  - Tập Val:   {len(val_records)} ảnh")

    def save_split_data(records, split_name):
        img_dir = dirs[f'{split_name}_img']
        lbl_dir = dirs[f'{split_name}_lbl']
        
        for rec in tqdm(records, desc=f"Ghi tập {split_name}"):
            # 1. Sao chép ảnh
            shutil.copy(rec['img_path'], os.path.join(img_dir, rec['img_file']))
            
            # 2. Sinh file label .txt
            base_name, _ = os.path.splitext(rec['img_file'])
            lbl_path = os.path.join(lbl_dir, base_name + '.txt')
            
            with open(lbl_path, 'w', encoding='utf-8') as f:
                # Nếu có đa giác hợp lệ, ghi nhãn phân đoạn
                if rec['polygon'] and rec['class_id'] != -1:
                    f.write(f"{rec['class_id']} {rec['polygon']}\n")
                # Nếu rỗng, file .txt sẽ để trống (YOLO Background learning)
                else:
                    pass

    save_split_data(train_records, 'train')
    save_split_data(val_records, 'val')

    # Viết file cấu hình traffic_sign_seg.yaml vào signtrain
    yaml_path = os.path.join(train_dest_dir, 'traffic_sign_seg.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"path: {os.path.abspath(train_dest_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n")
        for item in CLASS_MAPPING:
            f.write(f"  {item['id']}: {item['name']}\n")

    print(f"\n[Thành công] Đã xây dựng hoàn thành bộ dữ liệu biển báo tại: {train_dest_dir}")
    print(f"Mặt nạ phân đoạn lưu trữ đối chứng tại: {output_mask_dir}")
    print(f"File cấu hình YOLO được tạo tại: {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Custom Traffic Sign Segmentation Dataset using SAM and OpenCV.")
    parser.add_argument("--raw_dir", type=str, default="/workspace/signdata", help="Path to raw traffic sign images from Unity.")
    parser.add_argument("--mask_dir", type=str, default="/workspace/sign_outputs", help="Path to save intermediate segmentation masks.")
    parser.add_argument("--output_dir", type=str, default="/workspace/signtrain", help="Path to save the generated YOLO dataset.")
    parser.add_argument("--sam_checkpoint", type=str, default="sam_vit_b_01ec64.pth", help="Path to SAM vit_b checkpoint file.")
    parser.add_argument("--split", type=float, default=0.2, help="Validation split ratio (default: 0.2).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splitting.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.raw_dir):
        print(f"[Error] Thư mục chứa ảnh thô biển báo không tồn tại: {args.raw_dir}")
        print("Tạo thư mục rỗng phục vụ chứa ảnh thô sau này...")
        os.makedirs(args.raw_dir, exist_ok=True)
    else:
        prepare_dataset(
            raw_dir=args.raw_dir,
            output_mask_dir=args.mask_dir,
            train_dest_dir=args.output_dir,
            sam_checkpoint=args.sam_checkpoint,
            split_ratio=args.split,
            seed=args.seed
        )
