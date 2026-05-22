#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
export_onnx.py
--------------
Utility script to export a PyTorch YOLO11/v8 weights file (.pt) to optimized ONNX format.
It applies onnx-simplifier (simplify=True) and sets opset=13 to ensure maximum
compatibility with Unity Barracuda runtime and other embedded execution environments.
"""

import os
import argparse
from ultralytics import YOLO


def export_model(weights_path, imgsz=320):
    """Nạp trọng số .pt và chuyển đổi sang định dạng .onnx"""
    if not os.path.exists(weights_path):
        print(f"[Error] Không tìm thấy file trọng số tại: {weights_path}")
        return

    print("=========================================================================")
    print("🚀 Đang khởi chạy chương trình xuất bản mô hình YOLO sang định dạng ONNX 🚀")
    print(f"  - Trọng số đầu vào: {weights_path}")
    print(f"  - Kích thước ảnh đầu vào (imgsz): {imgsz}")
    print("=========================================================================")

    # 1. Nạp mô hình YOLO từ file trọng số .pt
    print("Đang nạp mô hình...")
    model = YOLO(weights_path)

    # 2. Thực hiện xuất mô hình sang định dạng ONNX
    print("Đang tiến hành export sang ONNX (simplify=True, opset=13)...")
    try:
        onnx_file_path = model.export(
            format="onnx",
            imgsz=imgsz,
            simplify=True,      # Tối giản đồ thị tính toán của mạng giúp suy diễn nhanh hơn
            opset=13            # Đặt opset=13 đảm bảo độ tương thích tối đa với Unity Barracuda
        )
        print("\n[Thành công] Xuất bản mô hình hoàn tất!")
        print(f"File mô hình ONNX mới đã được lưu trữ tại: {onnx_file_path}")
        print("Trọng số này đã sẵn sàng để kéo trực tiếp vào thư mục Assets của dự án Unity Simulator!")
    except Exception as e:
        print(f"\n[Lỗi] Có lỗi xảy ra trong quá trình export: {e}")
        print("Đảm bảo bạn đã cài đặt gói 'onnx' và 'onnx-simplifier' thông qua lệnh: pip install onnx onnxsim")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch YOLO segmentation model to optimized ONNX format.")
    parser.add_argument("--weights", type=str, default="/workspace/runs/segment/train_signs/weights/best.pt", 
                        help="Path to the trained PyTorch weights file (.pt).")
    parser.add_argument("--imgsz", type=int, default=320, 
                        help="Image size used for export (default: 320 to match traffic sign config).")
    
    args = parser.parse_args()
    
    # Tự động tìm kiếm trọng số dự phòng nếu không tìm thấy đường dẫn chỉ định
    weights_path = args.weights
    if not os.path.exists(weights_path):
        # Xác định trọng số mặc định của mô hình làn đường dựa theo cấu trúc thư mục dự án
        project_road_model = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "../Road_Seg_Model/modelYolo/weights/best.pt"
        ))
        
        if os.path.exists(project_road_model):
            print(f"[Info] Tự động chuyển sang trọng số làn đường của dự án: {project_road_model}")
            weights_path = project_road_model
        elif os.path.exists("best.pt"):
            weights_path = "best.pt"

    export_model(weights_path, args.imgsz)
