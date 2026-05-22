# 📖 Hướng dẫn Huấn luyện Mô hình YOLO Phân đoạn (YOLO Segmentation Training Guide)
> **Tài liệu hướng dẫn chi tiết từ A-Z cách tiền xử lý dữ liệu, thiết lập cấu hình và huấn luyện các mô hình YOLO11n-seg cho cả hai bài toán: Phân đoạn làn đường (Road Segmentation) và Phân đoạn biển báo giao thông (Traffic Sign Segmentation).**

---

## 🛠️ 1. Môi trường & Thư viện Cần thiết (Environment Setup)

Trước khi bắt đầu, hãy đảm bảo máy tính hoặc container của bạn đã cài đặt đầy đủ các thành phần sau:

### Yêu cầu hệ thống:
* **Hệ điều hành**: Windows 10/11 hoặc Ubuntu 20.04+ (Khuyên dùng Ubuntu để tối ưu hiệu suất GPU và Docker).
* **Python**: Phiên bản từ `3.8` đến `3.10`.
* **CUDA & cuDNN**: Đã được cài đặt tương thích để kích hoạt tăng tốc phần cứng GPU NVIDIA.

### Cài đặt thư viện Python:
Chạy lệnh sau trong Terminal/Command Prompt để cài đặt các thư viện lõi:
```bash
pip install ultralytics opencv-python numpy torch torchvision tqdm matplotlib albumentations scikit-learn
```

### Thiết lập Docker (Nếu chạy trong container):
Để kết nối Docker với môi trường đồ họa và sử dụng GPU, khởi chạy container bằng lệnh:
```bash
docker run --name it-car -it -p 11000:11000 --network="host" -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --shm-size=8G --gpus all quocle28/it_car_2023:v1
```
> [!IMPORTANT]
> Hãy luôn thêm tham số `--shm-size=8G` khi chạy lệnh `docker run`. Mặc định Docker chỉ cấp phát 64MB bộ nhớ dùng chung (`/dev/shm`), điều này sẽ trực tiếp gây lỗi crash luồng dữ liệu (DataLoader) khi huấn luyện mô hình sâu.

---

## 🛣️ 2. Huấn luyện Mô hình Phân đoạn Làn đường (Road Segmentation)

Mục tiêu là huấn luyện mô hình phân đoạn chính xác vùng mặt đường từ camera trước của xe.

### A. Cấu trúc Thư mục Dữ liệu
Dữ liệu nhãn của YOLO phân đoạn yêu cầu định dạng **Polygon** (tọa độ các đỉnh đa giác normalized). Cấu trúc thư mục sau khi chuyển đổi:

```text
/workspace/unet/
├── images/
│   ├── train/               # Ảnh RGB gốc dùng để train (ví dụ: png/jpg)
│   └── val/                 # Ảnh RGB dùng để kiểm thử (validation)
├── labels/
│   ├── train/               # Các file nhãn .txt chứa tọa độ đa giác đa đỉnh
│   └── val/                 # Các file nhãn .txt của tập validation
└── road_seg.yaml            # File cấu hình tập dữ liệu YOLO
```

### B. File Cấu hình `road_seg.yaml`
Tạo file `/workspace/unet/road_seg.yaml` với nội dung sau:
```yaml
path: /workspace/unet        # Đường dẫn gốc tới thư mục dataset

train: images/train          # Đường dẫn tương đối tới ảnh train
val: images/val              # Đường dẫn tương đối tới ảnh val

names:
  0: road                    # ID lớp 0 tương ứng với nhãn "road"
```

### C. Quy trình Chuyển đổi Mask PNG sang Đa giác YOLO (Polygon Label)
Nếu bạn có sẵn ảnh mặt nạ phân đoạn dạng màu (ví dụ: ảnh đen trắng hoặc ảnh chứa dải màu mặt đường cụ thể), bạn cần convert nó sang tọa độ đa giác YOLO.

#### Thuật toán trích xuất Contours bằng OpenCV:
1. Đọc ảnh Mask bằng OpenCV.
2. Lọc dải màu của đường (ví dụ màu hồng `(255, 8, 187)` hoặc `(255, 20, 180)` với ngưỡng sai số Tolerance $\pm 15$).
3. Tìm các đường bao bằng hàm `cv2.findContours()`.
4. Chuẩn hóa (Normalize) tọa độ các đỉnh về khoảng `[0, 1]` bằng cách chia cho chiều rộng/chiều cao của ảnh.
5. Ghi vào file `.txt` tương ứng theo định dạng:
   `<class_id> x1 y1 x2 y2 x3 y3 ...`

> [!WARNING]
> **Lỗi "segment dataset incorrectly formatted"**: Lỗi này xảy ra khi trong tập dữ liệu nhãn của bạn có sự trộn lẫn giữa nhãn Bounding Box (chỉ có 4 tọa độ: `x_center y_center width height`) và nhãn Polygon (có nhiều hơn 4 tọa độ đại diện cho các đỉnh). YOLO Segmentation yêu cầu **100% tất cả các dòng trong các file nhãn phải ở dạng Polygon (tối thiểu 3 đỉnh tức 6 tọa độ x y)**. Nếu có file nhãn nào chỉ có 4 tọa độ, quá trình train sẽ lập tức báo lỗi và dừng lại.

### D. Kịch bản Huấn luyện An toàn (Tránh Crash Bộ nhớ)
Khi huấn luyện mô hình bằng GPU trên Docker, lỗi cạn kiệt phân vùng `/dev/shm` hoặc RAM có thể làm crash luồng tải dữ liệu (`RuntimeError: DataLoader worker exited unexpectedly` hoặc `Pin memory thread exited`). 

Để quá trình chạy ổn định nhất, hãy sử dụng cấu hình huấn luyện an toàn dưới đây:

```python
# training/train_road.py
import os
from ultralytics import YOLO

# Trọng số nền phân đoạn YOLO11
BASE_MODEL = "yolo11n-seg.pt"

# Đường dẫn mặc định trong Docker và tương đối cục bộ
DOCKER_CONFIG = "/workspace/unet/road_seg.yaml"
LOCAL_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "road_seg.yaml")

def main():
    # Tự động xác định đường dẫn cấu hình phù hợp
    if os.path.exists(DOCKER_CONFIG):
        config_path = DOCKER_CONFIG
    elif os.path.exists(LOCAL_CONFIG):
        config_path = LOCAL_CONFIG
    else:
        config_path = "configs/road_seg.yaml"

    # Khởi tạo mô hình YOLO11n-seg pre-trained
    model = YOLO(BASE_MODEL)

    # Thực hiện huấn luyện
    results = model.train(
        data=config_path,
        epochs=50,                  # Số lượng vòng lặp huấn luyện khuyến nghị (50-100)
        imgsz=640,                  # Kích thước ảnh đầu vào (sử dụng 640 để đạt độ nét cao)
        batch=2,                    # Giảm batch size xuống 2-4 để kiểm soát dung lượng RAM/VRAM
        workers=0,                  # Đặt workers=0 để tắt đa luồng DataLoader (ngăn crash phân vùng Docker)
        cache='disk',               # Sử dụng cache trên ổ cứng thay vì RAM ('ram') để tránh cạn kiệt RAM hệ thống
        device=0,                   # Sử dụng GPU thứ 0
        patience=15,                # Dừng sớm nếu chỉ số mAP không cải thiện sau 15 epochs
        optimizer="AdamW",          # Sử dụng thuật toán tối ưu AdamW
        lr0=1e-3,                   # Tốc độ học ban đầu
        lrf=1e-5,                   # Tốc độ học tối thiểu ở epoch cuối cùng
        project="runs/segment",     # Thư mục lưu kết quả train
        name="train_road",          # Tên thư mục huấn luyện cụ thể
    )

if __name__ == "__main__":
    main()
```

---

## 🚦 3. Huấn luyện Mô hình Phân đoạn Biển báo (Traffic Sign Segmentation)

Mục tiêu là phát hiện và phân đoạn 5 loại biển báo giao thông trong mô phỏng Unity.

### A. Quy trình Sinh nhãn Tự động bằng SAM & Circularity Filtering
Do dữ liệu biển báo thu thập từ môi trường Unity dạng ảnh RGB thuần túy, ta ứng dụng công cụ phân đoạn tự động **Segment Anything Model (SAM)** của Meta kết hợp thuật toán lọc hình học để tự động tạo nhãn chất lượng cao:

1. **Khởi chạy SAM**: Dùng checkpoint `sam_vit_b_01ec64.pth` kết hợp dải điểm kiểm thử (points_per_side) để quét và dự đoán toàn bộ các mặt nạ (masks) trong ảnh.
2. **Lọc Hình Học Đường Tròn (Circularity Detection)**: Biển báo giao thông của chúng ta có hình tròn. Ta tính toán hệ số tròn của từng mặt nạ thu được:
   $$Circularity = \frac{4\pi \times Area}{Perimeter^2}$$
   - Nếu $Circularity \approx 1.0$: Mặt nạ là hình tròn hoàn hảo.
   - Nếu $Circularity < 0.8$: Loại bỏ vì đây là các vật thể không phải biển báo (nền đường, xe...).
3. **Phân loại lớp tự động**: Nhãn lớp được gán trực tiếp dựa trên dải chỉ số ảnh thô được chụp theo kịch bản:
   * Ảnh `0` - `140`: Lớp `0` (`di_thang`)
   * Ảnh `141` - `250`: Lớp `1` (`re_trai`)
   * Ảnh `251` - `350`: Lớp `2` (`re_phai`)
   * Ảnh `351` - `443`: Lớp `3` (`cam_re_trai`)
   * Ảnh `444` - `530`: Lớp `4` (`cam_re_phai`)

### B. Thiết lập Dataset & Nhãn Rỗng (Empty Labels)
Tạo cấu trúc thư mục `/workspace/signtrain/` tương tự như phân đoạn làn đường.

> [!TIP]
> **Kỹ thuật Nhãn Rỗng (Empty Labels - Background Images)**: 
> Trong thực tế chạy mô phỏng Unity, xe sẽ liên tục đi qua các đoạn đường không hề có biển báo. Nếu tập dữ liệu của bạn chỉ toàn ảnh chứa biển báo, YOLO sẽ rất dễ bị "ảo giác" (nhận diện nhầm các vật thể bên đường thành biển báo - False Positives).
> **Giải pháp**: Đưa thêm các bức ảnh phong cảnh trống không có biển báo vào tập train, đồng thời tạo ra một file nhãn `.txt` hoàn toàn rỗng (0 byte) trùng tên với ảnh. Việc này giúp YOLO học được bối cảnh nền và giảm thiểu tối đa tỷ lệ nhận diện sai lệch.

Tạo file cấu hình `/workspace/signtrain/traffic_sign_seg.yaml`:
```yaml
path: /workspace/signtrain

train: images/train
val: images/val

names:
  0: di_thang
  1: re_trai
  2: re_phai
  3: cam_re_trai
  4: cam_re_phai
```

### C. Chiến lược Tăng cường Dữ liệu (Data Augmentation) bằng Albumentations
Vì ảnh mô phỏng Unity có đặc tính là "quá sạch" và thiếu tính đa dạng môi trường thực tế, ta sử dụng thư viện **Albumentations** để tăng cường dữ liệu trước khi train. Tuy nhiên, đối với bài toán biển báo giao thông, cần cực kỳ lưu ý:

* **Phép biến đổi NÊN DÙNG**:
  - `RandomBrightnessContrast` & `CLAHE`: Thay đổi độ sáng/tương phản để giả lập thời tiết nắng/mưa/tối trong Unity.
  - `GaussNoise` & `MotionBlur`: Giả lập nhiễu hạt của camera hành trình và hiện tượng nhòe ảnh do xe di chuyển tốc độ cao.
  - `ShiftScaleRotate`: Giả lập góc nhìn nghiêng hoặc khoảng cách camera xa gần khác nhau.

* **Phép biến đổi CẦN TRÁNH HOẶC KHÔNG DÙNG**:
  - ❌ `HorizontalFlip` (Lật ngang): Biển báo "Rẽ trái" sau khi lật ngang sẽ biến thành biển báo "Rẽ phải", nhưng nhãn lớp vẫn là "Rẽ trái" $\rightarrow$ Gây nhiễu và phá hỏng khả năng học của mô hình. Hãy luôn đặt `fliplr: 0.0` trong cấu hình YOLO.
  - ❌ `ChannelShuffle` & `RGBShift` mạnh: Biển cấm rẽ có màu đỏ đặc trưng. Nếu đảo kênh màu làm biển đỏ thành biển xanh lá, mô hình sẽ mất khả năng nhận diện dựa vào màu sắc pháp lý đặc trưng của biển báo.

### D. Kịch bản Huấn luyện Biển báo Giao thông
Tạo script `training/train_yolo_signs.py`:
```python
import os
from ultralytics import YOLO

# Trọng số nền phân đoạn YOLO11
BASE_MODEL = "yolo11n-seg.pt"

# Đường dẫn mặc định trong Docker và tương đối cục bộ
DOCKER_CONFIG = "/workspace/signtrain/traffic_sign_seg.yaml"
LOCAL_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "traffic_sign_seg.yaml")

def main():
    # Tự động xác định đường dẫn cấu hình phù hợp
    if os.path.exists(DOCKER_CONFIG):
        config_path = DOCKER_CONFIG
    elif os.path.exists(LOCAL_CONFIG):
        config_path = LOCAL_CONFIG
    else:
        config_path = "configs/traffic_sign_seg.yaml"

    # Khởi tạo mô hình YOLO11n-seg
    model = YOLO(BASE_MODEL)

    # Huấn luyện mô hình phát hiện biển báo
    results = model.train(
        data=config_path,
        epochs=100,                 # Tập biển báo nhỏ nên cần train nhiều epochs hơn (100)
        imgsz=320,                  # Giảm kích thước ảnh xuống 320 giúp nhận diện biển báo nhỏ tốt hơn và tăng FPS thực tế
        batch=4,
        workers=0,
        device=0,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=1e-5,
        fliplr=0.0,                 # BẮT BUỘC đặt bằng 0.0 để tắt tự động lật ngang của YOLO
        mosaic=0.3,                 # Giảm hệ số mosaic xuống 0.3 để tránh làm nhỏ hình ảnh biển báo quá mức
        copy_paste=0.1,             # Kích hoạt copy-paste đối tượng giúp tăng lượng mẫu
        project="runs/segment",
        name="train_signs",
    )

if __name__ == "__main__":
    main()
```

---

## 🚀 4. Xuất Mô hình ONNX & Triển khai thời gian thực (ONNX Export)

Sau khi quá trình huấn luyện hoàn tất, các file trọng số tốt nhất sẽ được lưu tại thư mục `runs/segment/train*/weights/best.pt`. Để đưa vào ứng dụng thực tế chạy thời gian thực hoặc tích hợp vào plugin **Unity Barracuda**, ta cần xuất mô hình sang định dạng **ONNX**:

Chạy script Python `training/export_onnx.py` để xuất mô hình tối ưu nhất:
```python
import os
import argparse
from ultralytics import YOLO

def export_model(weights_path, imgsz=320):
    if not os.path.exists(weights_path):
        print(f"[Error] Không tìm thấy file trọng số tại: {weights_path}")
        return

    print("Đang nạp mô hình...")
    model = YOLO(weights_path)

    # Export sang ONNX với định dạng tối ưu hóa
    print("Đang tiến hành export sang ONNX (simplify=True, opset=13)...")
    model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,      # Tối giản đồ thị tính toán của mô hình (onnx-simplifier)
        opset=13            # Phiên bản opset tương thích cao nhất với Unity Barracuda
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch YOLO segmentation model to ONNX format.")
    parser.add_argument("--weights", type=str, default="/workspace/runs/segment/train_signs/weights/best.pt", 
                        help="Path to weights file (.pt).")
    parser.add_argument("--imgsz", type=int, default=320, 
                        help="Image size used for export.")
    
    args = parser.parse_args()
    weights_path = args.weights
    
    if not os.path.exists(weights_path):
        # Tự động tìm kiếm trọng số dự phòng dựa theo cấu trúc dự án
        project_road_model = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "../Road_Seg_Model/modelYolo/weights/best.pt"
        ))
        if os.path.exists(project_road_model):
            print(f"[Info] Tự động chọn mô hình làn đường của dự án: {project_road_model}")
            weights_path = project_road_model
        elif os.path.exists("best.pt"):
            weights_path = "best.pt"

    export_model(weights_path, args.imgsz)
```
Sau khi xuất, bạn sẽ nhận được file `best.onnx` sẵn sàng tích hợp trực tiếp vào dự án Unity của mình!

---
> [!NOTE]
> Mọi thắc mắc hoặc lỗi phát sinh trong quá trình huấn luyện, vui lòng đối chiếu với mã nguồn mẫu trong file `maycay.py` hoặc các log huấn luyện trước đó tại thư mục `runs/`.
