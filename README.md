# YOLOv8n + OCR biển số tự huấn luyện

Pipeline: **ảnh → YOLOv8n → cắt biển số → CCT-XS v2 → chuỗi ký tự**.

YOLOv8n fine-tune từ trọng số COCO cho một lớp `license_plate`. OCR dùng implementation
CCT-XS v2 của `fast-plate-ocr==1.1.0`, khởi tạo mới và train bằng nhãn của bạn; không tải
trọng số OCR global. Đây là project ghép hai model và quy trình train, chưa phải một
implementation CCT viết lại từ đầu. Code chạy độc lập với package `fast_alpr`.

## 1. Môi trường

Chạy tất cả lệnh bên dưới từ thư mục gốc repo `vn-plate-ocr`. Dùng Python 3.11 trên Linux
hoặc WSL2 cho máy NVIDIA. Máy Mac có thể chuẩn bị dữ liệu và chạy inference CPU.
Môi trường Python 3.14 mặc định trên máy hiện tại chưa được kiểm chứng cho training.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r custom_alpr/requirements-train.txt
```

Chỉ inference: cài `custom_alpr/requirements.txt`.
Cho OCR training trên NVIDIA/Linux/WSL2, cài thêm `tensorflow[and-cuda]` và kiểm tra:

```bash
python -m pip install 'tensorflow[and-cuda]'
python -c "import torch; print(torch.cuda.is_available())"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Hai framework phải nhận GPU trước khi train. Phần OCR ONNX inference trong project
hiện dùng CPU; `--device 0` chỉ chọn GPU cho detector. Chưa đo tốc độ hoặc accuracy.

## 2. Chuẩn bị dữ liệu

Input là dataset YOLO có `images/train`, `images/val`, `labels/train`, `labels/val`.
Mỗi dòng nhãn là bounding box YOLO hoặc polygon tọa độ chuẩn hóa. **Tất cả class đầu
vào phải là biển số**: script gộp chúng thành một lớp.

Dữ liệu đã chuẩn bị nằm trong `custom_alpr/data/`. Chỉ chạy bước dưới khi muốn
chuẩn bị một dataset YOLO mới (thay đường dẫn nguồn và chọn output chưa tồn tại):

```bash
python -m custom_alpr prepare \
  --source /path/to/licenseplates \
  --output custom_alpr/data
```

Script giữ split ban đầu, loại ảnh trùng theo pixel (ưu tiên giữ bản train), chuyển
polygon thành bounding box và tạo:

```text
custom_alpr/data/
  detection/dataset.yaml
  detection/images/{train,val}/
  detection/labels/{train,val}/
  ocr/train/images/
  ocr/train/annotations.csv
  ocr/val/images/
  ocr/val/annotations.csv
  report.json
```

Ảnh lỗi/thiếu nhãn hoặc tọa độ vượt ảnh thực sự được ghi vào `report.json` và bỏ qua.
Sai số số thực rất nhỏ được giới hạn vào ảnh. Script từ chối ghi đè output có sẵn.
Kiểm tra report và ảnh crop trước khi sử dụng. Giữ ảnh cùng xe/cùng phiên quay trong
cùng split để tránh rò rỉ dữ liệu; kiểm tra pixel trùng chỉ loại được bản sao giống hệt.
Tập test độc lập cần chuẩn bị riêng trước khi báo cáo kết quả cuối cùng.

**Phải điền `plate_text` thật vào CSV trước khi train OCR.** Dataset detection hiện tại
không có chuỗi ký tự; script để trống để bạn gán nhãn, không tự suy đoán.

```csv
image_path,plate_text
images/example_000.png,30A12345
images/example_001.png,59X112345
```

Đường dẫn ảnh tính từ thư mục chứa CSV. Dùng A-Z và 0-9, bỏ dấu chấm/gạch/khoảng trắng;
biển hai dòng đọc dòng trên rồi dòng dưới. CSV phải giữ cột text dạng chuỗi để không
mất số 0 ở đầu. Xóa mẫu không đọc được khỏi CSV thay vì điền nhãn đoán.

`configs/plate.yaml` đặt RGB 64×128, tối đa 10 ký tự, không region head. Dùng nguyên
crop hai dòng ở baseline này; cần đánh giá riêng biển một/hai dòng trước khi quyết định
tách dòng hay sửa kích thước input. Mọi thay đổi preprocessing phải áp dụng cả train
và inference qua cùng plate config.

## 3. Train detector

```bash
python -m custom_alpr train-detector \
  --data custom_alpr/data/detection/dataset.yaml \
  --device 0 --epochs 100 --batch 16
```

YOLO tự tải `yolov8n.pt` ở lần đầu. Đây là trọng số khởi tạo, chưa nhận diện biển số
được chỉ bằng cách đổi tên class. Trọng số đã fine-tune được lưu tại
`custom_alpr/outputs/train-detector/plates/weights/best.pt` (run lặp có thể tăng hậu tố).
Nếu hết VRAM, giảm batch. Có thể dùng `--device cpu` hoặc `--device mps` trên Mac.

## 4. Train OCR và export

Sau khi hoàn tất nhãn train và validation:

```bash
python -m custom_alpr train-ocr \
  --train custom_alpr/data/ocr/train/annotations.csv \
  --val custom_alpr/data/ocr/val/annotations.csv \
  --epochs 150 --batch 64
```

Lệnh kiểm tra nhãn trống, ký tự, độ dài và ảnh trước khi gọi CLI upstream với
`--validate-dataset error`. OCR dùng Keras backend TensorFlow; không truyền pretrained
OCR weights. Backend chọn GPU theo môi trường TensorFlow, không theo cờ của detector.
Thời gian train và batch phù hợp phải đo trên GPU thật.

Upstream tạo thư mục có timestamp dưới `custom_alpr/outputs/train-ocr/`, chứa
`best.keras`, `last.keras`, model/plate config và training log. Thay `RUN` bằng thư mục đó:

```bash
python -m custom_alpr export-ocr \
  --model custom_alpr/outputs/train-ocr/RUN/best.keras \
  --plate-config custom_alpr/outputs/train-ocr/RUN/plate_config.yaml
```

ONNX được lưu cạnh `.keras`. Giữ settings export mặc định để khớp `LicensePlateRecognizer`.
Luôn chuyển kèm `plate_config.yaml` của chính lần train, nhất là khi sửa alphabet/input.

## 5. Nhận diện và đánh giá

```bash
python -m custom_alpr predict \
  --image /path/to/test_image.png \
  --detector custom_alpr/outputs/train-detector/plates/weights/best.pt \
  --ocr custom_alpr/outputs/train-ocr/RUN/best.onnx \
  --plate-config custom_alpr/outputs/train-ocr/RUN/plate_config.yaml \
  --output custom_alpr/outputs/result.jpg

python -m custom_alpr evaluate-ocr \
  --annotations custom_alpr/data/ocr/val/annotations.csv \
  --ocr custom_alpr/outputs/train-ocr/RUN/best.onnx \
  --plate-config custom_alpr/outputs/train-ocr/RUN/plate_config.yaml
```

`predict` xuất ảnh có khung và JSON gồm box, text và confidence.
`evaluate-ocr` đo exact plate accuracy và character error rate trên crop có nhãn;
đây là đánh giá riêng recognizer, chưa bao gồm lỗi detector. Dùng tập test chưa tham gia
chọn model để báo cáo kết quả; accuracy trên validation chỉ dùng để phát triển model.

## Kiểm tra code

```bash
python -m pip install pytest ruff
python -m pytest custom_alpr/tests -q
ruff check custom_alpr
ruff format --check custom_alpr
```

Test dùng ảnh tổng hợp và model giả cho logic biên; không thay thế việc train, export
và kiểm thử model thật. Project chưa chứa trọng số đã huấn luyện.

## Nguồn

- [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/)
- [FastPlateOCR training](https://ankandrew.github.io/fast-plate-ocr/latest/training/cli/train/)
- [Dataset OCR](https://ankandrew.github.io/fast-plate-ocr/latest/training/dataset/)
- [Export OCR](https://ankandrew.github.io/fast-plate-ocr/latest/training/cli/export/)
- [Config CCT-XS v2 gốc, tag v1.1.0](https://github.com/ankandrew/fast-plate-ocr/blob/v1.1.0/models/cct_xs_v2.yaml)

Config CCT được chép nguyên bản từ upstream v1.1.0, giấy phép MIT được giữ trong
`configs/LICENSE.fast-plate-ocr`. Mã upstream và thư viện được dùng theo giấy phép của
từng dự án.
