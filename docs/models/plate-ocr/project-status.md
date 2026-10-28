# Trạng thái OCR biển số — 2026-10-28

## Pipeline hiện tại

Pipeline đang dùng trong repo vẫn là:

```text
Ảnh → YOLOv8n → crop biển → fast-plate-ocr CCT-XS v2 → chuỗi ký tự
```

Cấu hình CCT hiện tại nhận ảnh RGB `128×64`, `keep_aspect_ratio: false`. OCR crop được gọi riêng trong vòng lặp mỗi bounding box. Đây vẫn là đường train/inference hiện hành; chưa bị thay bởi mạng tự xây.

## Mạng OCR tự xây

`custom_alpr/custom_ocr.py` hiện có hai lớp mô hình:

- `PlateOCR`: CNN + Transformer encoder + CTC, baseline chuỗi ngang.
- `PlateOCR2D`: residual CNN + BlurPool, Transformer encoder giữ feature map 2D, và autoregressive Transformer decoder. Mạng nhận full-plate transcription, không yêu cầu nhãn `single/double` riêng. Alphabet gồm `0–9`, `A–Z`, `Đ`.

Với input mặc định `B×3×64×128`, CNN tạo feature map `B×256×4×32`. `PlateOCR2D` thêm positional embedding theo hàng/cột, encoder xử lý 128 visual tokens và decoder sinh chuỗi. Đầu vào phải là RGB float đã chuẩn hóa; tiền xử lý này chưa được nối thành adapter inference.

**Mạng tự xây mới là kiến trúc chưa train**, chưa có checkpoint và chưa tích hợp với YOLO/pipeline dự đoán.

Chi tiết 2 tầng Decoder, query/key/value, 4 head, và Dense 192 → 40 nằm ở [decoder-flow.md](layout-free-2d-transformer/decoder-flow.md). Decoder dùng 128 token ảnh làm memory ở từng bước sinh ký tự; 40 đầu ra gồm 37 ký tự và `PAD/BOS/EOS`. Đây là mô tả code, chưa phải kết quả accuracy.

## Data và augmentation

- Kaggle `topkek69/vietnamese-license-plate-ocr` đã được kiểm tra nhưng chưa được tải vào repo: 6.643 ảnh crop, 5.547 ảnh tổng hợp và hai CSV transcription. Không có split train/validation/test sẵn; cột `Type` không phải nhãn số dòng. Có ảnh trùng byte và nhãn chứa ký tự `Đ`.
- Mẫu review 12 ảnh và manifest nằm ở `docs/models/plate-ocr/dataset-review/kaggle-samples/`.
- Contact sheet HSV S/V và Hue nằm ở `docs/models/plate-ocr/dataset-review/`. Đây là ảnh minh họa để xem, không phải data train. Thay đổi Hue mạnh có thể tạo màu nền biển không thực tế.
- `custom_alpr/augment.py` có `motion_blur` và `random_motion_blur`: augmentation directional, mặc định xác suất 0.2 và kernel 3/5/7 pixel. Đây là helper, chưa được gọi bởi Dataset hoặc train loop.
- Perspective warp/rotation augmentation đã được đề xuất nhưng chưa cài đặt.

## Video nhiều frame

Ý tưởng đã thống nhất: phát hiện và theo dõi cùng biển qua các frame, chọn crop rõ hoặc gộp nhiều lần đọc. Hiện repo xử lý ảnh tĩnh; tracker, chọn frame, voting theo track và video inference chưa được triển khai. Nhiều crop có thể xếp thành batch tensor `B×3×64×128`; đây không phải ghép ảnh thành mosaic 2×2. Model mới nhận batch, nhưng pipeline hiện tại chưa gọi nó theo cách này.

## Việc còn lại

1. Chuẩn hóa/import dataset, giữ transcription đúng, bổ sung xử lý `Đ`, deduplicate trước khi chia train/validation/test.
2. Viết Dataset/collate và train/validation loop cho `PlateOCR2D`; tích hợp motion-blur augmentation chỉ trên train.
3. Chạy smoke test/overfit nhỏ, sau đó train và đo exact-plate accuracy/CER riêng cho biển một dòng, hai dòng, tối, chói và blur.
4. Viết tiền xử lý/adapter cho crop từ YOLO và tích hợp inference theo batch.
5. Sau khi ảnh tĩnh hoạt động, mới bổ sung video tracking và chọn/gộp nhiều frame.

## Kiểm tra hiện có

Ngày 2026-10-28, `python3 -m py_compile`, `git diff --check`, Ruff lint và Ruff format đều qua. Cả 14 bài test hiện có cũng qua khi chạy với pytest và các dependency cài trong thư mục tạm. Các test này kiểm tra pipeline/data hiện hành, **chưa kiểm tra forward hoặc training của `PlateOCR2D`**. Máy hiện dùng Python 3.14, ngoài phạm vi `<3.14` trong `pyproject.toml`; không có wheel PyTorch phù hợp trong môi trường này nên chưa chạy được smoke test cho mạng tự xây.
