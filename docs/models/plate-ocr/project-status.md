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

`PlateOCR2D` đã có lần train đầu trên Kaggle dataset bằng RTX 5060. Run dừng sớm ở epoch 41; checkpoint tốt nhất chọn tại epoch 33 (validation loss 0.1941). Đánh giá greedy autoregressive trên test holdout 1.203 ảnh đạt exact-plate accuracy **82.6%** và CER **6.2%**. Đây là kết quả baseline đầu tiên trên toàn bộ split; chưa có số liệu tách riêng biển một/hai hàng, điều kiện tối/chói/blur, và model chưa tích hợp với YOLO/pipeline dự đoán.

Phân tích ký tự trên cùng test set bằng LCS: **TP 9.221, FP 543, FN 509** trên 9.730 ký tự chuẩn; precision 94.44%, recall 94.77%, F1 94.60%. Ở đây FP là ký tự dự đoán không khớp với ký tự chuẩn, gồm ký tự chèn thừa và ký tự thay thế sai. Đây là FP cấp ký tự OCR, không phải false-positive bounding box của YOLO. Bộ đánh giá tái lập ở [evaluate_custom_ocr.py](../../../custom_alpr/evaluate_custom_ocr.py); report và dự đoán từng ảnh nằm cùng thư mục checkpoint.

Phân bố Levenshtein lỗi trên mỗi biển: 994 biển không lỗi (82.63%), 79 biển có 1 lỗi (6.57%), 38 có 2 lỗi, 26 có 3 lỗi, 16 có 4 lỗi, 20 có 5 lỗi, 13 có 6 lỗi, 7 có 7 lỗi, 7 có 8 lỗi và 3 có 9 lỗi. Trung bình 0.50 ký tự sai/biển; median 0; tối đa 9. [JSON distribution](../../../custom_alpr/outputs/train-custom-ocr/run-20261028-01/per-plate-error-distribution.json) và [per-plate CSV](../../../custom_alpr/outputs/train-custom-ocr/run-20261028-01/per-plate-errors.csv) có chi tiết.

Đánh giá mở rộng cùng checkpoint trên CUDA:

- Ma trận nhầm ký tự dùng edit alignment: 539 substitutions, 51 insertions, 17 deletions. Nhầm lẫn nhiều nhất: `1→0` (19), `0→1` (14), `8→0` (13), `6→5` (12), `8→6` (11).
- Confidence mỗi biển là geometric mean xác suất của các ký tự greedy đã chọn, không tính EOS. Ngưỡng 0.90 giữ lại 1.060/1.203 biển (88.1% coverage) với exact accuracy 92.5%; ngưỡng 0.99 giữ 905 biển (75.2%) với exact accuracy 98.3%. Đây là trade-off đo trên test, không phải ngưỡng đã calibrate cho production.
- Tốc độ model-only: batch 1 median 16.7 ms/biển, p95 18.3 ms; batch 16 throughput 635.7 biển/giây. Đo gồm chuyển tensor và autoregressive decoding, không gồm đọc/resize ảnh.
- Chưa có accuracy riêng biển một/hai hàng vì dataset không gắn nhãn layout đáng tin cậy. File [layout-label-template.csv](../../../custom_alpr/outputs/train-custom-ocr/run-20261028-01/extended-analysis/layout-label-template.csv) có 1.203 ảnh để gán `single` hoặc `double`; evaluator hỗ trợ file này qua `--layout-labels`.

Ma trận, confidence, timing và dự đoán từng ảnh nằm trong `custom_alpr/outputs/train-custom-ocr/run-20261028-01/extended-analysis/` (Git ignored).

Checkpoint, history và log được lưu trên desktop tại `C:\Users\trnghia\vn-plate-ocr\custom_alpr\outputs\train-custom-ocr\run-20261028-01` và được sao chép vào `custom_alpr/outputs/train-custom-ocr/run-20261028-01/` trong workspace này (thư mục bị Git ignore).

Chi tiết 2 tầng Decoder, query/key/value, 4 head, và Dense 192 → 40 nằm ở [decoder-flow.md](layout-free-2d-transformer/decoder-flow.md). Decoder dùng 128 token ảnh làm memory ở từng bước sinh ký tự; 40 đầu ra gồm 37 ký tự và `PAD/BOS/EOS`. Đây là mô tả code, chưa phải kết quả accuracy.

## Data và augmentation

- Kaggle `topkek69/vietnamese-license-plate-ocr` đã tải vào `custom_alpr/data/raw/kaggle-vietnamese-license-plate-ocr/` (thư mục bị Git ignore). Script `custom_alpr/prepare_ocr_dataset.py` tạo manifest split tái lập ở `custom_alpr/data/ocr/kaggle-v9-prepared/`: 9.541 train, 1.187 validation, 1.203 test. Đã loại trùng lặp byte trước khi chia và loại hai nhóm ảnh có transcription xung đột. Cột `Type` chỉ dùng để giữ phân bố nguồn, không phải nhãn số dòng. Chi tiết ở [kaggle-data-preparation.md](kaggle-data-preparation.md).
- Mẫu review 12 ảnh và manifest nằm ở `docs/models/plate-ocr/dataset-review/kaggle-samples/`.
- Contact sheet HSV S/V và Hue nằm ở `docs/models/plate-ocr/dataset-review/`. Đây là ảnh minh họa để xem, không phải data train. Thay đổi Hue mạnh có thể tạo màu nền biển không thực tế.
- `custom_alpr/augment.py` có `motion_blur` và `random_motion_blur`: augmentation directional, mặc định xác suất 0.2 và kernel 3/5/7 pixel. Dataset mới chỉ gọi helper này ở split train.
- Perspective warp/rotation augmentation đã được đề xuất nhưng chưa cài đặt.

## Video nhiều frame

Ý tưởng đã thống nhất: phát hiện và theo dõi cùng biển qua các frame, chọn crop rõ hoặc gộp nhiều lần đọc. Hiện repo xử lý ảnh tĩnh; tracker, chọn frame, voting theo track và video inference chưa được triển khai. Nhiều crop có thể xếp thành batch tensor `B×3×64×128`; đây không phải ghép ảnh thành mosaic 2×2. Model mới nhận batch, nhưng pipeline hiện tại chưa gọi nó theo cách này.

## Việc còn lại

1. Đo exact-plate accuracy/CER riêng cho biển một dòng, hai dòng, tối, chói và blur; cải thiện dữ liệu/hyperparameter dựa trên lỗi.
2. Viết tiền xử lý/adapter cho crop từ YOLO và tích hợp inference theo batch.
3. Sau khi ảnh tĩnh hoạt động ổn định, mới bổ sung video tracking và chọn/gộp nhiều frame.

## Kiểm tra hiện có

Ngày 2026-10-28, `python3 -m py_compile`, `git diff --check`, Ruff lint/format và cả 14 bài test qua. Một batch smoke train/validation/test chạy thành công trên Apple MPS và RTX CUDA. Full training chạy trên Windows Python 3.12 với PyTorch 2.14.1+cu130; checkpoint được tải về workspace, SHA-256 khớp với bản gốc trên desktop và được load/inference kiểm tra thành công.
