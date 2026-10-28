# YOLOv8n + OCR biển số tự huấn luyện

Trạng thái kiến trúc tự xây, dữ liệu, augmentation và phần còn thiếu được ghi tại
[`docs/models/plate-ocr/project-status.md`](docs/models/plate-ocr/project-status.md).

Pipeline hiện tại: **ảnh → YOLOv8n → cắt biển số → CCT-XS v2 → chuỗi ký tự**.

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

### Mạng OCR tự xây (baseline mới)

Repo có thêm `custom_alpr/custom_ocr.py` với hai mạng tự viết. `PlateOCR` là baseline CNN +
Transformer + CTC. `PlateOCR2D` là hướng nhận diện biển một/hai dòng không cần nhãn layout:
residual CNN + BlurPool tạo feature map `256×4×32`; encoder giữ lưới 2D thành 128 token
có vị trí hàng/cột; autoregressive Transformer decoder sinh chuỗi từ transcription đầy
đủ. Alphabet hỗ trợ số, A-Z và `Đ`. Đây là đường custom riêng, chưa thay pipeline CCT
hiện tại.

Với `PlateOCR2D`, ảnh RGB chuẩn hoá vào dạng `[B,3,64,128]`. Dùng `encode_target` để biến
transcription thành BOS + ký tự + EOS; pad batch bằng `PAD_INDEX`; đầu vào teacher forcing
là `target_ids[:, :-1]`; loss là `sequence_loss(logits, target_ids)`. `generate(images)`
giải mã chuỗi trực tiếp. Dataset, collate, train/validation loop và checkpoint nằm trong
`custom_alpr/custom_ocr_data.py` và `custom_alpr/train_custom_ocr.py`.

Đã tải và chuẩn bị bộ Kaggle OCR thành manifest train/validation/test. Xem [quy trình và
thống kê dữ liệu](docs/models/plate-ocr/kaggle-data-preparation.md); tái tạo manifest bằng
`python3 custom_alpr/prepare_ocr_dataset.py` sau khi tải bộ dữ liệu vào thư mục local.

Cài dependency cho mạng tự xây và bắt đầu training:

```bash
python -m pip install -e '.[custom-ocr-train]'
python -m custom_alpr.train_custom_ocr \
  --train custom_alpr/data/ocr/kaggle-v9-prepared/train.csv \
  --val custom_alpr/data/ocr/kaggle-v9-prepared/val.csv \
  --test custom_alpr/data/ocr/kaggle-v9-prepared/test.csv \
  --output custom_alpr/outputs/train-custom-ocr/run-001
```

Vòng lặp chỉ áp dụng motion blur trên ảnh train, chọn checkpoint theo validation loss với
teacher forcing, rồi đo exact-plate accuracy/CER bằng greedy autoregressive một lần trên
test. Để smoke test nhanh một batch mỗi split, thêm các cờ sau và dùng output tạm:

```bash
--epochs 1 --batch-size 2 --workers 0 \
  --max-train-batches 1 --max-val-batches 1 --max-test-batches 1
```

[Xem note chi tiết về query, 4 attention head và Dense của Decoder](docs/models/plate-ocr/layout-free-2d-transformer/decoder-flow.md).

`custom_alpr/augment.py` cung cấp `random_motion_blur(image, rng)` cho crop RGB uint8 ở
nhánh train; mặc định áp dụng xác suất 0.2 với kernel directional 3/5/7 pixel. Không gọi
augmentation này trên validation hay inference. Hàm này được gọi riêng cho train của
`PlateOCR2D`; mạng CCT vẫn giữ luồng hiện hành.

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

### Demo nhận diện video với YOLO augmentation và OCR model E

`custom_alpr/video_demo.py` dùng YOLO augmentation (`moderate-s42/weights/best.pt`)
và OCR E (`run-20261028-08-parseq-e-all/best.pt`). Detector tìm biển trên từng
frame; chỉ box hợp lệ có confidence YOLO cao nhất được chọn, resize RGB về 128×64 như lúc train,
rồi OCR PARSeq đọc chuỗi. Lệnh tạo MP4 có khung/text, CSV theo từng frame và JSON tóm tắt:

```bash
python -m pip install -e '.[custom-ocr-train]'
python -m custom_alpr.video_demo \
  --video /path/to/clip.mp4 \
  --detector custom_alpr/outputs/vast-yolo-20261028/remote-results/moderate-s42/weights/best.pt \
  --ocr-checkpoint custom_alpr/outputs/train-custom-ocr/run-20261028-08-parseq-e-all/best.pt \
  --output custom_alpr/outputs/video-demo/clip-recognized.mp4 \
  --device auto
```

`--device auto` chọn CUDA, MPS, rồi CPU. Ngưỡng detector mặc định 0.25; có thể đổi bằng
`--confidence`. MP4 đầu ra không giữ audio gốc. Khi cài FFmpeg, demo mã hóa H.264 để mở
trên trình duyệt; nếu không, video dùng codec MP4V.

Thêm `--denoise` và/hoặc `--sharpen` để lọc crop trước OCR. Hai cờ mặc định tắt.

## Kiểm tra code

### Web upload ảnh / MP4 và đối chiếu crop

Web chạy trực tiếp YOLO augmentation và OCR model E trên máy đang chạy server.
Cài Python 3.10–3.13, FFmpeg và dependencies, rồi khởi động trong repo này:

```bash
python -m pip install -e '.[web]'
python -m custom_alpr.web_app --port 8766 --device auto
```

Mở `http://127.0.0.1:8766`. Trên máy hiện tại có thể dùng môi trường sẵn có:

```bash
../vn-plate-detector/.venv/bin/python -m custom_alpr.web_app --port 8766
```

Chọn hoặc kéo thả JPG/PNG/WebP/MP4 (tối đa 300 MB), chỉnh ngưỡng YOLO, rồi bấm
**Nhận diện biển số**. Trang hiển thị ảnh/video có khung, các crop gốc được phóng lớn
cạnh text OCR và confidence của từng model. Bấm crop để mở cửa sổ đối chiếu, hoặc bấm
mốc thời gian để xem đúng frame của video. Có thể tải media kết quả và CSV.
Clip mẫu có sẵn qua nút **Dùng clip cao tốc mẫu** nếu artifact từ task 043 vẫn còn.

Trong **Tiền xử lý crop trước OCR**, **Thêm viền crop 5%** được bật sẵn trên web:
mở rộng box mỗi phía 5% (ít nhất một pixel), giới hạn trong ảnh gốc. Box vẽ của YOLO
vẫn giữ nguyên; `crop_box` ghi vùng thực sự cắt. **Nắn thẳng bốn góc** là tùy chọn
thử nghiệm, mặc định tắt. Thuật toán contour chỉ warp khi tìm thấy tứ giác lồi lớn,
có hỗ trợ cạnh đủ rõ và góc phù hợp; không tìm được thì giữ nguyên crop.
Luồng xử lý: YOLO trên frame gốc → crop có viền → nắn thẳng nếu bật và đủ điều kiện
→ resize RGB 128×64 → bilateral giảm nhiễu nếu bật → unsharp mask nếu bật → /255 → OCR E.
JSON ghi yêu cầu nắn thẳng, trạng thái thực sự đã nắn và góc tìm được trong tọa độ crop.
Giảm nhiễu/tăng nét vẫn mặc định tắt. Các thuật toán không phục hồi được chữ đã mất.

Trên mẫu 256 ảnh validation seed 42 với OCR D, bản nắn thẳng có kiểm tra cạnh
đạt 234/256 ảnh đúng, bằng ảnh gốc; chỉ 8 crop được biến đổi. Chi phí riêng bước chuẩn
bị OCR khoảng 0,34 ms/crop trên máy đo, chưa bao gồm inference và lưu preview.
Đây chưa phải bằng chứng tăng độ chính xác; padding cần đánh giá riêng với ảnh toàn cảnh
có nhãn. Báo cáo: `custom_alpr/outputs/web-demo/crop-rectification-validation/metrics.json`.

LIVE và cửa sổ phóng lớn hiển thị **crop gốc** cạnh **ảnh vào OCR**. PNG ảnh đầu vào
khớp pixel mà model nhận trước /255; PNG gốc vẫn được lưu riêng. CSV/JSON lưu cấu hình
giảm nhiễu/tăng nét của từng kết quả. Nút **Làm mượt hiển thị** chỉ đổi cách phóng ảnh,
không thay đầu vào model và không chạy lại OCR.

Kiểm tra nhanh trên cùng 256 ảnh validation (seed 20261028), không chọn mẫu theo lỗi:
gốc 220/256 đúng toàn biển, giảm nhiễu 218/256, tăng nét 220/256, cả hai 218/256.
Kết quả này chưa chứng minh cải thiện độ chính xác trên clip nhiễu; cần đối chiếu với
nhãn thực tế của clip. Xem artifact local `custom_alpr/outputs/web-demo/preprocessing-validation.json`.

Với MP4, web hiển thị luồng MJPEG từ frame **vừa chạy YOLO + OCR xong**, cùng crop và
text mới nhất ngay khi đang xử lý. Clip chạy theo đồng hồ của FPS nguồn; khi máy
không theo kịp, hệ thống bỏ qua frame đã trễ thay vì xếp hàng gây chậm hình. Trang
hiện tốc độ xử lý, độ trễ từng frame và số frame bỏ qua. FPS xử lý là số frame suy
luận chia cho thời gian xử lý; FPS cập nhật là số frame thực tế hiển thị trong một
giây chạy clip. Không đảm bảo OCR mọi frame nếu FPS suy luận thấp hơn FPS nguồn.

Crop OCR giữ độ phân giải gốc. Luồng xem trước và MP4 đầu ra có cạnh dài tối đa
1280 pixel để giảm thời gian vẽ/mã hóa video 4K. Frame bỏ qua được lưu vào MP4
không có annotation. Sau khi kết thúc luồng, server hoàn tất mã hóa H.264 rồi hiện
video có thanh tua và danh sách crop đầy đủ của các frame đã xử lý. URL có `?job=...`
cho phép mở lại tác vụ trong cùng phiên server.

Mỗi ảnh/frame được xử lý chỉ đưa **một box hợp lệ có confidence YOLO cao nhất** sang OCR sau khi
lọc theo ngưỡng đã chọn; nếu không có box qua ngưỡng thì bỏ qua OCR. Vì vậy ảnh có nhiều
xe cũng chỉ đọc một biển. Gallery gom quan sát gần nhau theo vị trí/thời gian.
Trong **Chọn kết quả**, mặc định **Tổng hợp 5 frame · confidence TB**:
trong mỗi track, giữ tối đa 5 quan sát được xử lý gần nhất; chọn chuỗi nhiều phiếu
nhất, hòa phiếu thì chọn confidence trung bình cao hơn. Confidence hiển thị là
trung bình của các quan sát đọc ra chuỗi thắng, không trộn điểm của chuỗi khác.
Dưới 5 quan sát thì dùng số đang có, không trì hoãn video. LIVE và khung/text video
cũng dùng chế độ được chọn khi bắt đầu tác vụ; JSON giữ `latest_raw` và kết quả từng
frame nguyên bản. Crop đại diện thuộc đúng chuỗi thắng; điểm gốc của crop lưu riêng
ở `representative_ocr_confidence`, cạnh `averaged_frames`, `window_frames` và
`mean_ocr_confidence`. Không trộn pixel ảnh giữa các frame.

Có thể đổi sang **OCR E confidence cao nhất** (chọn quan sát cao điểm nhất mỗi track,
kể cả chuỗi ít phiếu) hoặc **Đồng thuận nhiều frame + crop rõ** (bỏ phiếu toàn track,
chọn crop theo 0,65 × chất lượng ảnh + 0,35 × OCR confidence). Điểm chất lượng là
heuristic từ độ rõ, tương phản, ánh sáng và kích thước crop; không phải xác suất đọc đúng.
Gallery gộp track trùng chuỗi, nhưng ở chế độ trung bình giữ nguyên cửa sổ của track
đại diện thay vì trộn confidence giữa các track. Các chế độ đều giữ chuỗi đã quan sát
thực sự; confidence trung bình cũng không đảm bảo đọc đúng. Thẻ hiển thị số lần quan sát, thời gian và
mức đồng thuận. Đây là nhóm chuỗi OCR, không phải bộ đếm xe thật; association theo
vị trí/thời gian có thể nhầm ở cảnh đông hoặc che khuất.

Chọn **Từng frame (bản gốc)** để kiểm tra đầy đủ evidence. CSV tải theo chế độ xem:
CSV đã gom theo cách chọn trung bình/OCR/đồng thuận hiện tại hoặc `results.csv` từng frame. Gallery phân trang 24 thẻ.
Video xuất không có audio. Upload và kết quả được lưu trong
`custom_alpr/outputs/web-demo/<job-id>/`; các tác vụ có `job.json` được khôi phục sau
khi khởi động lại server. Kết quả cũ được gom lại mà không chạy lại model.
Mặc định server chỉ nhận truy cập từ máy này; dùng
`--host 0.0.0.0` nếu muốn cho các máy trong cùng mạng truy cập IP máy chạy server.

Web mặc định confidence 0.40 và input YOLO 640; **Độ phân giải phát hiện** cho phép
thử 320/640/960/1280. CLI có cờ `--imgsz`. Setting được lưu theo tác vụ.
Trên 192 ảnh validation cố định (483 box, seed 20261028, conf 0.25, NMS IoU 0.7,
match IoU 0.5), checkpoint hiện tại cho kết quả:

| Input | Precision | Recall |
| --- | --- | --- |
| 320 | 89.85% | 97.10% |
| 640 | 94.34% | 96.69% |
| 960 | 89.07% | 92.75% |
| 1280 | 77.50% | 83.44% |

Ở input 640/conf 0.40: precision 95.46%, recall 95.86%. Đây là sample validation
của dataset hiện tại, không phải accuracy trên clip người dùng chưa được gán nhãn.
Tăng resolution không tự động giải quyết bắt nhầm logo, quần áo, số trên thân xe.
Chi tiết: `custom_alpr/outputs/web-demo/detector-resolution-validation.json`.

Kế hoạch train YOLO với glare/noise/blur, nhiều biển và hard negatives nằm trong
[task 051](tasks/051-yolo-robustness-plan.md). Chưa khởi chạy lượt train mới.

### Kiểm tra mã nguồn

```bash
python -m pip install pytest ruff
python -m pytest custom_alpr/tests -q
ruff check custom_alpr
ruff format --check custom_alpr
```

Test dùng ảnh tổng hợp và model giả cho logic biên; không thay thế việc train, export
và kiểm thử model thật. Checkpoint model E và YOLO được lưu tại các đường dẫn nêu ở
phần demo video; các trọng số CCT-XS upstream vẫn cần được export riêng nếu muốn dùng
pipeline OCR ONNX hiện có.

## Nguồn

- [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/)
- [FastPlateOCR training](https://ankandrew.github.io/fast-plate-ocr/latest/training/cli/train/)
- [Dataset OCR](https://ankandrew.github.io/fast-plate-ocr/latest/training/dataset/)
- [Export OCR](https://ankandrew.github.io/fast-plate-ocr/latest/training/cli/export/)
- [Config CCT-XS v2 gốc, tag v1.1.0](https://github.com/ankandrew/fast-plate-ocr/blob/v1.1.0/models/cct_xs_v2.yaml)

Config CCT được chép nguyên bản từ upstream v1.1.0, giấy phép MIT được giữ trong
`configs/LICENSE.fast-plate-ocr`. Mã upstream và thư viện được dùng theo giấy phép của
từng dự án.
