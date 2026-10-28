# YOLO robustness — gói train cho Vast.ai

Gói này train YOLOv8n detector. OCR model C không được train hay thay trọng số trong gói này.
Không có token/API key/private key trong gói.

## Dữ liệu hiện có

- Source: 9.829 train / 1.229 validation / 1.228 test; 25.364 box train.
- Audit kiểm file ảnh/nhãn, class 0, box hợp lệ, hash pixel khớp manifest, không trùng pixel giữa split; `inventory.json` chứa SHA-256 từng file truyền đi.
- **Mặc định tách 580 ảnh train có nhãn rỗng chưa được xác nhận là không có biển.** Một số mẫu là cảnh xe có vùng biển đã làm mờ. Giữ nguyên file để rà nhãn, không tự sửa nhãn.
- **Train thực tế mặc định: 9.249 ảnh, 25.364 box.** Validation/test giữ nguyên. `train.runtime.txt` và `negative-quarantine.json` ghi danh sách chính xác.
- Các crop áo/logo từ clip người dùng đang nằm ngoài train. Ảnh có nhãn rỗng không được coi là negative đã được bảo đảm chỉ vì file label trống.
- Thêm dữ liệu/negative được rà sau này phải tạo bundle revision mới và chạy A/B trên cùng revision; không đổi dữ liệu giữa lúc resume.

## Augmentation

`corruptions.py` thực hiện Gaussian/Poisson/salt-and-pepper, motion/defocus blur, glare, low light, color cast, JPEG, low-resolution, fog/rain và che nhỏ dạng bóng/bụi. Chỉ chạy trên ảnh train, sau hình học và trước Format/tensor.

- 40% không thêm suy giảm; 45% một nhóm; 15% tối đa hai nhóm, giảm cường độ khi ghép.
- Chọn nhóm: noise 25%, blur 25%, ánh sáng 20%, codec/resolution 20%, weather/occlusion 10%.
- RNG riêng theo DataLoader worker, không làm thay đổi RNG hình học của YOLO.
- Mosaic 2×2 p=0,4, xoay ±5°, translate 5%, scale ±15%; tắt Mosaic 10 epoch cuối.
- Hook vẫn giữ khi Mosaic đóng. Control bỏ suy giảm tùy chỉnh; hai arm đều bỏ Albumentations ngầm, nên môi trường cài thêm package không tự thay đối chứng.
- Val/test không nhận random augmentation. PNG preview cho từng điều kiện nằm trong `previews/`; ảnh batch train cho thấy Mosaic/box thực tế. Counter mỗi worker nằm trong `runs/<name>/quality_counts/`.

## Môi trường

Đã kiểm tra logic với Ultralytics **8.4.183**, Torch **2.14.1** trên local. Vast dùng Python/CUDA PyTorch có sẵn ở `/venv/main`; script tạo `.venv` riêng kế thừa PyTorch, không nâng/hạ package của job OCR đang chạy.

RTX 5060 Ti (Blackwell) cần wheel PyTorch tương thích kiến trúc. Preflight chạy convolution/backward và torchvision NMS trên CUDA thật, kiểm hash toàn bộ dữ liệu trước train. Nếu image chưa có PyTorch, setup dùng cặp `torch==2.9.1`/`torchvision==0.24.1` cu128 từ index chính thức. Không cài NVIDIA driver trong container.

```bash
cd /workspace/vn-plate-yolo
bash custom_alpr/yolo_vast/scripts/setup.sh
```

Setup chạy tests, preflight và preview. Chỉ chạy sau khi GPU được bàn giao; có thể cài dependency vào `.venv` riêng trước đó mà không chạy preflight CUDA.

## Đồng bộ bằng rsync

Từ repo local, thay HOST/PORT bằng thông tin instance đang dùng. Lệnh không thuê/khởi động/dừng/xóa instance:

```bash
../vn-plate-detector/.venv/bin/python -m custom_alpr.yolo_vast.sync push \
  --bundle custom_alpr/outputs/vast-yolo-20261028/bundle \
  --archive custom_alpr/outputs/vast-yolo-20261028/dataset.tar \
  --host HOST --port PORT
```

Rsync dùng `--partial`, có thể chạy lại khi đứt kết nối. Mỗi bước phải thành công; archive phải khớp SHA-256 mới được giải nén. Runtime YAML được ghi lại với đường dẫn máy đích.

Kéo kết quả:

```bash
../vn-plate-detector/.venv/bin/python -m custom_alpr.yolo_vast.sync pull \
  --bundle custom_alpr/outputs/vast-yolo-20261028/bundle --host HOST --port PORT
```

## Smoke và train

```bash
cd /workspace/vn-plate-yolo
bash custom_alpr/yolo_vast/scripts/launch.sh smoke
# Theo dõi: supervisorctl status yolo-robustness-smoke
# Log: logs/yolo-robustness-smoke.log
```

Smoke: 32 ảnh train / 16 validation, một epoch mỗi arm. Đây chỉ là kiểm tra pipeline; không dùng metric/checkpoint smoke để báo cáo chất lượng model mới.

```bash
bash custom_alpr/yolo_vast/scripts/launch.sh pair
```

Full A/B: cùng `weights/initial.pt`, split, seed 42, AdamW LR 0,001, input 640, batch 16, 30 epoch/arm. `moderate-s42` chạy trước để có model ứng viên sớm, `control-s42` chạy sau; cả hai vẫn có cùng khởi tạo/split/seed. Nếu cần batch 8, dùng cùng batch cho hai arm qua `run.py --batch 8`; không thay điều kiện giữa hai arm.

Khởi động trực tiếp một arm:

```bash
.venv/bin/python -m custom_alpr.yolo_vast.run --root "$PWD" --arm moderate --name moderate-s42
```

Một lock ngăn hai job YOLO từ cùng bundle tranh GPU. Thư mục run đã có bị từ chối ghi đè. Supervisor giữ job qua lần đóng SSH; `autorestart=false` tránh tự lặp train khi crash.

## Đợi job OCR trước khi chạy

`wait_for.py` không import Torch và không tạo CUDA context khi chờ. Nó yêu cầu:

1. PID/command của lượt OCR trước đã kết thúc.
2. Log chứa cả `Held-out test (greedy autoregressive):` và `Best checkpoint:`.
3. Không có GPU compute process trong 60 giây liên tục.
4. Setup/tests/preflight thành công; smoke GPU cả A/B thành công.

Sau đó mới chạy pair. Job trước chết mà chưa chấm test xong thì queue dừng báo lỗi, không tự chiếm GPU.

```bash
bash custom_alpr/yolo_vast/scripts/queue_after_ocr.sh \
  --prior-pid 2171 --prior-match run-20261028-05-parseq-d \
  --prior-log /workspace/vn-plate-ocr/training-run-20261028-05-parseq-d.log \
  --stage pair
```

Đặt queue dưới supervisor để tồn tại qua mất SSH:

```bash
bash custom_alpr/yolo_vast/scripts/queue_launch.sh 2171 run-20261028-05-parseq-d \
  /workspace/vn-plate-ocr/training-run-20261028-05-parseq-d.log pair
```

Theo dõi `supervisorctl status yolo-robustness-queue` và `logs/queue.log`. Trạng thái ở `queue_status.json`. PID/path phải đối chiếu lại nếu instance hoặc lượt OCR thay đổi.

## Resume

```bash
.venv/bin/python -m custom_alpr.yolo_vast.run --root "$PWD" \
  --resume "$PWD/runs/moderate-s42"
```

Resume đọc arm từ `experiment.json`, kiểm fingerprint code/data, giữ optimizer/epoch từ `last.pt`. Code hoặc dataset đổi thì từ chối resume. Checkpoint đã hoàn tất/strip optimizer không phải checkpoint để resume; muốn fine-tune tiếp phải mở run mới có mục tiêu riêng. `save_period=1` giữ checkpoint từng epoch.

## Validation và stress

```bash
.venv/bin/python -m custom_alpr.yolo_vast.evaluate --root "$PWD" \
  --weights runs/control-s42/weights/best.pt --output evaluations/control --stress all
.venv/bin/python -m custom_alpr.yolo_vast.evaluate --root "$PWD" \
  --weights runs/moderate-s42/weights/best.pt --output evaluations/moderate --stress all
```

Clean validation dùng toàn bộ validation. Stress lấy cùng 192 ảnh validation, seed 42, mỗi hiệu ứng nhẹ/vừa, giữ nguyên box vì đây là các hiệu ứng image-only. Hình học/Mosaic được kiểm ở train và preview; stress hình học có biến đổi box riêng chưa nằm trong evaluator này.

Test clean chỉ chạy sau khi khóa model/threshold:

```bash
.venv/bin/python -m custom_alpr.yolo_vast.evaluate --root "$PWD" \
  --weights runs/moderate-s42/weights/best.pt --output evaluations/final-test --split test
```

Không chọn model bằng test. Bộ clip người dùng đã xem là development, không phải test độc lập. Chưa có tập hard-negative holdout được gán nhãn đầy đủ; cần bổ sung trước khi khẳng định giảm box sai ngoài thực tế.

## Kết quả và lưu trữ

Mỗi run có `experiment.json`, runtime/pip lock, `args.yaml`, `results.csv`, biểu đồ, counters, best/last/epoch checkpoints và `run_result.json`. Checkpoint vẫn là YOLO thông thường, không cần custom trainer để predict sau này.

Dữ liệu source + archive ~2,8 GB; thêm venv/checkpoint/stress cần khoảng 15–25 GB trống. Instance hiện dùng đĩa 60 GB. `/workspace` trên các máy đã kiểm tra không phải volume bền qua destroy; kéo kết quả về trước khi xóa/recycle.

Tham khảo: https://docs.vast.ai/guides/instances/connect/ssh ; https://pytorch.org/get-started/previous-versions/ ; https://docs.ultralytics.com/guides/yolo-data-augmentation/ .
