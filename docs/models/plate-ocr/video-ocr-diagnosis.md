# OCR trên video thật: thu dữ liệu, fine-tune và chẩn đoán lỗi

Cập nhật: 2026-10-30.

## Kết luận nhanh

- Model E đọc đúng 95% ảnh Kaggle cao từ 32 px, nhưng chỉ đúng **24%** trên 101 crop video do người xác nhận. Các crop video này cũng lớn: trung vị 58 px, 100/101 crop cao từ 32 px.
- Nét chữ vẫn còn trong ảnh. Lỗi chính là **khung YOLO quá lỏng**: crop chứa nhiều thân xe, đèn hậu, có khi cả một biển thứ hai. Khi kéo về 128×64, chữ bị nhỏ đi và mô hình đọc lẫn.
- Fine-tune trên 95 crop video (E-v1) không làm kém đi trên Kaggle. Trên camera chưa thấy, kết quả chỉ tăng nhẹ và chưa đủ bằng chứng.
- Phase C dùng synthetic crop lỏng có thân xe/đèn hậu, lề 10–60% và nén kiểu video. Model C tăng crop gốc từ 24.8% lên 34.7% exact (101 ảnh), trong khi Kaggle test gần như giữ nguyên: 89.35% → 89.60% (+0.25 điểm %). Sau `refine_box`, E đạt 36.6% và C 40.6%; paired intervals còn cắt 0, nên chưa đủ bằng chứng về lợi ích cộng thêm. `reject_eval` vẫn chỉ đạt điểm precision cao ở coverage khoảng 5%. Chi tiết: `tasks/076-reject-and-char-vote-then-targeted-training.md` và `custom_alpr/outputs/phase-c-076/evaluation/report.json`.
- Giới hạn ký tự theo kiểu biển khi giải mã **không đổi kết quả nào**. Mô hình đã tự học được cấu trúc biển.
- Việc nên làm tiếp:
  1. Cắt lại sát biển trước OCR (`refine_box.py`).
  2. Phase C đã train OCR chịu khung lỏng; lợi ích rõ nhất nằm ở crop gốc chưa refine.

## 1. Thu crop từ video

`custom_alpr/collect_crops.py` chạy YOLO, ByteTrack và OCR trên video YouTube. Mỗi track giữ 3–5 crop đa dạng, có điểm ưu tiên. Kết quả ghi vào `custom_alpr/data/collected/` gồm ba phần:

- `crops/`: ảnh crop;
- `metadata.csv`: có kết quả OCR của model;
- `queue.csv`: hàng đợi gán nhãn, không chứa kết quả OCR.

Đã thu 2236 crop từ 5 video, chia theo 3 camera:

| Camera | Video |
|---|---|
| danang-ptz | wzTiGOlidk4, dkAcXmw2hjA, RjeyplZXDCo |
| saigon-drive | gZWiLrQnb10 |
| hanoi-drive | TgbtWyS30A4 |

Tập train, val và test được chia **theo camera**. Các video của cùng một camera có chung xe đỗ, nên chia theo video sẽ bị lộ dữ liệu giữa các tập.

## 2. Gán nhãn

Quy trình gồm bốn bước.

1. **VLM gán nhãn trước.** Qwen2.5-VL-7B (4-bit, `custom_alpr/vlm_label.py`) đọc cả 2236 crop và ghi 652 crop là `label`.
   - VLM thường bịa chuỗi kiểu `...1234`. Luật đồng thuận coi các chuỗi này là không đọc được.
2. **Codex gán nhãn mù** 762 crop đầu hàng đợi, gồm 150 crop đầu tiên và 612 crop VLM ghi `label`. Codex làm theo `custom_alpr/labeling/GUIDE.md` và không xem kết quả OCR.
3. **Đối chiếu bằng `custom_alpr/consensus.py`.**
   - 249 crop Codex và VLM đọc cùng chuỗi.
   - 190 crop không bên nào đọc ra biển.
   - 323 crop bất đồng, chuyển cho người quyết định.
4. **Người quyết định** 323 crop bất đồng qua `custom_alpr/label_web.py`, một trang web cục bộ có phím tắt và nút "Bỏ, quá nhiễu".
   - Ghi nhãn `label`: 115 crop.
   - Bỏ: 208 crop. Crop bị bỏ được lưu là `unreadable` kèm ghi chú `skip`, nên không vào tập train.

Xuất nhãn bằng `consensus.py --confirmed-only` rồi chạy `ingest_labels.py --test-cameras hanoi-drive`. Chỉ giữ nhãn do người xác nhận, ra **101 crop**:

- train: 95 crop, mỗi crop có hai ảnh (sát biển và có viền);
- test Hà Nội: 6 crop.

14 crop bị loại vì cùng track mà nhãn khác nhau. Trong số này có lỗi gõ: `92A9826` phải là `92A39826`.

## 3. Fine-tune từ model E

**Code.** Ba thay đổi:

- `train_custom_ocr.py` có thêm `--init-checkpoint`.
- `custom_alpr/finetune_mix.py` trộn Kaggle với crop video lặp lại N lần và tạo các fold "bỏ ra một camera".
- `custom_alpr/scripts/finetune_video.sh` chạy train và đánh giá, so với E.

**Cấu hình.** Giữ cờ augment của E, learning rate 5e-5, tối đa 12 epoch, chọn epoch theo greedy exact trên val Kaggle. Chạy trên Vast.ai, máy RTX 5060 Ti.

Kết quả trên camera chưa thấy. Số liệu là số ảnh đúng, gồm cả ảnh sát biển và ảnh có viền:

| Bản | Dữ liệu | Train Sài Gòn → test Đà Nẵng (74) | Train Đà Nẵng → test Sài Gòn (116) | Test Kaggle |
|---|---|---|---|---|
| E | — | 16 | 28 | 89.4% |
| v1 | Kaggle đầy đủ + video ×15 | 18 (+2/−0) | 33 (+9/−4) | 89.4–89.7% |
| v2 | Chỉ ảnh Kaggle thật + video ×15 | 18 (+2/−0) | 27 (+6/−7) | 88.9–90.0% |

Cách đọc bảng:

- v1 sửa đúng 11 ảnh và làm sai 4 ảnh. Kết quả này chưa có ý nghĩa thống kê: khả năng do ngẫu nhiên khoảng 12%.
- Chọn epoch theo val Kaggle gần như không phân biệt được các epoch, vì val chỉ dao động 0.88–0.89. Nhiều fold dừng sớm và giữ epoch 1. Muốn chọn epoch có ý nghĩa thì cần một tập val gồm crop video.

**E-v1** là bản `all` của v1, nằm ở `custom_alpr/outputs/train-custom-ocr/run-20261030-10-parseq-e-v1-video/best.pt`. Đây là OCR mặc định trong `video_demo.py`.

- Độ chính xác theo từng chữ cái trên test Kaggle giữ như E: 0.949 so với 0.952.
- Chữ yếu nhất ở cả hai bản là N, khoảng 75%.
- E đã được train cùng `synthetic-v1`, bộ biển nhân tạo dùng để cân bằng chữ hiếm (khoảng 400 biển mỗi chữ).

## 4. Chẩn đoán lỗi

Dùng `custom_alpr/confusion_report.py`. Model E đọc test Kaggle và 101 crop video đã xác nhận (chỉ ảnh sát biển, manifest `collected-v1/confirmed-tight.csv`).

| Tập | < 16 px | 16–24 px | 24–32 px | ≥ 32 px |
|---|---|---|---|---|
| Kaggle test | 49% | 80% | 89% | **95%** |
| Video đã xác nhận | — | — | 1/1 | **24%** (100 crop) |

- Các cặp hay nhầm trên video là `7→2`, `N→M`, `2→0`, `9→3`, `8→6`. Chúng khác các cặp hay nhầm trên Kaggle, nên đây là lỗi do khác miền dữ liệu, không phải do không thấy nét chữ.
- Ảnh có viền 5% đọc **kém hơn** ảnh sát biển: 19.8% so với 24.8%. Có thêm ngữ cảnh quanh biển là E đọc kém đi.
- Ví dụ crop đọc sai:
  - `30H33154`: crop cao 113 px nhưng biển chỉ chiếm khoảng 1/4.
  - `51D62423`: crop có hai biển.
- Đo trước đây với `refine_box.py` (cắt lại sát biển bằng YOLO lần hai, ngữ cảnh 0.5, cỡ 320): biển `92A39826` từ đúng 3/38 lên 36/38 lần đọc.
- `layout_decode` trong `custom_ocr.py` chỉ cho mỗi vị trí nhận ký tự hợp lệ theo 10 kiểu biển có trong nhãn. Kết quả trùng 100% với giải mã tự do trên cả hai tập, nên mặc định để tắt.
- Task 076 (Phase A/B): ngưỡng confidence của E chọn trên Đà Nẵng/Sài Gòn chỉ nhận 1–3 crop ở mỗi chiều để đạt 98/99% precision thực nghiệm; khoảng Wilson rộng và coverage dưới 6%, chưa đủ để dùng làm điểm vận hành. Hanoi được giữ làm test, không dùng để chọn ngưỡng. Xem [task 076](../../../tasks/076-reject-and-char-vote-then-targeted-training.md) và các đường risk–coverage trong `custom_alpr/outputs/reject-eval-076/`.
- Vote theo từng ký tự trên 71 track đã xác nhận đạt 19/71, so với string vote 20/71; chưa có bằng chứng cải thiện. Kết quả chi tiết và tập 249 crop Codex+VLM riêng ở `custom_alpr/outputs/track-vote-076/`.

Lưu ý khi đọc số liệu:

- E-v1 đạt 93% trên 101 crop này **không có ý nghĩa**, vì nó đã được train trên chính các crop đó.
- Bản demo trên server đang bật `padding` và đặt `min_height = 0`. Theo số đo ở trên, viền 5% làm E đọc kém đi.

## 5. Đo refinement trên 101 crop (2026-10-30)

Task 070 đo trên CPU Mac Intel, torch 2.2.2 / numpy 1.26.4. Model E chính là `run-20261029-09-parseq-f-compress/best.pt`, dù tên thư mục có chữ `f`.

Đã khôi phục đủ 101 frame từ 5 video qua `resolve_stream` ưu tiên H.264; không có video lỗi. Seek kiểm tra đúng chỉ số frame trước và sau đọc. Đối chiếu crop metadata với ảnh đã lưu: đúng kích thước và vị trí, tương quan tối thiểu 0.923. Hai video trùng pixel; ba video khác pixel (MAE tối đa 18.97), phù hợp với khác biệt rendition/nén khi lấy lại stream. Vì vậy **cả ba cách cắt dưới đây dùng cùng frame vừa khôi phục**; không trộn crop cũ với crop mới.

| Model | (a) YOLO sát | (b) Viền 5% | (c) Refine 0.5/320 | (b) sửa đúng / làm sai so với (a) | (c) sửa đúng / làm sai so với (a) |
|---|---|---|---|---|---|
| E | 21/101 (20.8%) | 15/101 (14.9%) | 37/101 (36.6%) | +0 / −6 | +18 / −2 |
| E-v1, chỉ tham khảo | 75/101 (74.3%) | 84/101 (83.2%) | 62/101 (61.4%) | +10 / −1 | +6 / −19 |

- Fallback: **12/101 (11.9%)**, khi không đạt ngưỡng `accept_refined` thì giữ box gốc; không dựa vào nhãn hay độ tự tin OCR để chọn crop.
- Chi phí refine CPU: trung bình **37.65 ms/frame**, trung vị **29.96 ms/frame**; tập này có một crop cho mỗi frame, đã warm YOLO. Đây là thời gian riêng của refine, không gồm detector lần đầu và OCR; không đại diện cho frame nhiều biển.
- Kiểm tra lại chính ảnh tight đã lưu: E **25/101 (24.8%)**, E-v1 **94/101 (93.1%)**, tái lập mốc cũ. Khác biệt với bảng trên cho thấy rendition lấy lại ảnh hưởng đáng kể, đặc biệt với E-v1.
- E tăng ròng 16 crop sau refine; đủ vượt ngưỡng khoảng 10 crop của task. Đây là bằng chứng trên tập này, chưa chứng minh khả năng tổng quát sang camera mới. E-v1 giảm ròng 13 crop và đã train trên ảnh video gốc nên chỉ tham khảo; không thể kết luận bật refine sẽ cải thiện demo dùng E-v1.
- Đã xem ảnh lưới: refine có thể loại thân xe/đèn, nhưng cũng có trường hợp giữ box cũ hoặc vẫn đọc sai. Tất cả crop có kết quả thay đổi được lưu thành 7 trang cho E và 4 trang cho E-v1.

Tái chạy: `.venv-cpu/bin/python -m custom_alpr.refine_confirmed_eval`. Frame nằm ở `custom_alpr/data/collected/frames/`; kết quả chi tiết, thời gian từng crop và ảnh lưới nằm ở `custom_alpr/outputs/refine-confirmed-070/` (`report.json`, `rows.json`, `E-changed-*.png`, `E-v1-changed-*.png`). Dữ liệu/output bị gitignore.

Theo yêu cầu dừng sau số đo: chưa sửa `web_app.py`, chưa bắt đầu nhiệm vụ 2. Bước tiếp theo cần cân nhắc việc refine làm giảm E-v1 trước khi tích hợp vào demo hiện tại. Tăng crop video có nhãn bằng cách xem lại 249 crop đồng thuận và 6 track mâu thuẫn; khoảng 300 crop có thể tách val video.

### Kiểm tra camera chưa thấy — task 071

Đã kiểm tra train/holdout của hai fold không trùng video ID hoặc sha256. Bỏ _pad: held-out Sài Gòn n=58, Đà Nẵng n=37. Model train Đà Nẵng: tight 12/58 → refine 17/58 (+7/−2), chênh +8.6 điểm %, paired bootstrap 95% [−1.7, +19.0]. Model train Sài Gòn: 10/37 → 19/37 (+9/−0), chênh +24.3 điểm % [+10.8, +37.8]. CER lần lượt 38.3% → 23.1% và 28.5% → 11.2%. Đây là các crop khác biệt, không phải xe độc lập. E-v1/all bị loại khỏi phép đo.

Kaggle 1202 ảnh: E 1074, train Đà Nẵng 1075, train Sài Gòn 1078 ảnh đúng; tái lập accuracy lịch sử. Bảng Wilson CI, CER CI, fallback và CPU timing đầy đủ nằm ở tasks/071-measure-refine-on-held-out-cameras.md và outputs/refine-holdout-071/report.json. **Chưa nên tích hợp refine: chưa đủ dữ liệu theo tiêu chí cả hai fold, vì CI của held-out Sài Gòn chứa 0.**

### Pose + warp — task 073 (2026-10-30)

YOLOv8n-pose 4 góc được train100epoch trên desktop RTX5060; giữ split/loại trùng detection, bỏ polygon không có4điểm, giữ nhãn hình chữ nhật. Bộ pose cuối có3379ảnh train/3821biển và1117ảnh val/1278biển.

Trên cùng detection val1127ảnh/1295box: current detector mAP50=.9947, mAP50-95=.8883; pose=.9944/.9026. Sai số góc trung vị1.53px=3.33%GTboxheight (3.70%chiều cao theo cạnh tứ giác), ghép1266/1278biển.

OCR E, cùng95frame task071: chọn lề5% trên Đà Nẵng trước khi chấm Sài Gòn một lần.

| Camera | (a) Box cũ | (c) Refine | (d) Box pose | (e) Warp5% | Warp sửa/làm sai vs(a) | Warp không có crop |
|---|---|---|---|---|---|---|
| Đà Nẵng |9/37|20/37|17/37|16/37|7/0|8/37|
| Sài Gòn |11/58|15/58|11/58|13/58|8/6|11/58|

Warp so với refine: Đà Nẵng +2/−6, Δ−10.8điểm% [−24.3,+2.7]; Sài Gòn +6/−8, Δ−3.4điểm% [−15.5,+8.6], paired bootstrap95%. Chênh lệch dưới khoảng10crop và CI chứa0: chưa đủ bằng chứng cải thiện.

CPU toàn95frame, không gồm OCR: detector+refine trung bình162.34ms, pose+warp70.82ms (khoảng2.29lần nhanh hơn). **Chưa nên thay detector+refine bằng pose+warp: mAP và tốc độ tốt hơn chưa chuyển thành OCR video tốt hơn.** Bảng Wilson/CER/missing/timing đầy đủ: tasks/073-train-plate-pose-and-evaluate-ocr.md và custom_alpr/outputs/pose-073/report.json. Không tích hợp.

### Quét lề warp 1-5% — task 074 (thăm dò)

Dùng lại góc pose đã lưu và model E, không chạy lại detector. Lề1/2/3/4/5%: Đà Nẵng đúng14/15/14/15/16 trên37; Sài Gòn10/12/11/11/13 trên58; tổng24/27/25/26/29 trên95. Mốc5% tái lập đúng toàn bộ chuỗi OCR của task073. Không mức nào vượt refine35/95; chênh giữa các lề nhỏ, chưa đủ bằng chứng kết luận lề tối ưu tổng quát. Đây là phân tích sau khi đã xem Sài Gòn, không phải test sạch mới. Chi tiết Wilson/CER/paired bootstrap: tasks/074-score-pose-warp-margins.md và outputs/pose-margin-074/report.json.

Task075 mở rộng lề6/7% (thăm dò, cùng góc pose/E): Đà Nẵng đều16/37; Sài Gòn đều12/58; tổng28/95, so với5% là29/95 và refine35/95. Mốc5% tái lập đúng toàn bộ chuỗi OCR. Chênh1crop chưa đủ bằng chứng, không thấy lợi ích khi tăng lên6-7%; không thay mức lề hay kết luận test sạch của task073. Chi tiết: tasks/075-score-pose-warp-margins-6-7.md.
