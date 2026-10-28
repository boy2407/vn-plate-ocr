# Task 051: Plan moderate YOLO robustness training

## Goal

Plan a new YOLO training experiment covering glare, noise, blur, low resolution, compression, lighting, viewing angle and multiple plates, using moderate augmentation. This task prepares the plan; it does not launch training.

## Checklist

- [x] Specify image quality groups, mild/moderate ranges and corruption budget.
- [x] Cover real multi-plate scenes, 2×2 Mosaic, box correctness and hard negatives.
- [x] Specify controlled training runs, validation/stress evaluation and acceptance criteria.
- [x] Explain multi-plate inference needs per-track crop selection rather than global top one.
- [x] Deliver the plan with concrete outputs and startup checks.

## Kế hoạch triển khai

### 1. Dữ liệu và mục tiêu

- Giữ YOLOv8n; checkpoint hiện tại làm mốc so sánh. Lượt train cũ dùng 320, 50 epoch.
- Giữ split hiện tại: 9.829 train / 1.229 validation / 1.228 test. Bộ Kaggle `duydieunguyen/licenseplates` version 1 có 3.592 train / 449 validation / 450 test trong đó, đối chiếu được bằng pixel hash.
- Clip người dùng đã xem là dữ liệu phát triển. Tách thêm clip mới chưa dùng để đánh giá cuối cùng; chia theo video/xe, không rải frame liền kề vào train và holdout.
- Bổ sung khoảng 500–1.500 frame được gán box đầy đủ từ cảnh giống clip thực: biển nhỏ, xe chạy, nhiều xe, biển vàng/trắng/xanh, góc xiên, ban đêm và phản sáng. Con số là mục tiêu thu thập, không phải dữ liệu đã có.
- Bổ sung hard negatives: logo xe, đèn, lưới tản nhiệt, bảng hiệu, số trên thân xe. Ảnh âm phải thật sự không có biển chưa gán nhãn; ảnh còn biển thì gán đầy đủ mọi biển. Mục tiêu ban đầu 10–20% ảnh train là negative, xem lại tỉ lệ âm đã có trước khi thêm.
- Các case bắt nhầm trong clip phải được rà trực tiếp. Augmentation không thay thế positive/negative đúng miền dữ liệu.

### 2. Ngân sách làm xấu ảnh

Các phép sau chỉ chạy trên train, theo kiểu on-the-fly. Các khoảng là preset thăm dò ban đầu, chưa phải tối ưu đã được chứng minh.

- 40% ảnh không thêm hiệu ứng suy giảm chất lượng; vẫn có thể có augmentation hình học thông thường.
- 45% ảnh thêm một nhóm hiệu ứng.
- 15% ảnh thêm tối đa hai nhóm, giảm cường độ của mỗi nhóm khi ghép.
- Không ghép chói + tối hoặc nhiều blur liên tiếp. Không làm trắng, che kín hay phá hỏng toàn bộ biển rồi coi đó là một mẫu huấn luyện bình thường.

| Nhóm | Mức nhẹ–vừa dự kiến | Tình huống cần học |
| --- | --- | --- |
| Nhiễu cảm biến | Gaussian sigma 2–12 theo thang pixel 0–255; thêm Poisson nhẹ nếu phù hợp dữ liệu camera | ISO cao, camera tối |
| Nhiễu điểm | Salt-and-pepper 0,1–0,5% pixel, chọn ít hơn Gaussian | Pixel lỗi, nhiễu đường truyền |
| Mờ chuyển động | Kernel 3/5 px, góc ngẫu nhiên; xem kỹ tác động lên biển nhỏ | Xe chạy, camera rung |
| Mờ lấy nét | Gaussian sigma 0,3–0,8, kernel nhỏ | Mất nét nhẹ |
| Chói/phản sáng | Vùng sáng mềm, alpha khoảng 0,1–0,3; ưu tiên một phần biển, giữ phần lớn cấu trúc còn nhìn thấy | Nắng, đèn pha, phản quang |
| Thiếu sáng/ngược sáng | Gamma khoảng 0,7–1,5 theo hai hướng sáng/tối; độ sáng 0,7–1,2; bóng đổ mềm | Camera ban đêm, bóng xe |
| Màu/cân bằng trắng | Hue ±0,01; saturation khoảng ±0,3; đổi nhiệt màu nhẹ | Ánh đèn, camera khác nhau |
| Nén ảnh/video | JPEG quality 55–90; thêm mẫu video nén thật | Block/ringing do nén |
| Độ phân giải thấp | Downscale 0,6–0,9 rồi resize lại | Biển ở xa, camera kém |
| Góc nhìn | Xoay ±5°, shear nhỏ, perspective nhẹ; dịch chuyển 5%, scale khoảng 0,85–1,15 | Camera lệch, xe đổi hướng |
| Thời tiết/che khuất | Sương/mưa nhẹ, bóng và bụi nhỏ; che dưới khoảng 10–15% vùng biển | Môi trường thực tế |

Tần suất chọn nhóm trong nhánh làm xấu: noise 25%, blur 25%, ánh sáng 20%, nén/low-resolution 20%, thời tiết/che khuất 10%. Đây là trọng số chọn nhóm, không phải xác suất độc lập để áp dụng tất cả cùng lúc. Nhóm hình học và màu dùng cấu hình chung của cả hai arm đối chứng.

Ưu tiên Gaussian, motion blur, JPEG, glare trước; giữ salt-and-pepper, sương/mưa và che khuất ở tần suất thấp. Cảnh đêm/IR hoặc nước bám lens nặng cần ảnh thực, không trông chờ hoàn toàn vào ảnh giả lập.

### 3. Mosaic 2×2 và nhiều biển

- Hiểu 2×2 là Mosaic ghép bốn ảnh trong quá trình train. Khởi đầu `mosaic=0.4`, giữ đầy đủ box của cả bốn ảnh.
- Tắt Mosaic 10 epoch cuối để mô hình hoàn thiện trên cảnh tự nhiên; tránh tăng tần suất Mosaic chỉ vì muốn có nhiều biển.
- Bổ sung ảnh thật có 2–4 và trên 4 biển; đánh giá tách theo 0 / 1 / 2–4 / trên 4 biển và theo kích thước biển.
- Kiểm tra các box sau resize/Mosaic/perspective: clipping đúng, box không âm/rỗng, thống kê box quá nhỏ bị loại. Không để augmentation tạo hàng loạt biển chỉ còn 1–2 pixel.
- Với detector có thể thử lật ngang, không lật dọc. Không dùng nhãn OCR của ảnh lật để train recognizer trong thí nghiệm YOLO này.

### 4. Hai lượt train đối chứng

1. Chốt manifest, hash, augmentation code và bảng preview ảnh/box trước khi chạy.
2. Smoke run một epoch/giới hạn batch trên SSH desktop, xem VRAM, tốc độ và lỗi nhãn. Đồng bộ bằng rsync.
3. **A – control:** bắt đầu từ cùng checkpoint hiện tại, fine-tune 30 epoch ở 640, dữ liệu và hình học/màu/Mosaic như preset mới, chưa thêm các nhóm làm xấu tùy chỉnh.
4. **B – moderate:** cùng khởi tạo, cùng dữ liệu, seed 42, optimizer/schedule và số epoch; chỉ thêm nhánh suy giảm chất lượng ở mục 2.
5. Khởi đầu batch 8 trên RTX 5060, tăng lên nếu smoke run chứng minh đủ VRAM. Giữ cùng effective batch giữa hai arm. Lưu checkpoint, loss, metric và config riêng.
6. Nếu B đạt tiêu chí, mới cân nhắc thêm 20 epoch hoặc lặp seed. Không tăng độ nặng augmentation chỉ vì train loss thấp.

Không ước lượng thời gian train trước smoke run. Không thay YOLOv8n thành model lớn trong cùng phép so sánh augmentation.

### 5. Đánh giá trước khi chọn model

- Validation gốc giữ nguyên, dùng so sánh clean accuracy.
- Tạo các bản stress validation cố định theo seed và từng nhóm: noise / motion / defocus / glare / low-light / JPEG / low-resolution / góc nhìn / thời tiết; mỗi nhóm có mức nhẹ và vừa. Biến đổi hình học phải biến đổi cả box.
- Báo cáo AP50, AP50–95, precision, recall, FP/ảnh, FN và recall biển nhỏ; thêm FP trên negative-only/logo và kết quả nhóm nhiều biển.
- Chọn threshold bằng validation rồi khóa. Test chỉ chạy để báo cáo checkpoint cuối; không dùng test để chỉnh augmentation.
- Mục tiêu sàng lọc: clean AP50–95 giảm không quá 1 điểm %, FP trên hard negatives giảm khoảng 25%, FN ở điều kiện vừa giảm khoảng 15% hoặc recall tăng 3 điểm %. Đây là mục tiêu, không phải kết quả đã đạt.
- Đo FPS YOLO và end-to-end riêng, trên cùng thiết bị, input và số biển. Báo rõ frame bỏ qua; chưa cam kết xử lý đủ 60 FPS.
- 192 ảnh validation vừa đo cho checkpoint cũ cho thấy 640 tốt hơn 960/1280 về tradeoff FP/FN. Không suy ra cứ tăng kích thước inference là tốt hơn; train 640 mới phải được đo lại.

### 6. Pipeline demo sau khi cần nhận diện nhiều biển

Yêu cầu cũ chỉ đưa một box tốt nhất toàn frame sang OCR đang hạn chế demo nhiều biển. Phiên bản nhiều biển nên là:

YOLO lấy các box qua threshold/NMS → theo dõi từng biển → mỗi track chọn crop tốt nhất → OCR khi track mới hoặc crop tốt hơn → đồng thuận qua thời gian → một thẻ cho mỗi chuỗi biển đã gom.

Giữ một crop tốt nhất **cho từng track**, thay cho chỉ một crop cho toàn frame. Có thể giảm tần suất OCR trên track đã ổn định để giữ tốc độ. Raw observations vẫn được giữ để kiểm tra lỗi và tránh nhầm số nhóm với số xe thật.

### 7. Sản phẩm cần có trước khi chạy full

- Manifest và hash split, danh sách hard-negative đã rà nhãn.
- Config A/B, module augmentation có version, seed và log xác suất thực tế.
- Contact sheet gốc/augmented có box cho từng nhóm, bao gồm Mosaic 2×2.
- Báo cáo smoke run và ước lượng thời gian từ tốc độ đo thật.
- Sau train: best/last weights, args/config, CSV train log, bảng clean/stress/multi-plate/negative, FPS và demo trên clip mới.

Tham khảo khả năng Mosaic/custom augmentation: https://docs.ultralytics.com/guides/yolo-data-augmentation/ . Ultralytics có thể tự dùng blur/median blur/grayscale/CLAHE xác suất nhỏ nếu Albumentations được cài; cần ghi rõ dependency và thay bằng danh sách kiểm soát trong cả A/B để tránh augmentation ngầm khác nhau.

## Done criteria

A concrete experiment plan can be implemented without guessing augmentation severity, split policy, baselines or evaluation criteria.

## Progress log

- 2026-10-28: Requested by user, expanded to glare/noise/blur/2×2/multi-plate and other common quality factors at moderate severity.
- 2026-10-28: Completed the plan with moderate corruption budget, data/negative policy, paired 640 fine-tuning, fixed validation stress cases, multi-plate inference design and explicit acceptance targets. Full training has not been launched.
- 2026-10-28: User requested review of poor clip detections. Inspected 12 sampled observations from two clips with scene context and original crops. Hard negatives and better domain boxes are required alongside corruption augmentation; examples include body lettering/plates merged into one box and a shirt mistaken for a plate. Review artifact: `custom_alpr/outputs/web-demo/detector-clip-review.jpg`.
