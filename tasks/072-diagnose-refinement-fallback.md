# 072 — Diagnose refinement fallback and validate proposals

Status: stopped at step 3 — required baseline mismatch. Started/updated: 2026-10-30.

Model E = `custom_alpr/outputs/train-custom-ocr/run-20261029-09-parseq-f-compress/best.pt`. CPU Mac, .venv-cpu, 4 threads. Context0.5/size320 fixed. All 95 crop_id use cached seek-verified frames, no download. Production refine_box/web/deploy/checkpoints unchanged; no train/commit.

## 1. Fallback diagnosis

One category per crop, current check order; independently asserted equal to production refine_box and task071. No degenerate box. Candidates are predictor outputs at confidence>=.25, so no_candidate does not prove none exist below .25.

| Camera (n) | no_candidate | low_confidence | aspect | area | outside_slack | accepted |
|---|---|---|---|---|---|---|
| saigon-drive (58) | 4 | 1 | 0 | 5 | 0 | 48 |
| danang-ptz (37) | 1 | 0 | 0 | 0 | 0 | 36 |

Fallback Sài Gòn: 10/58 = 17.2% [9.6, 28.9]. Đà Nẵng: 1/37 = 2.7% [0.5, 13.8].

Sài Gòn area: 2 candidate dưới MIN_AREA, 3 trên MAX_AREA. Các fallback đều tối đa một candidate; không trường hợp có candidate thứ hai >=.4. Toàn tập có 3 crop với candidate thứ hai >=.4 (1 Sài Gòn, 2 Đà Nẵng), đều accepted. Chưa có bằng chứng chọn candidate gần tâm sẽ chữa fallback.

Contact sheet Sài Gòn cho thấy:
- f30172-x1240y681 và f50996-x1425y721: box gốc có thân xe, biển nhỏ; candidate area .120/.170 đúng vùng biển nhưng E vẫn đọc sai.
- f48524-x1703y705 và f49196-x1812y898: box gốc thiếu vùng biển; candidate lớn hơn (1.009/1.487) đọc đúng ở phép chẩn đoán bắt buộc.
- f47052-x1590y833: candidate ratio1.036 vẫn sai.
- Bốn no_candidate cùng biển51B22487: tight đã đúng3/4.
- f50572-x1616y692: có hai biển trong context; candidate duy nhất confidence .260 chọn biển nhỏ phía trên và đọc sai. Không có cơ sở hạ confidence để nhận candidate này.

rows.json chứa confidence/aspect/area_ratio/candidate_count/original_height/second_candidate_ge_04, mọi candidate và E tight/candidate. Contact sheet: context với box gốc xanh, candidate đỏ kể cả bị loại, nhãn, hai crop/OCR. No candidate dùng null.

## 2. Hai đề xuất cố định trước kiểm định

1. remove_min_area: bỏ riêng cận dưới area, giữ cận trên .95 và mọi kiểm tra khác. Ảnh f30172/f50996 cho thấy biển nhỏ trong box lỏng. Không chọn ngưỡng số mới theo Sài Gòn.
2. remove_max_area: bỏ riêng cận trên area, giữ cận dưới .2 và mọi kiểm tra khác. Ảnh f48524/f49196 cho thấy box gốc cắt thiếu. Không chọn ngưỡng số mới theo Sài Gòn.

Sài Gòn đã được dùng một lần ở bước chẩn đoán/tạo ý tưởng; không còn là holdout sạch cho ý tưởng này. OCR candidate bị loại là chẩn đoán theo yêu cầu, không phải score proposal. Không chỉnh lại proposal theo kết quả kiểm định.

## 3. Kiểm định

### (i) Original 358 ảnh / 429 plate: baseline không tái lập hoàn toàn

Nguồn do người dùng xác nhận: outputs/detval-sample/{images,labels}/val. Cách tạo: mỗi nguồn Dieu/Hung/Tgmt/carlong/greenpack lấy file thứ1,3,5,… theo ls, tối đa80/nguồn:64+54+80+80+80=358. min-height16, first conf.4/imgsz640, matchGT IoU>=.5. Tập con của lần đo val đã cache; đã đối chiếu byte ảnh/nhãn của các match.

| Baseline | n | median IoU trước → sau | box tệ hơn >.1 |
|---|---|---|---|
| Yêu cầu |429|.892 → .925|0|
| CPU tái lập |429|.892495 → .924949|**1**|

Dieu_0087.png: GT=first=[246,313,272,331], final=[247,313,271,330], confidence .9054. IoU1.000 → .871795, giảm .128205. Chạy lại độc lập refine_box nguyên bản trên ảnh mẫu xác nhận cùng kết quả. Chưa xác định nguồn khác biệt với báo cáo cũ, không tự quy cho CPU/GPU.

Đã báo người dùng trước khi sử dụng. Không chấm proposal trên429, không mở gate Sài Gòn. Script dừng có chủ đích nếu baseline không tái lập.

### Điều kiện phụ: toàn detector val

Đo trước khi nhận thông tin nguồn429:1127 ảnh,1274 GT cao>=16,1268 GT ghép firstIoU>=.5. Không thay thế điều kiện(i).

| Variant | n | median IoU trước → sau | tệ hơn first >.1 [Wilson95] | tệ hơn baseline >.1 [Wilson95] |
|---|---|---|---|---|
| baseline | 1268 | 0.894557 → 0.929217 | 2/1268 = 0.2% [0.0, 0.6] | 0/1268 = 0.0% [0.0, 0.3] |
| remove_min_area | 1268 | 0.894557 → 0.929217 | 2/1268 = 0.2% [0.0, 0.6] | 0/1268 = 0.0% [0.0, 0.3] |
| remove_max_area | 1268 | 0.894557 → 0.933653 | 8/1268 = 0.6% [0.3, 1.2] | 6/1268 = 0.5% [0.2, 1.0] |

Bỏ max_area thất bại ở điều kiện phụ: thêm6 box tệ hơn baseline>.1. Bỏ min_area không đổi số đo, chưa chứng minh lợi ích.

### (ii) E — Đà Nẵng37 crop

Đã đo trước khi nguồn429 được cung cấp, giữ kết quả riêng; không mở gate khi(i) chưa đạt. Wilson95 cho exact/fallback; CER bootstrap10000; paired delta bootstrap100000 seed72. Crop cùng xe có tương quan.

| Variant | Exact [Wilson95] | CER [bootstrap95], n=37/295chars | sửa/làm sai vs tight | Δ điểm% [paired95] | Fallback [Wilson95] |
|---|---|---|---|---|---|
| tight | 9/37 = 24.3% [13.4, 40.1] | 30.8% [22.3, 40.2] | 0/0 | +0.0 [+0.0, +0.0] | — |
| baseline | 20/37 = 54.1% [38.4, 69.0] | 11.9% [6.8, 18.0] | 11/0 | +29.7 [+16.2, +45.9] | 1/37 = 2.7% [0.5, 13.8] |
| remove_min_area | 20/37 = 54.1% [38.4, 69.0] | 11.9% [6.8, 18.0] | 11/0 | +29.7 [+16.2, +45.9] | 1/37 = 2.7% [0.5, 13.8] |
| remove_max_area | 20/37 = 54.1% [38.4, 69.0] | 11.9% [6.8, 18.0] | 11/0 | +29.7 [+16.2, +45.9] | 1/37 = 2.7% [0.5, 13.8] |

Cả hai proposal so với baseline refine: sửa0/làm sai0, delta0 [0,0]. Tỉ lệ sửa/làm sai đều0/37, Wilson[0,9.4%]. Đà Nẵng không có area fallback nên proposal không tác động crop nào. Không coi đây là chứng minh hiệu quả trên lỗi area. JSON lưu CI đầy đủ cho sửa/làm sai vs tight và vs baseline.

### Sài Gòn

**Proposal evaluation runs:0.** Điều kiện(i) bị dừng; bỏ max_area còn thất bại ở điều kiện phụ. Baseline E task071 tham chiếu: tight11/58, refine15/58 (+6/−2), delta+6.9điểm% [−1.7,+17.2], fallback10/58. Không tính OCR candidate chẩn đoán thành điểm proposal.

## Artifacts / checks

Outputs: custom_alpr/outputs/refine-fallback-072/{report.json,rows.json,counts.json,proposals.json,validation-report.json,original-429-check.json,detector-val-candidates.json,danang-proposal-rows.json,*-fallback.png}.

Scripts: .venv-cpu/bin/python -m custom_alpr.refine_fallback_report; .venv-cpu/bin/python -m custom_alpr.refine_proposal_eval. Validation script hiện dừng baseline gate như yêu cầu. Các kết quả supplemental đã đo trước đó được giữ lại.

Ruff hai file mới đạt. Pytest vẫn lỗi collection thiếu custom_alpr.yolo_vast.corruptions; không sửa. Ruff toàn cây116 lỗi sẵn có.

**Chưa nên đề xuất refine thành tùy chọn trong web_app: baseline429 chưa tái lập đúng và chưa đủ bằng chứng trên Sài Gòn.**
