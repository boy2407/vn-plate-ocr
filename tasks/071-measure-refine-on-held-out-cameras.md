# 071 — Measure refine on held-out cameras

Status: done (measurement only). Started/completed: 2026-10-30.

Không dùng E-v1 hoặc fold all; không train hay tích hợp. CPU Mac Intel, Python 3.12, torch 2.2.2, numpy 1.26.4, 4 threads. Refine context 0.5 / size 320. E = run-20261029-09-parseq-f-compress/best.pt.

## Kiểm tra rò dữ liệu trước đo

- train-danang-ptz: 10651 dòng train; 116 dòng holdout → 58 crop_id sau bỏ _pad. Train video: RjeyplZXDCo, dkAcXmw2hjA, wzTiGOlidk4; holdout: gZWiLrQnb10. Giao video ID = 0, giao sha256 manifest = 0.
- train-saigon-drive: 11281 dòng train; 74 dòng holdout → 37 crop_id. Train video: gZWiLrQnb10; holdout: RjeyplZXDCo, dkAcXmw2hjA, wzTiGOlidk4. Giao video ID = 0, giao sha256 manifest = 0.
- Code kiểm tra trước inference và dừng nếu phát hiện trùng. Không loại âm thầm mẫu nào. 95 crop_id là các quan sát frame, không phải 95 xe độc lập.
- Tái dùng đủ 95 frame task070 đã kiểm tra seek trước/sau đọc. Kiểm tra lại tương quan và vị trí từng crop ở lần chạy này (tối thiểu 0.923). Các biến thể đều cắt từ cùng frame; khác biệt rendition so với crop cũ vẫn là hạn chế đã ghi ở task070.

## Kết quả

Exact match: k/n, % [Wilson 95%]. CER: % [bootstrap 95%], n và số ký tự ghi riêng. Không dùng Wilson cho CER vì edit distance không phải phép thử nhị thức.

| Model — held-out camera | (a) tight exact | (b) padding exact | (c) refine exact | Refine sửa / làm sai | Δ exact điểm % [paired bootstrap 95%] |
|---|---|---|---|---|---|
| E — held-out camera: saigon-drive | 11/58 = 19.0% [10.9, 30.9] | 8/58 = 13.8% [7.2, 24.9] | 15/58 = 25.9% [16.3, 38.4] | 6 / 2 | +6.9 [-1.7, +17.2] |
| E — held-out camera: danang-ptz | 9/37 = 24.3% [13.4, 40.1] | 7/37 = 18.9% [9.5, 34.2] | 20/37 = 54.1% [38.4, 69.0] | 11 / 0 | +29.7 [+16.2, +45.9] |
| train-danang-ptz — held-out camera: saigon-drive | 12/58 = 20.7% [12.3, 32.8] | 8/58 = 13.8% [7.2, 24.9] | 17/58 = 29.3% [19.2, 42.0] | 7 / 2 | +8.6 [-1.7, +19.0] |
| train-saigon-drive — held-out camera: danang-ptz | 10/37 = 27.0% [15.4, 43.0] | 8/37 = 21.6% [11.4, 37.2] | 19/37 = 51.4% [35.9, 66.6] | 9 / 0 | +24.3 [+10.8, +37.8] |

| Model — held-out camera | CER (a) | CER (b) | CER (c) | n / ký tự |
|---|---|---|---|---|
| E--heldout-saigon-drive | 37.2% [30.0, 44.8] | 40.7% [33.1, 48.4] | 22.7% [16.9, 28.7] | 58 / 494 |
| E--heldout-danang-ptz | 30.8% [22.3, 40.2] | 41.7% [30.8, 53.3] | 11.9% [6.8, 18.0] | 37 / 295 |
| train-danang-ptz--heldout-saigon-drive | 38.3% [30.6, 46.2] | 42.3% [34.6, 50.3] | 23.1% [17.0, 29.4] | 58 / 494 |
| train-saigon-drive--heldout-danang-ptz | 28.5% [20.6, 37.1] | 36.9% [27.9, 46.1] | 11.2% [6.4, 17.3] | 37 / 295 |

| Model — held-out camera | Sửa đúng [Wilson 95%] | Làm sai [Wilson 95%] | Fallback [Wilson 95%] | CPU mean / median ms/frame |
|---|---|---|---|---|
| E--heldout-saigon-drive | 6/58 = 10.3% [4.8, 20.8] | 2/58 = 3.4% [1.0, 11.7] | 10/58 = 17.2% [9.6, 28.9] | 30.56 / 28.77 (n=58) |
| E--heldout-danang-ptz | 11/37 = 29.7% [17.5, 45.8] | 0/37 = 0.0% [0.0, 9.4] | 1/37 = 2.7% [0.5, 13.8] | 30.93 / 29.51 (n=37) |
| train-danang-ptz--heldout-saigon-drive | 7/58 = 12.1% [6.0, 22.9] | 2/58 = 3.4% [1.0, 11.7] | 10/58 = 17.2% [9.6, 28.9] | 30.56 / 28.77 (n=58) |
| train-saigon-drive--heldout-danang-ptz | 9/37 = 24.3% [13.4, 40.1] | 0/37 = 0.0% [0.0, 9.4] | 1/37 = 2.7% [0.5, 13.8] | 30.93 / 29.51 (n=37) |

Timing là riêng refine sau warm YOLO; một crop/frame, chưa gồm detector lần đầu hoặc OCR. Các model dùng lại cùng box refine nên timing giống nhau theo camera.

Paired bootstrap: 100000 lần, seed 71; CER bootstrap 10000 lần. Đơn vị lấy mẫu là crop_id; frame cùng xe có tương quan nên CI có thể lạc quan. Không gọi +5 hoặc +9 crop là bằng chứng đủ; fold Sài Gòn chứa 0 trong CI, chưa đạt điều kiện cả hai fold.

## Kaggle test, không refinement

| Model | Exact [Wilson 95%] | CER [bootstrap 95%] | Sửa / làm sai so với E | Tái lập accuracy lịch sử |
|---|---|---|---|---|
| E | 1074/1202 = 89.4% [87.5, 91.0] | 3.6% [2.8, 4.4] (n=1202, ký tự=9723) | 0 / 0 | True |
| train-danang-ptz | 1075/1202 = 89.4% [87.6, 91.0] | 3.3% [2.6, 4.0] (n=1202, ký tự=9723) | 14 / 13 | True |
| train-saigon-drive | 1078/1202 = 89.7% [87.8, 91.3] | 3.3% [2.6, 4.0] (n=1202, ký tự=9723) | 16 / 12 | True |

Kaggle test có 1202 ảnh. Accuracy cả ba model khớp chính xác báo cáo lịch sử; không có tụt accuracy quan sát được. JSON chứa tỉ lệ và Wilson CI cho cả sửa/làm sai so với E. Đây không phải bằng chứng không suy giảm ở mọi dữ liệu tương lai.

## Artifact và kiểm tra

- Tái chạy: `.venv-cpu/bin/python -m custom_alpr.refine_holdout_eval`.
- `custom_alpr/outputs/refine-holdout-071/report.json`, `leakage-audit.json`, `rows.json`, dự đoán Kaggle từng ảnh và contact sheet tất cả crop thay đổi kết quả. Đã xem sheet fold held-out Đà Nẵng: có crop cắt sát giúp đọc đúng và crop vẫn sai.
- `python -m pytest custom_alpr/tests -q`: lỗi collection do thiếu `custom_alpr.yolo_vast.corruptions` sẵn có.
- `ruff check custom_alpr`: 116 lỗi ở cây mã hiện có; file `refine_holdout_eval.py` đạt Ruff.
- Giữ nguyên web_app.py, web assets, deploy và checkpoint; không commit.

**Kết luận: Chưa nên tích hợp refine vào InferenceEngine.recognize vì chưa đủ dữ liệu chứng minh cải thiện ở cả hai fold camera giữ lại.**
