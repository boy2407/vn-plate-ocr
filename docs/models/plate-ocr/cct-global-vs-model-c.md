# So sánh CCT-XS v2 global và PARSeq model C

Ngày đánh giá: 2026-10-28. Hai OCR được chấm trên cùng test manifest Kaggle hiện tại: 1.202 ảnh biển và 9.723 ký tự tham chiếu. Một ảnh (`mb_275.jpg`) đã được gỡ khỏi manifest trong lần rà soát nhãn; dự đoán PARSeq được đối chiếu theo đúng 1.202 ảnh còn lại. CCT được chạy bằng `fast-plate-ocr==1.1.0`, model `cct-xs-v2-global-model`; PARSeq là checkpoint run `run-20261028-04-parseq-aug-noise`.

| OCR | Biển đúng hoàn toàn | CER |
| --- | ---: | ---: |
| PARSeq model C | 1.077/1.202 (89,60%) | 359/9.723 (3,69%) |
| CCT-XS v2 global | 948/1.202 (78,87%) | 881/9.723 (9,06%) |

Trong 1.202 ảnh, cả hai đúng 931; chỉ PARSeq đúng 146; chỉ CCT đúng 17; cả hai sai 108. Trên tập này, PARSeq hơn 10,73 điểm phần trăm exact plate accuracy, với CER thấp hơn 5,37 điểm phần trăm.

## Cách tái lập

Từ gốc repo, cài dependencies của project và chạy:

```bash
python -m pip install -e '.[onnx]'
python -m custom_alpr.compare_global_cct
```

Script đọc test manifest và dự đoán PARSeq đã lưu, chạy CCT trên cùng crop RGB, rồi ghi `summary.json` và `predictions.csv` trong `custom_alpr/outputs/train-custom-ocr/cct-xs-v2-global-test-comparison/`. Có thể đổi input/model/thiết bị/output bằng các cờ `--manifest`, `--reference-predictions`, `--model`, `--device`, và `--output-dir`.

## Giới hạn

- Đây là so sánh OCR trên crop biển đã có sẵn; không tính lỗi detector YOLO.
- Hai model dùng preprocessing riêng theo cấu hình, nhưng nhận cùng ảnh nguồn và nhãn test.
- Alphabet CCT global là `0–9`, `A–Z`, `_`; test có 28 nhãn chứa `Đ`, ký tự không có trong alphabet CCT. Đây là một giới hạn phù hợp dữ liệu có thể ảnh hưởng kết quả, nhưng chưa xác định được bao nhiêu lỗi CCT do riêng ký tự này.
- Kết quả phản ánh test split Kaggle này, chưa đại diện mọi camera/điều kiện triển khai.
