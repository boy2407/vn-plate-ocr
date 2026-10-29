# Hướng dẫn gán nhãn biển số (cho agent gán nhãn)

Bạn gán nhãn ảnh crop biển số xe Việt Nam cắt từ video YouTube. Nhãn của bạn sẽ
dùng để train lại mô hình OCR. **Một nhãn sai còn có hại hơn một nhãn bỏ trống**,
nên nếu không chắc chắn thì đừng đoán.

## File cần sửa

`queue.csv`, nằm cùng thư mục với file này. Mỗi dòng là một ảnh:

| Cột | Ý nghĩa | Bạn có sửa không? |
|---|---|---|
| `crop_id` | Mã ảnh | Không |
| `image_path` | Đường dẫn ảnh, tính từ thư mục này | Không |
| `video_id`, `track_id` | Ảnh cùng `track_id` là cùng một biển, chụp ở các thời điểm khác nhau | Không |
| `verdict` | Kết luận của bạn | **Có** |
| `plate_text` | Chữ trên biển | **Có** |
| `lines` | Biển có 1 hay 2 dòng chữ | **Có** |
| `note` | Ghi chú ngắn, có thể để trống | **Có** |

Chỉ sửa 4 cột cuối. Không đổi tên cột, không sửa các cột khác, không xoá dòng,
không đổi thứ tự dòng.

## Quy trình

1. Làm theo lô 50 dòng: lấy 50 dòng đầu tiên có `verdict` còn trống.
2. Với mỗi dòng, **mở ảnh phóng to và nhìn ảnh**: thay `crops/` trong
   `image_path` bằng `view/`, ví dụ `wzTiGOlidk4/view/wzTiGOlidk4-f540-x307y831.png`.
   Ảnh trong `crops/` chỉ cao vài chục pixel; ảnh trong `view/` là cùng ảnh đó
   phóng to lên cao 192 px để dễ nhìn (không thêm chi tiết nào). Đọc từ ảnh,
   không suy ra từ tên file hay từ các dòng khác.
3. Có thể mở thêm các file cùng tên trong `crops/` để nhìn rõ hơn:
   - đuôi `_pad.png`: crop rộng hơn 5%, để xem viền biển khi ảnh bị cắt sát quá;
   - đuôi `_ref.png` (không phải ảnh nào cũng có): crop cắt sát biển hơn, hữu ích
     khi ảnh chính có nhiều thân xe xung quanh.

   Cả ba file là cùng một biển, nên nhãn ghi một lần cho dòng đó.
4. Ghi `verdict`, `plate_text`, `lines` cho dòng đó.
5. Xong lô thì lưu file. Giữ nguyên dấu phân cách `,` và mã hoá UTF-8.

## `verdict`: chọn đúng một giá trị

| Giá trị | Khi nào dùng | `plate_text` | `lines` |
|---|---|---|---|
| `label` | Đọc được **chắc chắn từng ký tự** của cả biển | bắt buộc | bắt buộc |
| `unreadable` | Là biển số, nhưng có ít nhất một ký tự không chắc (mờ, nhỏ, lóa, bị che) | để trống | điền nếu thấy rõ |
| `partial` | Biển bị cắt mất một phần, thiếu ký tự | để trống | điền nếu thấy rõ |
| `not_plate` | Không phải biển số: logo, chữ trên thân xe, biển hiệu, áo… | để trống | để trống |

**Không chắc thì ghi `unreadable`.** Đừng điền ký tự "có vẻ giống".

Ngược lại, ảnh nhỏ hay hơi mờ không tự động là `unreadable`: nếu trên ảnh phóng
to bạn nhận ra rõ từng ký tự, kể cả ở mã tỉnh, thì ghi `label`. Biển bị khung
ảnh cắt mất một phần (ví dụ chỉ thấy dòng dưới) là `partial`, không phải
`unreadable`.

## Định dạng `plate_text`

- Chỉ dùng chữ in hoa và chữ số. Bỏ dấu `-`, `.` và khoảng trắng.
- Giữ chữ `Đ` (có gạch ngang) khi nó xuất hiện trên biển, không đổi thành `D`.
- Biển 2 dòng: đọc dòng trên rồi đến dòng dưới, nối liền không có dấu cách.

| Trên biển | `plate_text` | `lines` |
|---|---|---|
| `51G-123.45` (1 dòng) | `51G12345` | 1 |
| `92A` / `398.26` (2 dòng) | `92A39826` | 2 |
| `29-B1` / `123.45` (xe máy) | `29B112345` | 2 |
| `30E-1234` | `30E1234` | 1 |

Chuỗi kết quả phải khớp với mẫu (regex `PLATE_FORMAT` trong `plate_tracks.py`):

```
^(?:\d{2}[A-ZĐ]{1,2}\d?\d{4,5}|[A-Z]{2}\d{4,5})$
```

Nói cách khác: 2 chữ số mã tỉnh, rồi 1–2 chữ cái, có thể thêm 1 chữ số, rồi 4–5
chữ số. Biển quân đội có dạng 2 chữ cái và 4–5 chữ số. Nếu chuỗi bạn đọc được
không khớp mẫu này, nhiều khả năng bạn đã đọc sai: hãy xem lại ảnh. Nếu vẫn không
chắc thì ghi `unreadable`.

## Các cặp ký tự dễ nhầm

Xem thật kỹ các cặp: `8/B`, `0/D`, `1/7`, `5/S`, `2/Z`, `6/G`, và `0/8`, `3/8`, `5/6`.

- **Vị trí là manh mối.** 2 ký tự đầu luôn là **chữ số** (mã tỉnh, ví dụ 29, 30,
  43, 51, 92). Ngay sau đó là **chữ cái** chỉ sê-ri. Phần cuối toàn là **chữ số**.
- **Mã tỉnh là chỗ mô hình hay đọc sai nhất**, ví dụ 92 bị đọc thành 95, 97 hay 51.
  Hãy nhìn riêng 2 ký tự đầu.
- Nếu vị trí đó buộc phải là chữ số mà hình dạng vẫn mơ hồ, như `0/8` hay `5/6`,
  thì ghi `unreadable`.

## Ảnh cùng một track

Các ảnh cùng `track_id` thường là cùng một biển. Ảnh nào nét có thể giúp bạn
hiểu ảnh mờ hơn. Tuy vậy:

- **Mỗi ảnh chỉ được gán `label` khi chính ảnh đó đọc được.** Ảnh mờ trong một
  track rõ vẫn phải ghi `unreadable`. Đừng chép nhãn sang.
- Nếu hai ảnh cùng track rõ ràng là hai biển khác nhau, cứ ghi đúng từng ảnh và
  thêm `note` là `different plate in track`. Bước kiểm tra sau sẽ xử lý.

## `note`

Ghi ngắn, tuỳ chọn, ví dụ: `blur`, `glare`, `night`, `tilted`, `blue plate`,
`yellow plate`, `different plate in track`.
