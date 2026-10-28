# Decoder của PlateOCR2D hoạt động như thế nào

Đây là mô tả của kiến trúc **đã viết trong code nhưng chưa được train**. Các ví dụ về vùng chú ý là khả năng mô hình có thể học, không phải hành vi đã được kiểm chứng.

## Đầu vào

- Một biển số RGB chuẩn hóa: `[B, 3, 64, 128]`.
- CNN và encoder tạo **128 token ảnh**, sắp từ lưới `4 × 32`. Mỗi token có 192 chiều; memory của encoder là `[B, 128, 192]`. Positional embedding hàng/cột cho biết vị trí trong lưới. Một token không tương đương một ký tự.
- Decoder nhận `BOS` rồi các ký tự đã có. Embedding ký tự cộng embedding vị trí tạo `[B, L, 192]`, với `L` là số token văn bản trong prefix.

## Mỗi tầng Decoder

Mô hình có **2 tầng Decoder**. Mỗi tầng chạy theo thứ tự:

1. **Masked self-attention, 4 head:** mỗi vị trí văn bản nhìn `BOS` và các vị trí trước nó; causal mask chặn ký tự tương lai. Khi prefix dài `L`, có `L` query văn bản, không phải một query cho toàn chuỗi.
2. **Cross-attention, 4 head:** query đến từ trạng thái văn bản sau self-attention. Key (`K`) và value (`V`) đến từ **cả 128 token ảnh**. Mỗi head chiếu 192 chiều sang 48 chiều, so một query với 128 key, rồi Softmax tạo 128 trọng số. Tổng có trọng số của 128 value cho một vector 48 chiều. Bốn vector được ghép thành 192 chiều và chiếu qua lớp đầu ra của attention.
3. **Feed-forward:** xử lý từng vị trí theo chiều `192 → 768 → 192`, cùng residual connection và LayerNorm trong tầng Transformer.

Cross-attention không lấy cứng token ảnh có điểm cao nhất hoặc mặc định lấy token nằm kế bên. Nó có thể chia trọng số cho nhiều vùng. Cả 4 head đều có thể nhìn toàn bộ 128 token; mỗi head có các phép chiếu và trọng số chú ý riêng. Chúng *có thể* học cách chú ý khác nhau, nhưng không được gán trước nhiệm vụ.

## Từ 192 chiều thành ký tự

Sau tầng Decoder thứ hai, LayerNorm và Dense (`Linear(192, 40)`) tạo **40 logits** cho mỗi vị trí văn bản. Theo dạng toán học, một vector `[192]` nhân ma trận `[192, 40]` rồi cộng bias `[40]` cho ra `[40]`. Dense học các trọng số này khi train; bốn head thu thập thông tin, còn Dense chấm ký hiệu từ thông tin đã xử lý.

Bộ 40 token gồm: `PAD`, `BOS`, `EOS`, 10 chữ số, 26 chữ cái `A–Z`, và `Đ`. Vì vậy đây là **40 mục từ vựng**, không phải 40 ký tự biển số.

```text
Prefix [BOS, 3, 0] → 3 vị trí query ở mỗi tầng Decoder
                    → mỗi query nhìn 128 token ảnh
                    → logits [3, 40]
                    → lúc inference chỉ dùng logits ở vị trí cuối
                    → chọn ký hiệu tiếp theo, ví dụ A
```

Trong `generate`, code thêm ký hiệu vừa chọn vào prefix, chạy lại Decoder với prefix dài hơn, và dừng ở `EOS` hoặc giới hạn độ dài. Hai tầng Decoder không tương ứng hai ký tự. Ma trận memory `[128, 192]` vẫn được dùng ở từng bước đoán; nó không bị biến vĩnh viễn thành một vector duy nhất.

## Trọng số học như thế nào

Khi train, `encode_target` tạo `BOS + chuỗi đúng + EOS`. Decoder nhận chuỗi dịch sang phải (`target_ids[:, :-1]`) và phải dự đoán chuỗi dịch sang trái (`target_ids[:, 1:]`). Cross-entropy bỏ qua vị trí `PAD`. Sai số được truyền ngược để cập nhật convolution, encoder, embedding, self-attention, cross-attention và Dense. Kernel BlurPool là bộ lọc cố định, không được học. Hiện chưa có Dataset/collate, train loop hoặc checkpoint cho mạng tự xây này.
