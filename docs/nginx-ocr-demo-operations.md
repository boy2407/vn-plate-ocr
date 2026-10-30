# Hướng dẫn vận hành route demo OCR trên Nginx

Runbook này ghi lại cách chỉnh route demo OCR trên VM 107 `nginxgeneral` qua console Proxmox, kiểm tra, reload và rollback.

## Cấu hình hiện tại

- Hostname: `saigondepot.<PUBLIC_IP>.sslip.io`
- File cấu hình: `/etc/nginx/sites-available/vietjobs-demo.conf`
- Symlink đang bật: `/etc/nginx/sites-enabled/vietjobs-demo.conf` trỏ tới file trên.
- Trang cũ ở `/` tiếp tục proxy tới `<APP_HOST_IP>:80`.
- Route `/vietjobs-kltm/` tiếp tục proxy tới `<APP_HOST_IP>:7680`.
- Demo OCR ở `/vn-plate-ocr/`, proxy tới `<APP_HOST_IP>:8766`.

Ứng dụng Flask tạo URL tuyệt đối như `/static/...`, `/api/...` và `/media/...`. Location OCR dùng `sub_filter` để thêm prefix vào các URL này trong HTML/JavaScript. Dấu `/` ở cuối `proxy_pass http://<APP_HOST_IP>:8766/;` có tác dụng bỏ prefix `/vn-plate-ocr/` trước khi gửi URI tới Flask ở root. Giữ nguyên dấu `/` đó.

## Mở console VM bằng browser

1. Mở Proxmox tại `https://localhost:8006`. Địa chỉ này cần SSH tunnel hiện có từ máy Mac tới node Proxmox.
2. Chọn VM `107 (Nginx)` trên node `hpdl38035`.
3. Mở **Console**, đăng nhập vào Ubuntu của `nginxgeneral`.
4. Chạy các lệnh bên dưới trong shell của VM. Không cần SSH trực tiếp tới `<NGINX_VM_IP>`.

## Trước khi chỉnh sửa

Sao lưu file hiện tại trước mỗi lần sửa:

```bash
sudo cp /etc/nginx/sites-available/vietjobs-demo.conf /root/vietjobs-demo.conf.bak
```

Xem các symlink đang bật, toàn bộ cấu hình Nginx đã nạp, và mọi khai báo hostname:

```bash
sudo ls -la /etc/nginx/sites-enabled/
sudo nginx -T
sudo grep -R -n 'saigondepot.<PUBLIC_IP>.sslip.io' /etc/nginx/sites-available /etc/nginx/conf.d
```

Trong trạng thái hiện tại, hostname nằm trong `vietjobs-demo.conf` và symlink riêng của file này đang bật. Không gỡ symlink đó: file còn chứa các route cũ cần giữ.

## Chỉnh route OCR

Mở file:

```bash
sudo nano /etc/nginx/sites-available/vietjobs-demo.conf
```

Thêm hoặc cập nhật location này **bên trong server block hiện tại**, cùng cấp với hai location `/` và `/vietjobs-kltm/`, trước dấu `}` cuối của server block:

```nginx
    location /vn-plate-ocr/ {
        proxy_pass http://<APP_HOST_IP>:8766/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Accept-Encoding "";
        proxy_buffering off;
        proxy_read_timeout 900s;
        proxy_send_timeout 900s;
        sub_filter_types text/css application/javascript text/javascript;
        sub_filter '="/' '="/vn-plate-ocr/';
        sub_filter '/api/' '/vn-plate-ocr/api/';
        sub_filter '/media/' '/vn-plate-ocr/media/';
        sub_filter_once off;
    }
```

`text/html` đã được Nginx đưa vào loại nội dung của `sub_filter` mặc định; không thêm lại vào `sub_filter_types` để tránh cảnh báo MIME trùng. `proxy_buffering off` giữ luồng MJPEG `/api/jobs/<id>/live` chạy trực tiếp; timeout 900 giây cho phép tác vụ kéo dài tối đa 10 phút.

Lưu nano bằng `Ctrl+O`, `Enter`, rồi thoát bằng `Ctrl+X`.

## Test và reload

Luôn test trước:

```bash
sudo nginx -t
```

Chỉ khi kết quả có `syntax is ok` và `test is successful` mới reload:

```bash
sudo systemctl reload nginx
```

Không restart Nginx và không chạy Certbot cho thay đổi route này. Có thể xuất hiện cảnh báo cấu hình cũ ở các site khác; không sửa các file đó trong lúc cập nhật demo OCR.

## Kiểm tra sau reload

Chạy từ VM 107:

```bash
curl -sI -H 'Host: saigondepot.<PUBLIC_IP>.sslip.io' http://127.0.0.1/vn-plate-ocr/
curl -sI -H 'Host: vietjobs.<PUBLIC_IP>.sslip.io' http://127.0.0.1/
curl -sI http://<APP_HOST_IP>:8766/
```

Kết quả đã xác nhận ngày 2026-10-30:

- `/vn-plate-ocr/`: `HTTP/1.1 200 OK`, `Server: nginx/1.24.0 (Ubuntu)`.
- `vietjobs.../`: `HTTP/1.1 301 Moved Permanently`, `Location: https://vietjobs.<PUBLIC_IP>.sslip.io/`.
- Backend `<APP_HOST_IP>:8766/`: `HTTP/1.1 200 OK`, `Server: Werkzeug/3.1.9 Python/3.11.18`.

Có thể kiểm tra thêm rằng HTML và JavaScript được viết lại theo prefix:

```bash
curl -s -o /tmp/ocr.html -H 'Host: saigondepot.<PUBLIC_IP>.sslip.io' http://127.0.0.1/vn-plate-ocr/
grep -o 'vn-plate-ocr' /tmp/ocr.html
curl -s -o /tmp/ocr-app.js -H 'Host: saigondepot.<PUBLIC_IP>.sslip.io' http://127.0.0.1/vn-plate-ocr/static/app.js
grep -o 'vn-plate-ocr/api' /tmp/ocr-app.js
grep -o 'vn-plate-ocr/media' /tmp/ocr-app.js
```

## Rollback nếu `nginx -t` thất bại

Khôi phục bản sao lưu:

```bash
sudo cp /root/vietjobs-demo.conf.bak /etc/nginx/sites-available/vietjobs-demo.conf
sudo nginx -t
```

Không reload khi test cấu hình mới thất bại. File cấu hình Nginx đang chạy tiếp tục có hiệu lực cho tới khi reload thành công. Không cần bật lại symlink trong quy trình này vì không gỡ symlink nào.

## Đổi path hoặc port sau này

- Nếu đổi port backend, cập nhật port trong `proxy_pass` rồi test và reload.
- Nếu đổi prefix `/vn-plate-ocr/`, cập nhật đồng bộ location, giá trị đích của rule HTML `sub_filter`, và hai rule `/api/` và `/media/`.
- Luôn sao lưu, chạy `sudo nginx -t`, và chỉ reload khi test pass.
