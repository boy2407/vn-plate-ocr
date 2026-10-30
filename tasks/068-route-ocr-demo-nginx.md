# Route OCR demo through shared Nginx

## Goal
Route `saigondepot.<PUBLIC_IP>.sslip.io` to the existing OCR demo through the nginxgeneral VM's shared Nginx configuration, preserving all other sites and keeping the change reversible.

## Checklist
- [x] Inspect the enabled sites and `sudo nginx -T`; locate the existing SaigonDepot server block.
- [x] Back up `/etc/nginx/sites-available/vietjobs-demo.conf` to `/root/vietjobs-demo.conf.bak`.
- [x] Add `/vn-plate-ocr/` inside the existing server block, preserving `/` and `/vietjobs-kltm/`.
- [x] Run `sudo nginx -t`, then reload Nginx after it passed.
- [x] Verify the OCR path, the existing vietjobs redirect, and the direct backend.

## Progress log
- 2026-10-30: SSH to `<NGINX_VM_IP>` was unavailable, so used the existing Proxmox browser console for VM 107.
- 2026-10-30: Found `vietjobs-demo.conf` contains the existing SaigonDepot vhost with `/` proxied to `<APP_HOST_IP>:80` and `/vietjobs-kltm/` to port `7680`. The user chose `/vn-plate-ocr/` for the OCR demo on port `8766`, so the existing routes remain in place.
- 2026-10-30: Backed up the file at `/root/vietjobs-demo.conf.bak`; added a prefix proxy with response rewriting for the app's absolute `/static/`, `/api/`, and `/media/` URLs. The trailing slash in `proxy_pass http://<APP_HOST_IP>:8766/;` strips the prefix before forwarding to Flask.
- 2026-10-30: `sudo nginx -t` passed and `sudo systemctl reload nginx` ran. No symlink was removed and no other site file was changed.
- 2026-10-30: Verified `/vn-plate-ocr/` returns `200`, the direct backend on port `8766` returns `200`, and `vietjobs.<PUBLIC_IP>.sslip.io/` still returns `301` to HTTPS. Confirmed the returned HTML and JavaScript contain the `/vn-plate-ocr/` prefix.
