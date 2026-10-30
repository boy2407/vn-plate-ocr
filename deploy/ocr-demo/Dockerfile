FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# CPU-only torch: the Proxmox hosts have no GPU, and the CUDA wheel is ~2 GB larger.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY deploy/ocr-demo/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY custom_alpr/__init__.py custom_alpr/web_app.py custom_alpr/video_demo.py \
     custom_alpr/custom_ocr.py custom_alpr/preprocessing.py custom_alpr/plate_tracks.py \
     custom_alpr/
COPY custom_alpr/web custom_alpr/web
# Same relative paths as DEFAULT_OCR / DEFAULT_DETECTOR, so no flags are needed.
COPY custom_alpr/outputs/train-custom-ocr/run-20261029-09-parseq-f-compress/best.pt \
     custom_alpr/outputs/train-custom-ocr/run-20261029-09-parseq-f-compress/best.pt
COPY custom_alpr/outputs/vast-yolo-20261028/remote-results/moderate-s42/weights/best.pt \
     custom_alpr/outputs/vast-yolo-20261028/remote-results/moderate-s42/weights/best.pt

ENV PYTHONUNBUFFERED=1 YOLO_CONFIG_DIR=/tmp/ultralytics
EXPOSE 8766
CMD ["python", "-m", "custom_alpr.web_app", "--host", "0.0.0.0", "--port", "8766", "--device", "cpu"]
