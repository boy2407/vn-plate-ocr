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
     custom_alpr/custom_ocr.py custom_alpr/ocr_onnx.py custom_alpr/preprocessing.py \
     custom_alpr/plate_tracks.py custom_alpr/plate_pose.py custom_alpr/pose_data.py \
     custom_alpr/yolo_export.py custom_alpr/speed_benchmark.py \
     custom_alpr/
COPY custom_alpr/web custom_alpr/web
# Same relative paths as DEFAULT_OCR / DEFAULT_DETECTOR, so no flags are needed.
COPY custom_alpr/outputs/train-custom-ocr/run-20261030-10-parseq-e-v1-video/best.pt \
     custom_alpr/outputs/train-custom-ocr/run-20261030-10-parseq-e-v1-video/best.pt
COPY custom_alpr/outputs/vast-yolo-20261028/remote-results/moderate-s42/weights/best.pt \
     custom_alpr/outputs/vast-yolo-20261028/remote-results/moderate-s42/weights/best.pt
# Four-corner pose model (task073); web_app reads straightened crops when it finds the plate.
COPY custom_alpr/outputs/pose-073/best.pt custom_alpr/outputs/pose-073/best.pt

# OCR runs through ONNX Runtime; export the checkpoint once at build time.
RUN python -m custom_alpr.ocr_onnx
# Detector and pose run through OpenVINO; web_app picks the *_openvino_model folders up.
RUN YOLO_CONFIG_DIR=/tmp/ultralytics python -m custom_alpr.yolo_export

ENV PYTHONUNBUFFERED=1 YOLO_CONFIG_DIR=/tmp/ultralytics
EXPOSE 8766
CMD ["python", "-m", "custom_alpr.web_app", "--host", "0.0.0.0", "--port", "8766", "--device", "cpu", \
     "--ocr-checkpoint", "custom_alpr/outputs/train-custom-ocr/run-20261030-10-parseq-e-v1-video/best.onnx"]
