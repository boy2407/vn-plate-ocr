# 054 Train run D on a rented vast.ai GPU

## Goal
Desktop went offline during run D (epoch ~8). Re-run run D on a vast.ai instance, detached from SSH, and bring results back for comparison with run C.

## Checklist
- [x] User rents instance (on-demand, PyTorch CUDA image, >= 8 CPU cores, >= 16 GB VRAM, >= 30 GB disk)
- [x] Pack code + data (custom_alpr code, data/ocr/kaggle-v9-prepared, data/raw/kaggle-vietnamese-license-plate-ocr, ~65 MB) and upload with scp
- [x] Install Pillow/opencv/numpy, run pytest custom_alpr/tests/test_custom_ocr.py -q on the instance
- [x] Short smoke run (1 epoch) to measure seconds per epoch
- [x] Launch run D inside tmux/nohup with the exact run D flags
- [x] Download best.pt, history.csv, log; stop/destroy instance
- [x] Evaluate D vs C (dev/test, real-only, <25 px, McNemar, blue/rotated/rare-letter groups, 21 clear-wrong dev images, noise robustness)

## Done criteria
run-20261028-05-parseq-d outputs exist locally and D vs C comparison is reported.

## Progress log
- 2026-10-28: Plan written; waiting for user to rent the instance.
- 2026-10-28: Rented vast.ai RTX 5060 Ti (torch 2.14.1+cu132), 19 tests pass, trained in tmux 17:03-18:09 (~65 s/epoch, main-process CPU bound, GPU ~35%), 60 epochs. Outputs downloaded to custom_alpr/outputs/train-custom-ocr/run-20261028-05-parseq-d and noise-robustness-test-C-D.json. User still has to destroy the instance.
