# Train PlateOCR2D on the desktop GPU

## Goal
Transfer the custom OCR code and prepared Kaggle dataset to the Windows desktop and run a full GPU training job.

## Checklist
- [x] Package and transfer model code, manifests, and source images over SSH.
- [x] Create a Python environment and install GPU-enabled PyTorch plus image dependencies.
- [x] Confirm CUDA is available and run a one-batch smoke step.
- [x] Run full train/validation/test, preserve logs and best checkpoint on the desktop.
- [x] Record metrics, paths, and the final run status.

## Done criteria
A full training run completes on the desktop GPU, and the checkpoint, history, and held-out test metrics are available there.

## Progress log
- 2026-10-28: User authorized training on the SSH desktop. Confirmed Windows host `desktop-68dps08`, Python 3.12.10, NVIDIA GeForce RTX 5060 with 8,151 MiB VRAM, and about 238 GiB free on C:.
- 2026-10-28: Transferred the project training bundle and dataset by SSH/SCP. Installed PyTorch 2.14.1+cu130 and verified CUDA on RTX 5060. One-batch smoke passed. Full run: batch 16, 4 workers, seed 20261028, early stop at epoch 41, best epoch 33 with validation loss 0.1941. Held-out greedy test metrics: exact accuracy 0.826, CER 0.062. Best checkpoint SHA-256 matched after copying to the local workspace; loaded and generated a matching sample transcription.
- Artifacts on desktop: `C:\Users\trnghia\vn-plate-ocr\custom_alpr\outputs\train-custom-ocr\run-20261028-01`; workspace copy: `custom_alpr/outputs/train-custom-ocr/run-20261028-01/`.
