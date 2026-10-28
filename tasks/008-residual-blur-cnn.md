# Add residual blur downsampling to custom CNN

## Goal
Update the custom OCR CNN to use residual feature paths and anti-aliased downsampling.

## Checklist
- [x] Replace strided convolutions with residual blocks and BlurPool downsampling.
- [x] Preserve the existing CNN output dimensions for the Transformer interface.
- [x] Update the architecture diagram and document the effect/limits.
- [x] Run available code checks.

## Progress log
- 2026-10-27: Replaced CNN blocks with residual main/skip paths and fixed depthwise blur downsampling. Python syntax and diff checks pass; runtime tests unavailable in the host environment (Python 3.14, no pytest/Ruff).
