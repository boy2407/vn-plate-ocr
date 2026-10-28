# Retrain with stronger augmentation (run A) and a PARSeq decoder (run B)

## Goal
Measure, on the unchanged train/val/test split, how much stronger augmentation helps the current AR decoder (run A), and how much a PARSeq-style permutation decoder adds on top with the same augmentation (run B).

## Checklist
- [x] Add training augmentations: HSV/brightness, sun glare, low-resolution downscale.
- [x] Add training options: label smoothing, LR warmup, checkpoint selection by val greedy exact.
- [x] Implement PARSeq-style decoder (position queries, permutation masks, AR + refinement decode).
- [x] Unit tests for augmentations and PARSeq masks/shapes; ruff + pytest.
- [x] Run A (AR) and run B (PARSeq) on desktop GPU with identical settings.
- [x] Compare A vs B vs run-20261028-01: exact, CER, @0.99, letter/digit confusions, old high-confidence errors, McNemar.

## Done criteria
Both runs finish; a comparison report on dev and test exists with significance tests.

## Progress log
- 2026-10-28: Started. User chose 2 runs (A: AR + aug, B: PARSeq + aug) and to keep the current split.
- 2026-10-28: Added augmentations (downscale p=0.3, HSV/contrast p=0.5, glare p=0.15, existing motion blur), `--label-smoothing`, `--warmup-epochs`, `--select-by greedy_exact` (CER tie-break), `--decoder parseq` (PlateOCRParseq: position queries, K=6 orders incl. L2R/R2L flips, AR + 1 cloze refinement). Eval scripts load either class via `build_model`. Tests: custom_alpr/tests/test_custom_ocr.py 9 passed on desktop; ruff clean. Smoke runs of both decoders passed.
- 2026-10-28: Launched run A `run-20261028-02-ar-aug` then run B `run-20261028-03-parseq-aug` on desktop (batch 16, workers 4, seed 20261028, epochs 60, patience 10, lr 3e-4, warmup 2, label smoothing 0.1).
- 2026-10-28: Run A finished all 60 epochs (no early stop), best epoch 58. Test exact 0.8795 (run01 0.8263; fixed 80 / broke 16, McNemar p=2e-11), val 0.8711. Label smoothing caps confidence near 0.92, so the 0.99 threshold accepts nothing; compare at equal coverage instead: test @85% coverage 98.3% exact (17 wrong) vs run01 94.2% (59 wrong). Run B still training.
- 2026-10-28: Run B early-stopped at epoch 51, best epoch 41. Test exact 0.8753 vs A 0.8795 (B fixes 24, breaks 29, McNemar p=0.58); val 0.8736 vs A 0.8711 (p=0.80). No significant difference. B has fewer letter<->digit confusions (test 26 vs 38, val 25 vs 28) and fewer insertions (17 vs 31), but more letter<->letter (58 vs 47). Both still read 13346HZ as H2, 60HB1851 as H8, 83PT as P1. mb_310 (glare) now correct in A and B. Done.
