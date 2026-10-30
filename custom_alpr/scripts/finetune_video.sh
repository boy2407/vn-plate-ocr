#!/usr/bin/env bash
# Fine-tune OCR model E on Kaggle + person-labelled video crops (see finetune_mix.py).
# Run from the repo root on a CUDA machine; writes everything under $OUT.
#   bash custom_alpr/scripts/finetune_video.sh
set -euo pipefail

E=custom_alpr/outputs/train-custom-ocr/run-20261029-09-parseq-f-compress/best.pt
K=custom_alpr/data/ocr/kaggle-v9-prepared
D=${D:-custom_alpr/data/ocr/finetune-video-v1}
OUT=${OUT:-custom_alpr/outputs/finetune-video-v1}
FOLDS=${FOLDS:-"train-saigon-drive train-danang-ptz all"}
mkdir -p "$OUT/logs" "$OUT/eval"

# Model E's own training flags, so fine-tuning keeps its augmentations; low LR, few epochs.
TRAIN=(
  --decoder parseq --label-smoothing 0.1 --noise-augment --blue-plate-augment
  --strong-downscale --view-tilt --compression-augment --balance-rare-letters
  --select-by greedy_exact --init-checkpoint "$E"
  --learning-rate "${LR:-5e-5}" --warmup-epochs 0.5 --epochs "${EPOCHS:-12}" --patience 4
  --workers "${WORKERS:-8}" --val "$K/val.csv" --test "$K/test.csv"
)

evaluate() {  # checkpoint manifest name
  [ -f "$OUT/eval/$3/character-fp-report.json" ] && return
  python -m custom_alpr.evaluate_custom_ocr --checkpoint "$1" --manifest "$2" \
    --output-dir "$OUT/eval/$3" --workers 4 > "$OUT/logs/eval-$3.log"
}

evaluate "$E" "$K/test.csv" E-on-kaggle-test
if [ -f "$D/base-test.csv" ]; then evaluate "$E" "$D/base-test.csv" E-on-base-test; fi
for fold in $FOLDS; do
  evaluate "$E" "$D/$fold/holdout.csv" "E-on-$fold-holdout"
  if [ ! -f "$OUT/$fold/best.pt" ]; then
    rm -rf "${OUT:?}/$fold"
    python -m custom_alpr.train_custom_ocr "${TRAIN[@]}" --train "$D/$fold/train.csv" \
      --output "$OUT/$fold" 2>&1 | tee "$OUT/logs/train-$fold.log"
  fi
  evaluate "$OUT/$fold/best.pt" "$D/$fold/holdout.csv" "$fold-on-holdout"
  evaluate "$OUT/$fold/best.pt" "$K/test.csv" "$fold-on-kaggle-test"
  if [ -f "$D/base-test.csv" ]; then evaluate "$OUT/$fold/best.pt" "$D/base-test.csv" "$fold-on-base-test"; fi
done

python - "$OUT/eval" <<'EOF'
import json, sys
from pathlib import Path
for report in sorted(Path(sys.argv[1]).glob("*/character-fp-report.json")):
    data = json.loads(report.read_text())
    print(f"{report.parent.name:40s} exact={data['exact_plate_accuracy']:.3f} "
          f"cer={data['cer']:.3f} n={data['test_samples']}")
EOF
echo FINETUNE_DONE
