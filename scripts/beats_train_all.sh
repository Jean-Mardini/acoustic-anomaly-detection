#!/bin/bash
# Train BEATs+LoRA+MGA+DLCL separately on DCASE and MIMII-DUE,
# then evaluate both. Runs fully sequentially in one shot.

set -e
cd "$(dirname "$0")/.."

BEATS_CKPT="models/BEATs_iter3_plus_AS2M.pt"
LOG_DIR="logs"
mkdir -p "$LOG_DIR" artifacts

# ── Shared hyperparameters ────────────────────────────────────────────────────
COMMON="
  --beats-ckpt $BEATS_CKPT
  --lora-rank 32 --lora-alpha 32.0
  --mga-bottleneck 64
  --temperature 0.07 --frame-weight 0.5 --num-frames 10
  --time-mask 0.15 --feat-mask 0.15
  --batch-size 4 --grad-accum 4
  --epochs 30 --lr-lora 2e-4 --lr-mga 5e-4 --lr-head 1e-3
  --early-stopping 10
"

# ── 1. DCASE (dev + additional, 16 machine types) ─────────────────────────────
echo "============================================================"
echo "  TRAINING: DCASE dev + additional"
echo "============================================================"
python3 scripts/beats_train.py \
  --manifests \
    data/processed/manifests/dcase2024_development.csv \
    data/processed/manifests/dcase2024_additional.csv \
  --out-dir artifacts/beats_lora_dcase \
  $COMMON \
  2>&1 | tee "$LOG_DIR/beats_dcase_train.log"

echo ""
echo "============================================================"
echo "  EVALUATING: DCASE dev test set"
echo "============================================================"
python3 scripts/beats_evaluate.py \
  --beats-ckpt "$BEATS_CKPT" \
  --lora-ckpt artifacts/beats_lora_dcase/beats_lora.pt \
  --manifests data/processed/manifests/dcase2024_development.csv \
  --calibrate-split train --eval-split test \
  --gmm-components 32 \
  --out-json artifacts/beats_lora_dcase_results.json \
  2>&1 | tee "$LOG_DIR/beats_dcase_eval.log"

echo ""

# ── 2. MIMII-DUE (5 machine types) ───────────────────────────────────────────
echo "============================================================"
echo "  TRAINING: MIMII-DUE"
echo "============================================================"
python3 scripts/beats_train.py \
  --manifests \
    data/processed/manifests/mimii_due.csv \
  --out-dir artifacts/beats_lora_mimii \
  $COMMON \
  2>&1 | tee "$LOG_DIR/beats_mimii_train.log"

echo ""
echo "============================================================"
echo "  EVALUATING: MIMII-DUE test set"
echo "============================================================"
python3 scripts/beats_evaluate.py \
  --beats-ckpt "$BEATS_CKPT" \
  --lora-ckpt artifacts/beats_lora_mimii/beats_lora.pt \
  --manifests data/processed/manifests/mimii_due.csv \
  --calibrate-split train --eval-split test \
  --gmm-components 32 \
  --out-json artifacts/beats_lora_mimii_results.json \
  2>&1 | tee "$LOG_DIR/beats_mimii_eval.log"

echo ""
echo "============================================================"
echo "  ALL DONE"
echo "============================================================"

# Print summary
echo ""
echo "=== DCASE Results ==="
python3 -c "
import json
r = json.load(open('artifacts/beats_lora_dcase_results.json'))
for m, v in r['per_machine'].items():
    if 'auc_roc' in v:
        print(f'  {m:20s} AUC={v[\"auc_roc\"]:.4f}')
print(f'  {\"MEAN\":20s} AUC={r[\"mean_auc\"]:.4f}')
"

echo ""
echo "=== MIMII-DUE Results ==="
python3 -c "
import json
r = json.load(open('artifacts/beats_lora_mimii_results.json'))
for m, v in r['per_machine'].items():
    if 'auc_roc' in v:
        print(f'  {m:20s} AUC={v[\"auc_roc\"]:.4f}')
print(f'  {\"MEAN\":20s} AUC={r[\"mean_auc\"]:.4f}')
"
