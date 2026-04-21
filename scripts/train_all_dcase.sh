#!/bin/bash
set -e
export PYTHONUNBUFFERED=1

cd /home/charbel-mezeraani/acoustic-anomaly-detection

MACHINE_TYPES=("bearing" "fan" "gearbox" "slider" "ToyCar" "ToyTrain" "valve")
MANIFEST="data/processed/manifests/dcase2024_development.csv"
FEATURE_MANIFEST="data/processed/manifests/dcase2024_development_features.csv"

echo "Starting training for all dcase2024_development machine types"
echo "Machine types: ${MACHINE_TYPES[*]}"
echo "Started at: $(date)"
echo ""

for MACHINE in "${MACHINE_TYPES[@]}"; do
    echo "========================================"
    echo "[$MACHINE] Started at: $(date)"
    echo "========================================"

    # Step 1: Preprocess (--force re-extracts with current hop_length=256)
    echo "[$MACHINE] Preprocessing..."
    python3 scripts/preprocess.py \
        --manifests "$MANIFEST" \
        --machine-types "$MACHINE" \
        --force

    NORM_STATS="data/processed/features/norm_stats_${MACHINE}.json"
    RUN_NAME="${MACHINE}_best_v1"

    # Step 2: Train
    echo "[$MACHINE] Training..."
    python3 scripts/train.py \
        --machine-types "$MACHINE" \
        --feature-manifests "$FEATURE_MANIFEST" \
        --norm-stats "$NORM_STATS" \
        --per-file-norm \
        --epochs 60 \
        --run-name "$RUN_NAME"

    # Step 3: Evaluate
    echo "[$MACHINE] Evaluating..."
    python3 scripts/evaluate.py \
        --checkpoint "artifacts/runs/${RUN_NAME}/best_model.pt" \
        --manifests "$MANIFEST" \
        --machine-types "$MACHINE"

    echo "[$MACHINE] Done at: $(date)"
    echo "[$MACHINE] Results: artifacts/runs/${RUN_NAME}/evaluation.json"
    echo ""
done

echo "========================================"
echo "All machine types completed at: $(date)"
echo "========================================"

# Print a summary of all AUC results
echo ""
echo "=== RESULTS SUMMARY ==="
python3 -c "
import json
from pathlib import Path

machine_types = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']
print(f'{'Machine':<12} {'AUC-ROC':>10} {'pAUC':>10} {'Bal.Acc':>10}')
print('-' * 45)
for m in machine_types:
    path = Path(f'artifacts/runs/{m}_best_v1/evaluation.json')
    if not path.exists():
        print(f'{m:<12} {'MISSING':>10}')
        continue
    r = json.loads(path.read_text())['per_machine'].get(m, {})
    auc  = r.get('auc_roc', float('nan'))
    pauc = r.get('pauc_fpr_le_0.1', float('nan'))
    bacc = r.get('balanced_accuracy_at_threshold', float('nan'))
    print(f'{m:<12} {auc:>10.4f} {pauc:>10.4f} {bacc:>10.4f}')
"
