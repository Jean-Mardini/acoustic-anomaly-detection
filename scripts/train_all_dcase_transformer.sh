#!/bin/bash
set -e
export PYTHONUNBUFFERED=1

cd /home/charbel-mezeraani/acoustic-anomaly-detection

MACHINE_TYPES=("bearing" "fan" "gearbox" "slider" "ToyCar" "ToyTrain" "valve")
MANIFEST="data/processed/manifests/dcase2024_development.csv"
FEATURE_MANIFEST="data/processed/manifests/dcase2024_development_features.csv"

echo "Starting Transformer AE training for all DCASE2024 machine types"
echo "Machine types: ${MACHINE_TYPES[*]}"
echo "Started at: $(date)"
echo ""

for MACHINE in "${MACHINE_TYPES[@]}"; do
    echo "========================================"
    echo "[$MACHINE] Started at: $(date)"
    echo "========================================"

    echo "[$MACHINE] Preprocessing..."
    python3 scripts/preprocess.py \
        --manifests "$MANIFEST" \
        --machine-types "$MACHINE" \
        --force

    NORM_STATS="data/processed/features/norm_stats_${MACHINE}.json"
    RUN_NAME="${MACHINE}_transformer_v1"

    echo "[$MACHINE] Training Transformer AE..."
    python3 scripts/train.py \
        --model transformer \
        --machine-types "$MACHINE" \
        --feature-manifests "$FEATURE_MANIFEST" \
        --norm-stats "$NORM_STATS" \
        --per-file-norm \
        --epochs 60 \
        --run-name "$RUN_NAME"

    echo "[$MACHINE] Evaluating with Mahalanobis..."
    python3 scripts/evaluate.py \
        --checkpoint "artifacts/runs/${RUN_NAME}/best_model.pt" \
        --manifests "$MANIFEST" \
        --machine-types "$MACHINE" \
        --scorer mahalanobis \
        --out-json "artifacts/runs/${RUN_NAME}/evaluation_mahalanobis.json"

    echo "[$MACHINE] Evaluating with GMM..."
    python3 scripts/evaluate.py \
        --checkpoint "artifacts/runs/${RUN_NAME}/best_model.pt" \
        --manifests "$MANIFEST" \
        --machine-types "$MACHINE" \
        --scorer gmm \
        --gmm-components 10 \
        --out-json "artifacts/runs/${RUN_NAME}/evaluation_gmm.json"

    echo "[$MACHINE] Done at: $(date)"
    echo ""
done

echo "========================================"
echo "All DCASE machine types completed at: $(date)"
echo "========================================"

echo ""
echo "=== RESULTS SUMMARY ==="
python3 -c "
import json
from pathlib import Path

machines = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']
print(f'{'Machine':<12} {'Maha AUC':>10} {'Maha pAUC':>10} {'GMM AUC':>10} {'GMM pAUC':>10}')
print('-' * 55)
for m in machines:
    p1 = Path(f'artifacts/runs/{m}_transformer_v1/evaluation_mahalanobis.json')
    p2 = Path(f'artifacts/runs/{m}_transformer_v1/evaluation_gmm.json')
    if not p1.exists() or not p2.exists():
        print(f'{m:<12} MISSING')
        continue
    r1 = json.loads(p1.read_text())['per_machine'].get(m, {})
    r2 = json.loads(p2.read_text())['per_machine'].get(m, {})
    auc1  = r1.get('auc_roc', float('nan'))
    pauc1 = r1.get('pauc_fpr_le_0.1', float('nan'))
    auc2  = r2.get('auc_roc', float('nan'))
    pauc2 = r2.get('pauc_fpr_le_0.1', float('nan'))
    print(f'{m:<12} {auc1:>10.4f} {pauc1:>10.4f} {auc2:>10.4f} {pauc2:>10.4f}')
"
