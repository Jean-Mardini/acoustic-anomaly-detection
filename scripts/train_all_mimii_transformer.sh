#!/bin/bash
set -e
export PYTHONUNBUFFERED=1

cd /home/charbel-mezeraani/acoustic-anomaly-detection

MACHINE_TYPES=("fan" "gearbox" "pump" "slider" "valve")
MANIFEST="data/processed/manifests/mimii_due.csv"
FEATURE_MANIFEST="data/processed/manifests/mimii_due_features.csv"

echo "Starting Transformer AE training for all MIMII-DUE machine types"
echo "Machine types: ${MACHINE_TYPES[*]}"
echo "Started at: $(date)"
echo ""

for MACHINE in "${MACHINE_TYPES[@]}"; do
    echo "========================================"
    echo "[$MACHINE] Started at: $(date)"
    echo "========================================"

    NORM_STATS="data/processed/features/norm_stats_mimii_${MACHINE}.json"
    RUN_NAME="mimii_${MACHINE}_transformer_v1"
    RESUME_CKPT="artifacts/runs/${RUN_NAME}/resume_checkpoint.pt"
    BEST_CKPT="artifacts/runs/${RUN_NAME}/best_model.pt"

    # Skip if already fully trained
    if [ -f "$BEST_CKPT" ] && [ ! -f "$RESUME_CKPT" ]; then
        echo "[$MACHINE] Already trained, skipping to evaluation..."
    else
        # Only preprocess if norm stats don't exist yet
        if [ ! -f "$NORM_STATS" ]; then
            echo "[$MACHINE] Preprocessing..."
            python3 scripts/preprocess.py \
                --manifests "$MANIFEST" \
                --machine-types "$MACHINE" \
                --force
            if [ -f "data/processed/features/norm_stats_${MACHINE}.json" ]; then
                mv "data/processed/features/norm_stats_${MACHINE}.json" "$NORM_STATS"
            fi
        else
            echo "[$MACHINE] Norm stats found, skipping preprocessing."
        fi

        # Resume if checkpoint exists, otherwise start fresh
        RESUME_FLAG=""
        if [ -f "$RESUME_CKPT" ]; then
            echo "[$MACHINE] Resuming training from checkpoint..."
            RESUME_FLAG="--resume"
        else
            echo "[$MACHINE] Training Transformer AE from scratch..."
        fi

        python3 scripts/train.py \
            --model transformer \
            --machine-types "$MACHINE" \
            --manifests "$MANIFEST" \
            --feature-manifests "$FEATURE_MANIFEST" \
            --norm-stats "$NORM_STATS" \
            --per-file-norm \
            --epochs 60 \
            --run-name "$RUN_NAME" \
            $RESUME_FLAG

        # Remove resume checkpoint after successful training
        rm -f "$RESUME_CKPT"
    fi

    echo "[$MACHINE] Evaluating with Mahalanobis..."
    python3 scripts/evaluate.py \
        --checkpoint "$BEST_CKPT" \
        --manifests "$MANIFEST" \
        --machine-types "$MACHINE" \
        --scorer mahalanobis \
        --out-json "artifacts/runs/${RUN_NAME}/evaluation_mahalanobis.json"

    echo "[$MACHINE] Evaluating with GMM..."
    python3 scripts/evaluate.py \
        --checkpoint "$BEST_CKPT" \
        --manifests "$MANIFEST" \
        --machine-types "$MACHINE" \
        --scorer gmm \
        --gmm-components 10 \
        --out-json "artifacts/runs/${RUN_NAME}/evaluation_gmm.json"

    echo "[$MACHINE] Evaluating with Domain GMM..."
    python3 scripts/evaluate.py \
        --checkpoint "$BEST_CKPT" \
        --manifests "$MANIFEST" \
        --machine-types "$MACHINE" \
        --scorer domain_gmm \
        --gmm-components 10 \
        --out-json "artifacts/runs/${RUN_NAME}/evaluation_domain_gmm.json"

    echo "[$MACHINE] Evaluating with TTA GMM..."
    python3 scripts/evaluate.py \
        --checkpoint "$BEST_CKPT" \
        --manifests "$MANIFEST" \
        --machine-types "$MACHINE" \
        --scorer tta_gmm \
        --gmm-components 10 \
        --out-json "artifacts/runs/${RUN_NAME}/evaluation_tta_gmm.json"

    echo "[$MACHINE] Done at: $(date)"
    echo ""
done

echo "========================================"
echo "All MIMII-DUE machine types completed at: $(date)"
echo "========================================"

echo ""
echo "=== RESULTS SUMMARY ==="
python3 -c "
import json
from pathlib import Path

machines = ['fan', 'gearbox', 'pump', 'slider', 'valve']
print(f'{'Machine':<12} {'Maha':>8} {'GMM':>8} {'DomGMM':>8} {'TTA':>8}')
print('-' * 50)
for m in machines:
    scores = {}
    for s,f in [('Maha','mahalanobis'),('GMM','gmm'),('DomGMM','domain_gmm'),('TTA','tta_gmm')]:
        p = Path(f'artifacts/runs/mimii_{m}_transformer_v1/evaluation_{f}.json')
        r = json.loads(p.read_text())['per_machine'].get(m, {}) if p.exists() else {}
        scores[s] = r.get('auc_roc', float('nan'))
    print(f'{m:<12} {scores[\"Maha\"]:>8.4f} {scores[\"GMM\"]:>8.4f} {scores[\"DomGMM\"]:>8.4f} {scores[\"TTA\"]:>8.4f}')
"
