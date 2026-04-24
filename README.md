# Acoustic Anomaly Detection

> Unsupervised anomaly detection for industrial machines — benchmarked on **DCASE 2024 Task 2** and **MIMII-DUE**, progressing from a classical autoencoder baseline to the DCASE 2024 winning architecture.

---

## Overview

This project implements and benchmarks a full progression of acoustic anomaly detection methods, from handcrafted spectrogram features to fine-tuned audio foundation models. All models are trained exclusively on normal machine audio and evaluated by how well they separate normal from anomalous sounds at test time — no anomaly labels are ever seen during training.

---

## Results

### DCASE 2024 Task 2 — Development Set (AUC-ROC)

| Model | ToyCar | ToyTrain | Bearing | Fan | Gearbox | Slider | Valve | **Mean** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Convolutional Autoencoder | 0.490 | 0.463 | 0.630 | 0.525 | 0.517 | 0.551 | 0.396 | 0.510 |
| Transformer Autoencoder | 0.443 | 0.555 | 0.623 | 0.532 | 0.556 | 0.493 | 0.392 | 0.513 |
| WavLM + GMM | 0.428 | 0.533 | 0.541 | 0.539 | 0.535 | 0.560 | 0.461 | 0.514 |
| WavLM + Mahalanobis | 0.470 | 0.584 | 0.566 | 0.557 | 0.532 | 0.569 | 0.490 | 0.538 |
| WavLM + LOF | 0.483 | 0.563 | 0.640 | 0.542 | 0.523 | 0.571 | 0.564 | 0.555 |
| **BEATs (frozen)** | **0.504** | **0.709** | **0.609** | **0.618** | **0.776** | **0.654** | **0.597** | **0.638** |
| BEATs + LoRA + MGA + DLCL | — | — | — | — | — | — | — | *training* |

### MIMII-DUE (AUC-ROC)

| Model | Fan | Gearbox | Pump | Slider | Valve | **Mean** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Transformer Autoencoder | 0.507 | 0.583 | 0.547 | 0.581 | 0.513 | 0.546 |
| BEATs + LoRA + MGA + DLCL | — | — | — | — | — | *training* |

---

## Methods

### Stage 1 · Convolutional Autoencoder

A reconstruction-based anomaly detector on log-mel spectrograms.

- **Features** — log-mel (`n_fft=1024`, `hop=512`, `n_mels=128`), z-score normalized on normal training data only
- **Model** — symmetric Conv encoder/decoder, MSE reconstruction loss
- **Scoring** — mean window reconstruction error → per-machine GMM calibration
- **Scripts** — `scripts/train.py`, `scripts/evaluate.py`

### Stage 2 · Transformer Autoencoder

Patch-based self-attention encoder replacing the convolutional backbone.

- **Architecture** — spectrogram patches → linear projection → multi-head self-attention → reconstruction
- **Same scoring pipeline** as Conv AE
- **Scripts** — `scripts/train.py --model transformer`, `scripts/evaluate.py`

### Stage 3 · WavLM Frozen Features

Microsoft WavLM Base Plus (pretrained on 60 000 hours) as a frozen feature extractor.

- **Features** — 768-dim frame embeddings, mean-pooled over time
- **Scorers benchmarked** — GMM, Mahalanobis, LOF
- **Scripts** — `scripts/wavlm_evaluate.py`

### Stage 4 · BEATs Frozen Features

Microsoft BEATs (92 M params, pretrained on AudioSet 2 M clips) — strongest result without fine-tuning.

- **Features** — 768-dim BEATs embeddings, mean-pooled over ~500 frames
- **Scorer** — per-machine GMM (32 components, full covariance)
- **Calibrated thresholds** — 95th / 99th percentile of normal training scores
- **Deployment** — fitted GMMs exported in `beats_frozen_export/` for direct use in inference apps
- **Scripts** — `scripts/beats_evaluate.py`, `scripts/beats_export_scorers.py`

### Stage 5 · BEATs + LoRA + MGA + DLCL

Full reproduction of the **DCASE 2024 Task 2 winning system** (without ensembling).

| Component | Detail |
|---|---|
| **LoRA** | Rank-32 adaptation of Q/V/Out projections in all 12 attention layers → 1.77 M trainable params (1.9 %) |
| **Machine-Aware Adapters (MGA)** | Per-machine bottleneck MLP (768 → 64 → 768) with residual connection |
| **SpecAugment** | Two independent augmented views per sample — time mask 15 % + feature mask 15 % on BEATs frame features |
| **Dual-Level Contrastive Loss** | File-level SupCon (mean-pooled) + frame-level SupCon (K=10 random frames), both views concatenated [2B, 256] |
| **Projection head** | 768 → 512 → 256, L2-normalized |
| **Optimizer** | AdamW — LoRA: 2e-4, MGA: 5e-4, head: 1e-3, weight decay 1e-4 |
| **Scheduler** | CosineAnnealingLR → 1e-6 |
| **Effective batch** | 4 × 4 gradient accumulation = 16 |
| **Early stopping** | Patience 10 on full-validation SupCon loss |
| **Checkpointing** | Saves on every val improvement, full resume support |

Two separate models trained — one for DCASE (dev + additional, 16 machine types), one for MIMII-DUE (5 machine types).

- **Scripts** — `scripts/beats_train.py`, `scripts/beats_train_all.sh`, `scripts/beats_evaluate.py`

---

## Datasets

| Dataset | Machine Types | Train Files | Notes |
|---|---|---|---|
| DCASE 2024 Task 2 (dev) | ToyCar, ToyTrain, bearing, fan, gearbox, slider, valve | ~1 000 normal / machine | Domain shift: 990 source / 10 target |
| DCASE 2024 (additional) | 16 machine types | large | Used for BEATs+LoRA training only |
| MIMII-DUE | fan, gearbox, pump, slider, valve | ~30 000 total | Multi-section, source + target |

Manifests: `data/processed/manifests/`

---

## Installation

```bash
git clone https://github.com/Jean-Mardini/acoustic-anomaly-detection.git
cd acoustic-anomaly-detection
pip install -r requirements.txt
```

**Pretrained checkpoints** (not in git):
- `models/BEATs_iter3_plus_AS2M.pt` — 345 MB · [download from Microsoft UniLM](https://github.com/microsoft/unilm/tree/master/beats)
- `models/wavlm/` — WavLM Base Plus via HuggingFace `microsoft/wavlm-base-plus`

---

## Usage

### Train Conv AE / Transformer AE
```bash
# Single machine type
python3 scripts/train.py \
  --machine-types fan \
  --manifests data/processed/manifests/dcase2024_development.csv \
  --epochs 50 --early-stopping 8

# All DCASE machines
bash scripts/train_all_dcase.sh

# All MIMII machines
bash scripts/train_all_mimii.sh
```

### Evaluate Conv AE / Transformer AE
```bash
python3 scripts/evaluate.py \
  --checkpoint artifacts/runs/fan_best_v1/best_model.pt \
  --manifests data/processed/manifests/dcase2024_development.csv
```

### Train BEATs + LoRA (full pipeline)
```bash
bash scripts/beats_train_all.sh
```
Trains DCASE → evaluates → trains MIMII → evaluates → prints final AUC summary.  
Kill anytime with `Ctrl+C` — resumes automatically from the last checkpoint.

### Evaluate BEATs frozen
```bash
python3 scripts/beats_evaluate.py \
  --beats-ckpt models/BEATs_iter3_plus_AS2M.pt \
  --manifests data/processed/manifests/dcase2024_development.csv \
  --gmm-components 32
```

### Evaluate BEATs + LoRA
```bash
python3 scripts/beats_evaluate.py \
  --beats-ckpt models/BEATs_iter3_plus_AS2M.pt \
  --lora-ckpt artifacts/beats_lora_dcase/beats_lora.pt \
  --manifests data/processed/manifests/dcase2024_development.csv
```

### Export scorers for deployment
```bash
python3 scripts/beats_export_scorers.py \
  --manifests data/processed/manifests/dcase2024_development.csv \
  --beats-ckpt models/BEATs_iter3_plus_AS2M.pt \
  --out-dir beats_frozen_export
```

---

## Inference (Demo App)

```python
import pickle, json, numpy as np, torch, sys
sys.path.insert(0, "models")
from BEATs import BEATs, BEATsConfig

# 1. Load BEATs
raw = torch.load("models/BEATs_iter3_plus_AS2M.pt", map_location="cpu")
beats = BEATs(BEATsConfig(raw["cfg"]))
beats.load_state_dict(raw["model"])
beats.eval()

# 2. Load GMM + threshold for target machine
with open("beats_frozen_export/gmms/fan.pkl", "rb") as f:
    gmm = pickle.load(f)
meta = json.load(open("beats_frozen_export/embeddings_info.json"))
threshold = meta["machines"]["fan"]["threshold_99"]

# 3. Score audio  (wav: numpy float32, 16 kHz mono, 10 seconds)
wav_t = torch.from_numpy(wav).unsqueeze(0)
pad   = torch.zeros(1, wav_t.size(1), dtype=torch.bool)
with torch.no_grad():
    feats, _ = beats.extract_features(wav_t, padding_mask=pad)
emb = feats.mean(dim=1).numpy()

score      = float(-gmm.score_samples(emb)[0])   # higher → more anomalous
is_anomaly = score > threshold
```

> Use `threshold_99` for a production/demo setting (fewer false alarms).  
> Use `threshold_95` if you prefer higher sensitivity.

---

## Project Structure

```
acoustic-anomaly-detection/
│
├── data/processed/manifests/          # CSV file lists (train / test splits, per dataset)
│
├── src/aad/
│   ├── config.py                      # AudioConfig, FeatureConfig, WindowConfig
│   ├── dataset.py                     # manifest loading, FileRecord dataclass
│   ├── preprocess.py                  # audio loading, log-mel, z-score, windowing
│   ├── model.py                       # ConvAutoencoder, TransformerAutoencoder
│   └── evaluate_utils.py              # AUC-ROC, pAUC, partial ROC helpers
│
├── scripts/
│   ├── train.py                       # Conv AE / Transformer AE training
│   ├── evaluate.py                    # Conv AE / Transformer AE evaluation
│   ├── train_all_dcase.sh             # train all DCASE machine types (Conv AE)
│   ├── train_all_dcase_transformer.sh # train all DCASE machine types (Transformer)
│   ├── train_all_mimii.sh             # train all MIMII machine types (Conv AE)
│   ├── train_all_mimii_transformer.sh # train all MIMII machine types (Transformer)
│   ├── wavlm_evaluate.py              # WavLM frozen + GMM / LOF / Mahalanobis
│   ├── beats_train.py                 # BEATs + LoRA + MGA + DLCL training
│   ├── beats_evaluate.py              # BEATs GMM evaluation
│   ├── beats_export_scorers.py        # export fitted GMMs + thresholds for deployment
│   └── beats_train_all.sh             # full sequential pipeline (DCASE + MIMII)
│
├── models/
│   ├── BEATs.py                       # BEATs model definition (Microsoft)
│   ├── backbone.py                    # TransformerEncoder
│   └── modules.py                     # MultiheadAttention and building blocks
│
├── beats_frozen_export/
│   ├── gmms/                          # fitted GMM scorers per machine type
│   ├── embeddings_info.json           # calibrated thresholds + metadata
│   └── README.txt                     # standalone inference instructions
│
├── artifacts/                         # trained model checkpoints + eval results (not in git)
│   ├── runs/                          # Conv AE + Transformer AE (per machine)
│   ├── beats_lora_dcase/              # BEATs+LoRA DCASE checkpoint
│   └── beats_lora_mimii/              # BEATs+LoRA MIMII checkpoint
│
└── requirements.txt
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| Normal-only training | Anomaly labels unavailable at train time — standard DCASE protocol |
| Per-machine GMM | Avoids contamination between machine types with different acoustic profiles |
| Separate DCASE / MIMII models | Different recording conditions; a shared model confuses machine-aware adapters |
| Full-validation SupCon loss | Small batch sizes produce zero positive pairs — computing loss over the full val set prevents false early stopping |
| Soundfile primary, librosa fallback | Avoids multiprocessing issues and librosa deprecation warnings on broken files |
| Checkpoint on every val improvement | Safe to interrupt long GPU training (20 min/epoch) and resume without losing progress |
