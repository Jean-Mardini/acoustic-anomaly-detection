# Acoustic Anomaly Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![BEATs](https://img.shields.io/badge/BEATs-AudioSet-orange)
![DCASE](https://img.shields.io/badge/DCASE-2024_Task_2-green)
![MIMII](https://img.shields.io/badge/MIMII-DUE-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

**An end-to-end unsupervised acoustic anomaly detection pipeline — progressing from a convolutional autoencoder baseline to the DCASE 2024 winning architecture, benchmarked on DCASE 2024 Task 2 and MIMII-DUE.**

[Overview](#overview) · [Results](#results) · [Methods](#methods) · [Installation](#installation) · [Usage](#usage) · [Inference](#inference) · [Structure](#project-structure)

</div>

---

## Authors

| Name | Institution |
|---|---|
| Jean Mardini | Master in Artificial Intelligence — ESIB, USJ |
| Marc Khattar | Master in Artificial Intelligence — ESIB, USJ |
| Christy Tannoury | Master in Artificial Intelligence — ESIB, USJ |
| Angela Nabhan | Master in Artificial Intelligence — ESIB, USJ |
| Charbel Mezraani | Master in Artificial Intelligence — ESIB, USJ |

---

## Overview

This project implements and benchmarks a full progression of acoustic anomaly detection methods under real-world constraints: unsupervised training on normal audio only, domain shift between source and target recordings, and evaluation across multiple machine types.

The pipeline covers five stages — from classical handcrafted features to fine-tuned audio foundation models — culminating in a reproduction of the DCASE 2024 Task 2 winning system.

---

## Results

### DCASE 2024 Task 2 — Development Set (AUC-ROC ↑)

| Model | ToyCar | ToyTrain | Bearing | Fan | Gearbox | Slider | Valve | **Mean** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Convolutional Autoencoder | 0.490 | 0.463 | 0.630 | 0.525 | 0.517 | 0.551 | 0.396 | 0.510 |
| Transformer Autoencoder | 0.443 | 0.555 | 0.623 | 0.532 | 0.556 | 0.493 | 0.392 | 0.513 |
| WavLM + GMM *(frozen)* | 0.428 | 0.533 | 0.541 | 0.539 | 0.535 | 0.560 | 0.461 | 0.514 |
| WavLM + Mahalanobis *(frozen)* | 0.470 | 0.584 | 0.566 | 0.557 | 0.532 | 0.569 | 0.490 | 0.538 |
| WavLM + LOF *(frozen)* | 0.483 | 0.563 | 0.640 | 0.542 | 0.523 | 0.571 | 0.564 | 0.555 |
| **BEATs *(frozen, AudioSet)*** | **0.504** | **0.709** | **0.609** | **0.618** | **0.776** | **0.654** | **0.597** | **0.638** |
| BEATs + LoRA + MGA + DLCL | — | — | — | — | — | — | — | *in progress* |

### MIMII-DUE (AUC-ROC ↑)

| Model | Fan | Gearbox | Pump | Slider | Valve | **Mean** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Transformer Autoencoder | 0.507 | 0.583 | 0.547 | 0.581 | 0.513 | 0.546 |
| BEATs + LoRA + MGA + DLCL | — | — | — | — | — | *in progress* |

---

## Methods

### Stage 1 · Convolutional Autoencoder

Reconstruction-based anomaly detection on log-mel spectrograms.

- **Features** — log-mel (`n_fft=1024`, `hop=512`, `n_mels=128`), z-score normalized on normal train only
- **Model** — symmetric Conv encoder → bottleneck → Conv decoder, MSE reconstruction loss
- **Scoring** — mean window reconstruction error, per-machine GMM / LOF / Mahalanobis calibration
- **Scripts** — `scripts/train.py`, `scripts/evaluate.py`

### Stage 2 · Transformer Autoencoder

Patch-based self-attention encoder replacing the convolutional backbone.

- **Architecture** — spectrogram patches → linear projection → multi-head self-attention → reconstruction
- **Same scoring pipeline** as Conv AE
- **Scripts** — `scripts/train.py`, `scripts/evaluate.py`

### Stage 3 · WavLM Frozen Features

Microsoft WavLM Base Plus (pretrained on 60 000 hours) as a frozen feature extractor.

- **Features** — 768-dim frame embeddings, mean-pooled over time
- **Scorers benchmarked** — GMM, Mahalanobis, LOF
- **Scripts** — `scripts/wavlm_evaluate.py`

### Stage 4 · BEATs Frozen Features

Microsoft BEATs (92 M params, pretrained on AudioSet 2 M clips) — strongest result without fine-tuning.

- **Features** — 768-dim BEATs embeddings, mean-pooled over ~500 frames
- **Scorer** — per-machine GMM (32 components, full covariance)
- **Calibration** — 95th / 99th percentile thresholds on normal training scores
- **Scripts** — `scripts/beats_evaluate.py`, `scripts/beats_export_scorers.py`

### Stage 5 · BEATs + LoRA + MGA + DLCL

Full reproduction of the **DCASE 2024 Task 2 winning system** (without ensembling).

| Component | Detail |
|---|---|
| **LoRA** | Rank-32 adaptation of Q/V/Out projections in all 12 attention layers → 1.77 M trainable params (1.9 %) |
| **Machine-Aware Adapters (MGA)** | Per-machine bottleneck MLP (768 → 64 → 768) with residual connection |
| **SpecAugment** | Two independent augmented views — time mask 15 % + feature mask 15 % on BEATs frame features [B, T, 768] |
| **Dual-Level Contrastive Loss** | File-level SupCon (mean-pooled) + frame-level SupCon (K=10 frames), both views concatenated [2B, 256] |
| **Projection Head** | 768 → 512 → 256, L2-normalized |
| **Optimizer** | AdamW — LoRA: 2e-4 · MGA: 5e-4 · Head: 1e-3 · Weight decay: 1e-4 |
| **Scheduler** | CosineAnnealingLR decaying to 1e-6 |
| **Effective Batch** | 4 × 4 gradient accumulation = 16 |
| **Early Stopping** | Patience 10 on full-validation SupCon loss |
| **Checkpointing** | Full state saved on every val improvement — kill and resume anytime |

- **Scripts** — `scripts/beats_train.py`, `scripts/beats_train_all.sh`, `scripts/beats_evaluate.py`

---

## Datasets

| Dataset | Machine Types | Split | Notes |
|---|---|---|---|
| DCASE 2024 Task 2 (dev) | ToyCar, ToyTrain, bearing, fan, gearbox, slider, valve | train / test | Domain shift: 990 source / 10 target per section |
| DCASE 2024 (additional) | 16 machine types | train | Used for BEATs+LoRA training only |
| MIMII-DUE | fan, gearbox, pump, slider, valve | train / test | Multi-section, source + target domains |

Manifests: `data/processed/manifests/`

---

## Installation

```bash
git clone https://github.com/Jean-Mardini/acoustic-anomaly-detection.git
cd acoustic-anomaly-detection
pip install -r requirements.txt
```

**Pretrained checkpoints** (not in git — download separately):

| File | Size | Source |
|---|---|---|
| `models/BEATs_iter3_plus_AS2M.pt` | 345 MB | [Microsoft UniLM / BEATs](https://github.com/microsoft/unilm/tree/master/beats) |
| `models/wavlm/` | 721 MB | HuggingFace `microsoft/wavlm-base-plus` |

---

## Usage

### Conv AE / Transformer AE

```bash
# Train single machine
python3 scripts/train.py \
  --machine-types fan \
  --manifests data/processed/manifests/dcase2024_development.csv \
  --epochs 50 --early-stopping 8

# Train all DCASE machines
bash scripts/train_all_dcase.sh
bash scripts/train_all_dcase_transformer.sh

# Train all MIMII machines
bash scripts/train_all_mimii.sh
bash scripts/train_all_mimii_transformer.sh

# Evaluate
python3 scripts/evaluate.py \
  --checkpoint artifacts/runs/fan_best_v1/best_model.pt \
  --manifests data/processed/manifests/dcase2024_development.csv
```

### BEATs Frozen

```bash
python3 scripts/beats_evaluate.py \
  --beats-ckpt models/BEATs_iter3_plus_AS2M.pt \
  --manifests data/processed/manifests/dcase2024_development.csv \
  --gmm-components 32
```

### BEATs + LoRA (Full Pipeline)

```bash
# Train DCASE + MIMII sequentially, then evaluate both
bash scripts/beats_train_all.sh
```

> Supports pause and resume — checkpoint saved on every val loss improvement.  
> Kill with `Ctrl+C` at any time and re-run the same command to continue.

### Export Scorers for Deployment

```bash
python3 scripts/beats_export_scorers.py \
  --manifests data/processed/manifests/dcase2024_development.csv \
  --beats-ckpt models/BEATs_iter3_plus_AS2M.pt \
  --out-dir beats_frozen_export
```

---

## Inference

Minimal inference example using the exported BEATs frozen scorers:

```python
import pickle, json, numpy as np, torch, sys
sys.path.insert(0, "models")
from BEATs import BEATs, BEATsConfig

# Load BEATs
raw   = torch.load("models/BEATs_iter3_plus_AS2M.pt", map_location="cpu")
beats = BEATs(BEATsConfig(raw["cfg"]))
beats.load_state_dict(raw["model"])
beats.eval()

# Load GMM + calibrated threshold
with open("beats_frozen_export/gmms/fan.pkl", "rb") as f:
    gmm = pickle.load(f)
meta      = json.load(open("beats_frozen_export/embeddings_info.json"))
threshold = meta["machines"]["fan"]["threshold_99"]

# Score a 10-second audio clip (numpy float32, 16 kHz mono)
wav_t = torch.from_numpy(wav).unsqueeze(0)
pad   = torch.zeros(1, wav_t.size(1), dtype=torch.bool)
with torch.no_grad():
    feats, _ = beats.extract_features(wav_t, padding_mask=pad)
emb = feats.mean(dim=1).numpy()

anomaly_score = float(-gmm.score_samples(emb)[0])  # higher → more anomalous
is_anomaly    = anomaly_score > threshold
```

> `threshold_99` — fewer false alarms, recommended for demos  
> `threshold_95` — higher sensitivity, catches more anomalies

---

## Project Structure

```
acoustic-anomaly-detection/
│
├── data/processed/manifests/          # CSV file lists (train / test splits)
│
├── src/aad/
│   ├── config.py                      # AudioConfig, FeatureConfig, WindowConfig
│   ├── dataset.py                     # manifest loading, FileRecord dataclass
│   ├── preprocess.py                  # audio loading, log-mel, z-score, windowing
│   ├── model.py                       # ConvAutoencoder, TransformerAutoencoder
│   └── evaluate_utils.py              # AUC-ROC, pAUC helpers
│
├── scripts/
│   ├── train.py                       # Conv AE / Transformer AE training
│   ├── evaluate.py                    # Conv AE / Transformer AE evaluation
│   ├── train_all_dcase.sh             # all DCASE machines — Conv AE
│   ├── train_all_dcase_transformer.sh # all DCASE machines — Transformer AE
│   ├── train_all_mimii.sh             # all MIMII machines — Conv AE
│   ├── train_all_mimii_transformer.sh # all MIMII machines — Transformer AE
│   ├── wavlm_evaluate.py              # WavLM frozen + GMM / LOF / Mahalanobis
│   ├── beats_train.py                 # BEATs + LoRA + MGA + DLCL training
│   ├── beats_evaluate.py              # BEATs GMM evaluation
│   ├── beats_export_scorers.py        # export GMMs + thresholds for deployment
│   └── beats_train_all.sh             # full sequential pipeline
│
├── models/
│   ├── BEATs.py                       # BEATs model (Microsoft)
│   ├── backbone.py                    # TransformerEncoder
│   └── modules.py                     # MultiheadAttention and building blocks
│
├── beats_frozen_export/
│   ├── gmms/                          # fitted GMM scorers per machine type
│   ├── embeddings_info.json           # thresholds + metadata
│   └── README.txt                     # standalone inference instructions
│
└── artifacts/                         # model checkpoints + results (not in git)
    ├── runs/                          # Conv AE + Transformer AE per machine
    ├── beats_lora_dcase/              # BEATs+LoRA DCASE checkpoint
    └── beats_lora_mimii/              # BEATs+LoRA MIMII checkpoint
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| Normal-only training | Anomaly labels unavailable at train time — standard DCASE protocol |
| Per-machine GMM | Avoids cross-machine contamination between different acoustic profiles |
| Separate DCASE / MIMII models | Different recording conditions; mixing datasets confuses machine-aware adapters |
| Full-validation SupCon loss | Small batches → zero positive pairs → false early stopping. Full-set validation fixes this |
| Soundfile primary, librosa fallback | Avoids multiprocessing issues and librosa deprecation warnings |
| Checkpoint on every val improvement | Safe to kill 20-min/epoch GPU training and resume without losing progress |

---

## License

MIT
