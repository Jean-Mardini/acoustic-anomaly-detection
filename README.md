# Acoustic Anomaly Detection

A research-grade pipeline for unsupervised acoustic anomaly detection, benchmarked on **DCASE 2024 Task 2** and **MIMII-DUE**. The project demonstrates a full progression from a classical autoencoder baseline to a state-of-the-art fine-tuned foundation model, reproducing the core methodology of the DCASE 2024 winning system.

---

## Results Summary

### DCASE 2024 Task 2 (Development Set)

| Model | ToyCar | ToyTrain | Bearing | Fan | Gearbox | Slider | Valve | **Mean AUC** |
|---|---|---|---|---|---|---|---|---|
| Convolutional Autoencoder | 0.4895 | 0.4625 | 0.6295 | 0.5247 | 0.5173 | 0.5514 | 0.3963 | 0.5102 |
| Transformer Autoencoder | 0.4430 | 0.5553 | 0.6225 | 0.5320 | 0.5558 | 0.4925 | 0.3916 | 0.5132 |
| WavLM + GMM (frozen) | 0.4279 | 0.5327 | 0.5409 | 0.5385 | 0.5353 | 0.5597 | 0.4605 | 0.5136 |
| WavLM + LOF (frozen) | 0.4830 | 0.5631 | 0.6395 | 0.5423 | 0.5229 | 0.5710 | 0.5641 | 0.5551 |
| WavLM + Mahalanobis (frozen) | 0.4698 | 0.5836 | 0.5661 | 0.5566 | 0.5321 | 0.5692 | 0.4896 | 0.5381 |
| **BEATs (frozen, AudioSet)** | 0.5044 | 0.7087 | 0.6088 | 0.6180 | 0.7763 | 0.6540 | 0.5972 | **0.6382** |
| BEATs + LoRA + MGA + DLCL | — | — | — | — | — | — | — | *in progress* |

### MIMII-DUE

| Model | Fan | Gearbox | Pump | Slider | Valve | **Mean AUC** |
|---|---|---|---|---|---|---|
| Transformer Autoencoder | 0.5068 | 0.5834 | 0.5469 | 0.5806 | 0.5128 | 0.5461 |
| BEATs + LoRA + MGA + DLCL | — | — | — | — | — | *in progress* |

---

## Method Progression

### Stage 1 — Convolutional Autoencoder
Classical reconstruction-based anomaly detection on log-mel spectrograms.

- **Features:** log-mel spectrogram (`n_fft=1024`, `hop=512`, `n_mels=128`), z-score normalized on normal train only
- **Model:** Conv encoder → bottleneck → Conv decoder, MSE reconstruction loss
- **Scoring:** mean reconstruction error over spectrogram windows, per-machine GMM / LOF / Mahalanobis
- **Training:** per machine type, early stopping on validation MSE
- **Scripts:** `scripts/train.py`, `scripts/evaluate.py`

### Stage 2 — Transformer Autoencoder
Replaced the convolutional backbone with a patch-based self-attention encoder for richer temporal modelling.

- **Architecture:** spectrogram patches → linear embedding → multi-head self-attention → reconstruction
- **Same scoring pipeline** as Conv AE
- **Trained on:** DCASE 2024 (7 machine types) and MIMII-DUE (5 machine types) separately
- **Scripts:** `scripts/train.py` (with `--model transformer`), `scripts/evaluate.py`

### Stage 3 — WavLM Frozen Feature Extractor
Microsoft WavLM Base Plus (pretrained on 60k hours) as a frozen backbone, replacing hand-crafted features.

- **Features:** 768-dim WavLM embeddings, mean-pooled over time
- **Scorers compared:** GMM, LOF, Mahalanobis
- **Best result:** LOF → mean AUC 0.5551
- **Scripts:** `scripts/evaluate.py` with WavLM extractor

### Stage 4 — BEATs Frozen Feature Extractor
Microsoft BEATs (92M params, pretrained on AudioSet 2M clips) as frozen backbone — strongest audio foundation model tested.

- **Features:** 768-dim BEATs embeddings (`extract_features`), mean-pooled over ~500 frames
- **Scorer:** per-machine GMM (32 components, full covariance)
- **Result:** mean AUC **0.6382** — best result achieved without fine-tuning
- **Deployment:** fitted GMMs exported with calibrated thresholds (95th/99th percentile on normal train)
- **Scripts:** `scripts/beats_evaluate.py`, `scripts/beats_export_scorers.py`

### Stage 5 — BEATs + LoRA + MGA + DLCL (Winners' Method)
Full reproduction of the DCASE 2024 Task 2 winning methodology (without ensembling).

**Components:**
- **LoRA** (rank=32, alpha=32): low-rank adaptation of BEATs Q/V/Out attention projections → 1.77M trainable params (1.9% of total)
- **Machine-Aware Adapters (MGA)**: per-machine-type bottleneck MLP (768→64→768) with residual connection
- **SpecAugment**: two independent augmented views per sample (time mask 15% + feature mask 15%) applied to BEATs frame features [B, T, 768]
- **Dual-Level Contrastive Loss (DLCL)**: file-level SupCon (mean-pooled embeddings) + frame-level SupCon (K=10 random frames), both views concatenated [2B, 256]
- **Projection head**: 768 → 512 → 256, L2-normalized
- **Optimizer**: AdamW with separate LRs (LoRA: 2e-4, MGA: 5e-4, head: 1e-3), weight decay 1e-4
- **Scheduler**: CosineAnnealingLR decaying to 1e-6
- **Gradient accumulation**: 4 steps × batch 4 = effective batch 16
- **Early stopping**: patience=10 on full-validation-set SupCon loss
- **Checkpointing**: saves on every val loss improvement, full resume support
- **Scoring**: per-machine GMM (32 components)
- **Trained separately**: DCASE model (dev + additional, 16 machine types) and MIMII-DUE model (5 machine types)
- **Scripts:** `scripts/beats_train.py`, `scripts/beats_train_all.sh`, `scripts/beats_evaluate.py`

---

## Datasets

### DCASE 2024 Task 2
7 machine types: **ToyCar, ToyTrain, bearing, fan, gearbox, slider, valve**
- Development set: ~1000 normal train + 200 test (normal + anomaly) per machine
- Additional training data included for BEATs+LoRA training
- Domain shift: 990 source / 10 target training files per section

### MIMII-DUE
5 machine types: **fan, gearbox, pump, slider, valve**
- Multi-section, source + target domain recordings
- ~30,000 normal training files total

Manifests: `data/processed/manifests/`

---

## Installation

```bash
pip install torch torchaudio librosa soundfile scikit-learn numpy tqdm
```

**Required pretrained checkpoints** (not in git — place in `models/`):
- `models/BEATs_iter3_plus_AS2M.pt` — 345MB, [download from Microsoft](https://github.com/microsoft/unilm/tree/master/beats)
- `models/wavlm/` — WavLM Base Plus via HuggingFace

---

## Training

**Conv AE / Transformer AE (per machine type):**
```bash
python3 scripts/train.py \
  --machine-types fan \
  --manifests data/processed/manifests/dcase2024_development.csv \
  --epochs 50 --early-stopping 8
```

**BEATs + LoRA — full pipeline (DCASE then MIMII):**
```bash
bash scripts/beats_train_all.sh
```
Supports pause and resume — checkpoint saved every time val loss improves. Kill anytime and re-run the same command.

---

## Evaluation

**Conv AE / Transformer AE:**
```bash
python3 scripts/evaluate.py \
  --checkpoint artifacts/runs/fan_best_v1/best_model.pt \
  --manifests data/processed/manifests/dcase2024_development.csv
```

**BEATs frozen:**
```bash
python3 scripts/beats_evaluate.py \
  --beats-ckpt models/BEATs_iter3_plus_AS2M.pt \
  --manifests data/processed/manifests/dcase2024_development.csv \
  --gmm-components 32
```

**BEATs + LoRA:**
```bash
python3 scripts/beats_evaluate.py \
  --beats-ckpt models/BEATs_iter3_plus_AS2M.pt \
  --lora-ckpt artifacts/beats_lora_dcase/beats_lora.pt \
  --manifests data/processed/manifests/dcase2024_development.csv \
  --gmm-components 32
```

**Export GMM scorers for deployment:**
```bash
python3 scripts/beats_export_scorers.py \
  --manifests data/processed/manifests/dcase2024_development.csv \
  --beats-ckpt models/BEATs_iter3_plus_AS2M.pt \
  --out-dir beats_frozen_export
```

---

## Deployment (Demo App)

The `beats_frozen_export/` folder contains everything needed for inference on a new machine:

```python
import pickle, json, numpy as np, torch, sys
sys.path.insert(0, "models")
from BEATs import BEATs, BEATsConfig

# Load BEATs
raw = torch.load("models/BEATs_iter3_plus_AS2M.pt", map_location="cpu")
beats = BEATs(BEATsConfig(raw["cfg"]))
beats.load_state_dict(raw["model"])
beats.eval()

# Load GMM + calibrated threshold for a machine type
with open("beats_frozen_export/gmms/fan.pkl", "rb") as f:
    gmm = pickle.load(f)
info = json.load(open("beats_frozen_export/embeddings_info.json"))
threshold = info["machines"]["fan"]["threshold_99"]

# Score a 10-second audio clip (numpy float32, 16 kHz mono)
wav_t = torch.from_numpy(wav).unsqueeze(0)
pad = torch.zeros(1, wav_t.size(1), dtype=torch.bool)
with torch.no_grad():
    feats, _ = beats.extract_features(wav_t, padding_mask=pad)
emb = feats.mean(dim=1).numpy()

anomaly_score = float(-gmm.score_samples(emb)[0])  # higher = more anomalous
is_anomaly = anomaly_score > threshold
```

**Threshold guide:**
- `threshold_95` — higher sensitivity, catches more anomalies (more false alarms)
- `threshold_99` — higher precision, fewer false alarms (recommended for demo)

---

## Project Structure

```
acoustic-anomaly-detection/
├── data/
│   └── processed/
│       ├── manifests/                    # CSV file lists (train/test splits)
│       └── features/                     # normalization stats
├── src/aad/
│   ├── config.py                         # AudioConfig, FeatureConfig, WindowConfig
│   ├── dataset.py                        # manifest loading, FileRecord
│   ├── preprocess.py                     # audio loading, log-mel, z-score, windowing
│   ├── model.py                          # ConvAutoencoder, TransformerAutoencoder
│   └── evaluate_utils.py                 # AUC, pAUC, partial ROC
├── scripts/
│   ├── train.py                          # Conv AE / Transformer AE training
│   ├── evaluate.py                       # Conv AE / Transformer AE evaluation
│   ├── beats_train.py                    # BEATs + LoRA + MGA + DLCL training
│   ├── beats_evaluate.py                 # BEATs GMM evaluation
│   ├── beats_export_scorers.py           # export fitted GMMs + thresholds
│   └── beats_train_all.sh                # full sequential pipeline
├── models/
│   ├── BEATs.py                          # BEATs model definition
│   ├── backbone.py                       # TransformerEncoder
│   └── modules.py                        # MultiheadAttention, etc.
├── beats_frozen_export/
│   ├── gmms/                             # fitted GMM scorers per machine (on SSD)
│   ├── embeddings_info.json              # thresholds + metadata
│   └── README.txt                        # inference instructions
└── artifacts/                            # trained models + results (not in git)
    ├── runs/                             # Conv AE + Transformer AE checkpoints
    ├── beats_lora_dcase/                 # BEATs+LoRA DCASE model
    └── beats_lora_mimii/                 # BEATs+LoRA MIMII model
```

---

## Key Design Decisions

- **No data leakage**: normalization stats and GMM fitting use only normal training files; test labels never seen during calibration
- **Per-machine models**: separate GMM per machine type — no cross-machine contamination
- **Separate DCASE / MIMII models**: different recording conditions; mixing datasets degrades MGA adapters
- **Validation fix**: SupCon validation computed over the full validation set as one batch — prevents false early stopping with small batches and many pseudo-classes
- **Robust audio loading**: soundfile primary, librosa fallback with retry loop for broken files
