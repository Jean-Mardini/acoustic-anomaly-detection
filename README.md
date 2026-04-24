# Acoustic Anomaly Detection

End-to-end acoustic anomaly detection pipeline benchmarked on **DCASE 2024 Task 2** and **MIMII-DUE**, demonstrating a clear progression from a simple autoencoder baseline to a state-of-the-art fine-tuned foundation model.

---

## Method Progression

| Model | DCASE Mean AUC |
|---|---|
| Convolutional Autoencoder | 0.5102 |
| Transformer Autoencoder | 0.5132 |
| BEATs (frozen, pretrained on AudioSet) | **0.6382** |
| BEATs + LoRA + MGA + DLCL *(winners' method)* | in progress |

---

## Models

### 1. Convolutional Autoencoder
Log-mel spectrogram → encoder → bottleneck → decoder → reconstruction error as anomaly score.
- `scripts/train.py` — training
- `scripts/evaluate.py` — evaluation (GMM / LOF / Mahalanobis scoring)

### 2. Transformer Autoencoder
Patch-based spectrogram encoder with self-attention, replacing the convolutional backbone.
- Same training/evaluation scripts as Conv AE

### 3. BEATs Frozen
Microsoft BEATs (92M params, pretrained on AudioSet) used as a frozen feature extractor. Per-machine GMM fitted on 768-dim embeddings.
- `scripts/beats_evaluate.py` — evaluation
- `beats_frozen_export/` — fitted GMM scorers + calibrated thresholds (ready for deployment)

### 4. BEATs + LoRA + MGA + DLCL
Full winners' method from DCASE 2024:
- **LoRA** (rank=32) fine-tuning of BEATs attention layers
- **Machine-Aware Adapters** — per-machine bottleneck MLP
- **Dual-Level Contrastive Loss** — file-level + frame-level SupCon
- **SpecAugment** — two-view time + feature masking
- `scripts/beats_train.py` — training (with checkpoint/resume support)
- `scripts/beats_train_all.sh` — trains DCASE + MIMII sequentially

---

## Datasets

- **DCASE 2024 Task 2** — 7 machine types (ToyCar, ToyTrain, bearing, fan, gearbox, slider, valve)
- **MIMII-DUE** — 5 machine types (fan, gearbox, pump, slider, valve)

Manifests: `data/processed/manifests/`

---

## Installation

```bash
pip install torch torchaudio librosa soundfile scikit-learn numpy tqdm
```

Download pretrained BEATs checkpoint:
```
models/BEATs_iter3_plus_AS2M.pt  ← place here (345MB, not in git)
```

---

## Training

**Conv AE / Transformer AE:**
```bash
python3 scripts/train.py --machine-types fan --epochs 50
```

**BEATs + LoRA (DCASE + MIMII sequentially):**
```bash
bash scripts/beats_train_all.sh
```

Supports pause/resume — checkpoint saved every time val loss improves.

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
  --manifests data/processed/manifests/dcase2024_development.csv
```

**BEATs + LoRA:**
```bash
python3 scripts/beats_evaluate.py \
  --beats-ckpt models/BEATs_iter3_plus_AS2M.pt \
  --lora-ckpt artifacts/beats_lora_dcase/beats_lora.pt \
  --manifests data/processed/manifests/dcase2024_development.csv
```

---

## Deployment (Demo App)

The `beats_frozen_export/` folder contains everything needed for inference:

```python
import pickle, json, numpy as np, torch, sys
sys.path.insert(0, "models")
from BEATs import BEATs, BEATsConfig

# Load BEATs
raw = torch.load("models/BEATs_iter3_plus_AS2M.pt", map_location="cpu")
beats = BEATs(BEATsConfig(raw["cfg"]))
beats.load_state_dict(raw["model"])
beats.eval()

# Load GMM + threshold for a machine type
with open("beats_frozen_export/gmms/fan.pkl", "rb") as f:
    gmm = pickle.load(f)
info = json.load(open("beats_frozen_export/embeddings_info.json"))
threshold = info["machines"]["fan"]["threshold_99"]

# Score audio (numpy float32, 16kHz, 10 seconds)
wav_t = torch.from_numpy(wav).unsqueeze(0)
pad = torch.zeros(1, wav_t.size(1), dtype=torch.bool)
with torch.no_grad():
    feats, _ = beats.extract_features(wav_t, padding_mask=pad)
emb = feats.mean(dim=1).numpy()
score = float(-gmm.score_samples(emb)[0])
is_anomaly = score > threshold
```

---

## Project Structure

```
acoustic-anomaly-detection/
├── data/processed/manifests/     # dataset CSVs
├── src/aad/                      # core library (config, dataset, preprocess, evaluate_utils)
├── scripts/
│   ├── train.py                  # Conv AE / Transformer AE training
│   ├── evaluate.py               # Conv AE / Transformer AE evaluation
│   ├── beats_train.py            # BEATs + LoRA + MGA + DLCL training
│   ├── beats_evaluate.py         # BEATs evaluation + GMM scoring
│   ├── beats_export_scorers.py   # export fitted GMMs for deployment
│   └── beats_train_all.sh        # full pipeline (DCASE + MIMII)
├── models/                       # BEATs model code + pretrained checkpoint
├── beats_frozen_export/          # fitted GMM scorers + thresholds (deployment-ready)
└── artifacts/                    # trained models + evaluation results (not in git)
```
