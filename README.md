# Acoustic Anomaly Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![BEATs](https://img.shields.io/badge/BEATs-AudioSet-orange)
![DCASE](https://img.shields.io/badge/DCASE-2024_Task_2-green)
![MIMII](https://img.shields.io/badge/MIMII-DUE-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

**An end-to-end unsupervised acoustic anomaly detection pipeline - progressing from convolutional and transformer autoencoder baselines to BEATs frozen features, benchmarked on DCASE 2024 Task 2 and MIMII-DUE.**

[Overview](#overview) · [Results](#results) · [Methods](#methods) · [Installation](#installation) · [Usage](#usage) · [Inference](#inference) · [Structure](#project-structure)

</div>

---

## Authors

| Name | Institution |
|---|---|
| Jean Mardini |   Artificial Intelligence - ESIB, USJ |
| Marc Khattar |   Artificial Intelligence - ESIB, USJ |
| Christy Tannoury |  Artificial Intelligence - ESIB, USJ |
| Angela Nabhan |  Artificial Intelligence - ESIB, USJ |


---

## Overview

This project benchmarks acoustic anomaly detection under realistic industrial constraints:

- training on normal audio only
- source/target domain shift
- machine-specific scoring and thresholds

The pipeline covers Conv AE, Transformer AE, and BEATs frozen features, then exports deployment-ready scorer artifacts used directly by the API/UI.

---

## Results

Metric: **AUC-ROC (higher is better)**  
Source files: `artifacts/dcase/**/evaluation*.json`, `artifacts/mimii/**/evaluation*.json`, `beats_frozen_export/beats_frozen_results.json`

### DCASE 2024 Task 2 - Development Set

| Model | ToyCar | ToyTrain | Bearing | Fan | Gearbox | Slider | Valve | **Mean** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Convolutional Autoencoder (best scorer per machine) | 0.531 | 0.488 | 0.630 | 0.599 | 0.575 | 0.551 | 0.496 | 0.553 |
| Transformer Autoencoder (best scorer per machine) | 0.511 | 0.555 | 0.623 | 0.621 | 0.561 | 0.507 | 0.533 | 0.559 |
| **BEATs (frozen, AudioSet) + GMM** | **0.504** | **0.709** | **0.609** | **0.618** | **0.776** | **0.654** | **0.597** | **0.638** |

### MIMII-DUE

| Model | Fan | Gearbox | Pump | Slider | Valve | **Mean** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Convolutional Autoencoder (best scorer per machine)** | **0.624** | **0.652** | **0.628** | **0.640** | **0.506** | **0.610** |
| Transformer Autoencoder (best scorer per machine) | 0.514 | 0.584 | 0.547 | 0.586 | 0.515 | 0.549 |

---

## Methods

### Stage 1 · Convolutional Autoencoder

Reconstruction-based anomaly detection on log-mel spectrograms.

- **Features** - log-mel (`n_fft=1024`, `hop=512`, `n_mels=128`), z-score normalized
- **Model** - symmetric Conv encoder -> bottleneck -> Conv decoder, MSE loss
- **Scoring** - per-machine reconstruction threshold + scorer calibration
- **Scripts** - `scripts/train.py`, `scripts/evaluate.py`

### Stage 2 · Transformer Autoencoder

Patch-based self-attention encoder replacing the convolutional backbone.

- **Architecture** - spectrogram patches -> linear projection -> MHSA -> reconstruction
- **Scoring** - same evaluation stack as Conv AE (method chosen per machine)
- **Scripts** - `scripts/train.py`, `scripts/evaluate.py`

### Stage 3 · BEATs Frozen Features

Microsoft BEATs pretrained on AudioSet used as a frozen embedding backbone.

- **Features** - 768-d embeddings, mean pooled over time
- **Scorer** - per-machine GMM
- **Calibration** - percentile thresholds exported for deployment
- **Scripts** - `scripts/beats_evaluate.py`, `scripts/beats_export_scorers.py`

---

## Datasets

| Dataset | Machine Types | Split | Notes |
|---|---|---|---|
| DCASE 2024 Task 2 (dev) | ToyCar, ToyTrain, bearing, fan, gearbox, slider, valve | train / test | Domain shift: source + target |
| DCASE 2024 (additional) | multiple machine types | train | Extra training data support |
| MIMII-DUE | fan, gearbox, pump, slider, valve | train / test | Multi-section, source + target domains |

Manifests: `data/processed/manifests/`

---

## Installation

```bash
git clone https://github.com/Jean-Mardini/acoustic-anomaly-detection.git
cd acoustic-anomaly-detection
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Conv AE / Transformer AE

```bash
# Train single machine
python scripts/train.py \
  --machine-types fan \
  --manifests data/processed/manifests/dcase2024_development.csv \
  --epochs 50 --early-stopping 8

# Train all DCASE machines
bash scripts/train_all_dcase.sh
bash scripts/train_all_dcase_transformer.sh

# Train all MIMII machines
bash scripts/train_all_mimii.sh
bash scripts/train_all_mimii_transformer.sh

# Evaluate a checkpoint
python scripts/evaluate.py \
  --checkpoint artifacts/dcase/conv_ae/fan_best_v1/best_model.pt \
  --manifests data/processed/manifests/dcase2024_development.csv
```

### BEATs Frozen

```bash
python scripts/beats_evaluate.py \
  --beats-ckpt models/BEATs_iter3_plus_AS2M.pt \
  --manifests data/processed/manifests/dcase2024_development.csv \
  --gmm-components 32
```

### Export Scorers for Deployment

```bash
python scripts/export_best_scorer_artifacts.py

python scripts/beats_export_scorers.py \
  --manifests data/processed/manifests/dcase2024_development.csv \
  --beats-ckpt models/BEATs_iter3_plus_AS2M.pt \
  --out-dir beats_frozen_export
```

---

## Inference

Run the API:

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- UI: [http://localhost:8000](http://localhost:8000)
- Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

Minimal BEATs frozen scorer usage:

```python
import pickle, json, numpy as np, torch, sys
sys.path.insert(0, "models")
from BEATs import BEATs, BEATsConfig

raw = torch.load("models/BEATs_iter3_plus_AS2M.pt", map_location="cpu")
beats = BEATs(BEATsConfig(raw["cfg"]))
beats.load_state_dict(raw["model"])
beats.eval()

with open("beats_frozen_export/gmms/fan.pkl", "rb") as f:
    gmm = pickle.load(f)
meta = json.load(open("beats_frozen_export/embeddings_info.json"))
threshold = meta["machines"]["fan"]["threshold_99"]

# wav: numpy float32 mono waveform at 16kHz
wav_t = torch.from_numpy(wav).unsqueeze(0)
pad = torch.zeros(1, wav_t.size(1), dtype=torch.bool)
with torch.no_grad():
    feats, _ = beats.extract_features(wav_t, padding_mask=pad)
emb = feats.mean(dim=1).numpy()

anomaly_score = float(-gmm.score_samples(emb)[0])
is_anomaly = anomaly_score > threshold
```

---

## Project Structure

```text
acoustic-anomaly-detection/
│
├── app/                               # FastAPI app + web UI
├── data/processed/manifests/          # manifest CSVs (train / test splits)
├── src/aad/
│   ├── config.py                      # AudioConfig, FeatureConfig, WindowConfig
│   ├── dataset.py                     # manifest loading, FileRecord
│   ├── preprocess.py                  # audio loading, log-mel, z-score, windowing
│   ├── model.py                       # ConvAutoencoder, TransformerAutoencoder
│   └── evaluate_utils.py              # scoring/evaluation helpers
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── train_all_dcase.sh
│   ├── train_all_dcase_transformer.sh
│   ├── train_all_mimii.sh
│   ├── train_all_mimii_transformer.sh
│   ├── beats_evaluate.py
│   ├── beats_export_scorers.py
│   └── export_best_scorer_artifacts.py
│
├── models/
│   ├── BEATs.py
│   ├── backbone.py
│   └── modules.py
│
├── beats_frozen_export/
│   ├── gmms/
│   ├── embeddings_info.json
│   └── beats_frozen_results.json
│
└── artifacts/
    ├── dcase/
    ├── mimii/
    └── best_scorer_export_summary.json
```

---

## License

MIT
