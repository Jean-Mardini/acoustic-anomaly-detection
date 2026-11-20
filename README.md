# Acoustic Anomaly Detection under Domain Shift

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![BEATs](https://img.shields.io/badge/BEATs-AudioSet-orange)
![DCASE](https://img.shields.io/badge/DCASE-2024_Task_2-green)
![MIMII](https://img.shields.io/badge/MIMII-DUE-purple)
![Docker](https://img.shields.io/badge/Docker-Hub-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow)

Detecting faulty industrial machines from their sound, training on normal audio only and testing under changed recording conditions.

[Problem](#problem) · [Results](#results) · [Methods](#methods) · [Docker](#run-with-docker) · [Reproduce](#reproduce) · [Structure](#project-structure)

</div>

---

## Problem

A pump, fan or gearbox often sounds different before it fails. The system takes a short audio clip and returns an anomaly score and a decision (normal / check this machine). The setting has three constraints:

1. Unsupervised: only normal recordings are available for training; anomalies appear only at test time.
2. Domain shift: test clips may come from a different microphone, speed or background noise than training clips (source vs target domain).
3. Per-machine scoring: each machine type gets its own scorer and calibrated threshold.

We compare three approaches, two reconstruction autoencoders and frozen self-supervised audio embeddings, on DCASE 2024 Task 2 and MIMII-DUE. The models are served through a FastAPI web app and a public Docker image.

## Results

AUC-ROC (0.5 = random, 1.0 = perfect), best per machine in bold.

### DCASE 2024 Task 2, development set

| Model | ToyCar | ToyTrain | Bearing | Fan | Gearbox | Slider | Valve | **Mean** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Convolutional AE ¹ | **0.531** | 0.488 | **0.630** | 0.599 | 0.575 | 0.551 | 0.496 | 0.553 |
| Transformer AE ¹ | 0.511 | 0.555 | 0.623 | **0.621** | 0.561 | 0.507 | 0.533 | 0.559 |
| BEATs (frozen) + GMM | 0.504 | **0.709** | 0.609 | 0.618 | **0.776** | **0.654** | **0.597** | **0.638** |

BEATs + GMM partial AUC (FPR ≤ 0.1): ToyCar 0.037 · ToyTrain 0.167 · Bearing 0.289 · Fan 0.253 · Gearbox 0.297 · Slider 0.054 · Valve 0.098.

### MIMII-DUE

| Model | Fan | Gearbox | Pump | Slider | Valve | **Mean** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Convolutional AE ¹ | **0.624** | **0.652** | **0.628** | **0.640** | 0.506 | **0.610** |
| Transformer AE ¹ | 0.514 | 0.584 | 0.547 | 0.586 | **0.515** | 0.549 |

¹ For the autoencoders, each machine uses the best of several scoring methods (reconstruction error, Mahalanobis, GMM, LOF on latents), selected on the evaluation set. These numbers are therefore an optimistic upper bound. BEATs + GMM uses one fixed scorer for every machine, so its results carry no selection bias. BEATs was not run on MIMII-DUE.

### Observations

- Frozen BEATs features with a simple GMM get a mean AUC of 0.638 on DCASE, against about 0.55 for both autoencoders, even with the autoencoders' scorer chosen on the evaluation set. The biggest gain is on Gearbox (+20 points over the Conv AE).
- No model is best on every machine. ToyCar stays close to chance with all three methods.
- The Transformer autoencoder did not improve on the convolutional one: same mean on DCASE, lower on MIMII-DUE.
- pAUC stays below 0.3 for every machine, so the false-alarm rate is still too high for real use without per-machine threshold tuning.

---

## Methods

### Features

Audio at 16 kHz → log-mel spectrogram (`n_fft=1024`, `hop=512`, `n_mels=128`), z-score normalised per machine.

### Stage 1 · Convolutional autoencoder

Symmetric conv encoder → bottleneck → conv decoder, trained with MSE on normal clips. Anomaly score = reconstruction error, or a density score (Mahalanobis, GMM, LOF) fitted on the latent codes.

### Stage 2 · Transformer autoencoder

Spectrogram patches → linear projection → multi-head self-attention → reconstruction, with the same scoring stack.

### Stage 3 · BEATs frozen features + GMM

[Microsoft BEATs](https://github.com/microsoft/unilm/tree/master/beats) pretrained on AudioSet, used as a frozen encoder: 768-d embeddings mean-pooled over time, one Gaussian Mixture Model per machine fitted on normal clips. Score = negative log-likelihood; the decision threshold is the 99th percentile of normal scores, exported with the model for deployment.

### Splits

Train/test manifests keep machine type, section and domain for every clip, so domains and sections are never mixed across splits (see the [dataset card](data/README.md)).

## Datasets

| Dataset | Machine types | Notes |
|---|---|---|
| [DCASE 2024 Task 2](https://dcase.community/challenge2024/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring) (dev) | ToyCar, ToyTrain, bearing, fan, gearbox, slider, valve | Source + target domains |
| [DCASE 2024 additional training data](https://zenodo.org/records/11259435) | multiple | Extra normal clips |
| [MIMII-DUE](https://zenodo.org/records/4740355) | fan, gearbox, pump, slider, valve | Multi-section, source + target domains |

Manifests are in `data/processed/manifests/`; raw audio is not stored in the repo.

---

## Run with Docker

Scoring needs the pretrained BEATs checkpoint `BEATs_iter3_plus_AS2M.pt`. It is too large for the repo and is not inside the Docker image, so download it first from the [official BEATs repository](https://github.com/microsoft/unilm/tree/master/beats) and put it in a folder, for example `~/beats`. Then:

```bash
docker run --rm -p 8000:8000 \
  -v ~/beats:/ckpt \
  -e BEATS_CHECKPOINT=/ckpt/BEATs_iter3_plus_AS2M.pt \
  jeanmardini/acoustic-anomaly-detection:latest
```

Open http://localhost:8000 for the web UI (upload a .wav clip, pick a machine, get a score and a decision) or http://localhost:8000/docs for the API. Without the checkpoint the app still starts, but `/score` returns an error. The app serves the BEATs + GMM scorers; the autoencoder checkpoints are not included in the repo.

| Endpoint | Method | Description |
|---|---|---|
| `/score` | POST | Audio clip (.wav/.flac) + machine type + model → anomaly score, threshold, decision |
| `/models/catalog` | GET | Available machines and scorers |
| `/health` | GET | Health check |

Image on Docker Hub: [jeanmardini/acoustic-anomaly-detection](https://hub.docker.com/r/jeanmardini/acoustic-anomaly-detection). A GitHub Actions workflow (`.github/workflows/docker-publish.yml`) can rebuild and publish it on demand once the `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` repository secrets are set.

## Reproduce

```bash
git clone https://github.com/Jean-Mardini/acoustic-anomaly-detection.git
cd acoustic-anomaly-detection
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the datasets into `data/raw/` (layout in [data/README.md](data/README.md)), then build manifests and features:

```bash
python scripts/build_manifests.py
python scripts/preprocess.py
```

Autoencoders:

```bash
bash scripts/train_all_dcase.sh                 # Conv AE, all DCASE machines
bash scripts/train_all_dcase_transformer.sh     # Transformer AE
bash scripts/train_all_mimii.sh                 # same on MIMII-DUE
bash scripts/train_all_mimii_transformer.sh

python scripts/evaluate.py \
  --checkpoint artifacts/dcase/conv_ae/fan_best_v1/best_model.pt \
  --manifests data/processed/manifests/dcase2024_development.csv
```

BEATs + GMM: download `BEATs_iter3_plus_AS2M.pt` from the [official BEATs repository](https://github.com/microsoft/unilm/tree/master/beats) into `models/`, then:

```bash
python scripts/beats_evaluate.py \
  --beats-ckpt models/BEATs_iter3_plus_AS2M.pt \
  --manifests data/processed/manifests/dcase2024_development.csv \
  --gmm-components 32

python scripts/beats_export_scorers.py \
  --manifests data/processed/manifests/dcase2024_development.csv \
  --beats-ckpt models/BEATs_iter3_plus_AS2M.pt \
  --out-dir beats_frozen_export
```

Training outputs (checkpoints, per-run `evaluation*.json`) go to `artifacts/`, which is git-ignored. The exported BEATs scorers used by the app are committed in `beats_frozen_export/`.

Run the API locally (with `BEATs_iter3_plus_AS2M.pt` in `models/`):

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Project structure

```text
acoustic-anomaly-detection/
├── app/                      # FastAPI app + web UI
├── src/aad/
│   ├── config.py             # audio, feature and window configs
│   ├── dataset.py            # manifest loading
│   ├── preprocess.py         # audio loading, log-mel, normalisation, windowing
│   ├── model.py              # ConvAutoencoder, TransformerAutoencoder
│   └── evaluate_utils.py     # scorers (reconstruction, Mahalanobis, GMM, LOF), metrics
├── scripts/                  # manifests, preprocessing, training, evaluation, export
├── models/                   # BEATs model code
├── beats_frozen_export/      # deployed GMM scorers, thresholds, results
├── data/
│   ├── README.md             # dataset card
│   └── processed/manifests/  # train/test manifests
├── docs/
│   ├── report.docx           # project report
│   └── slides.pptx           # presentation
├── Dockerfile
└── .github/workflows/        # Docker Hub publishing
```

## Team

Jean Mardini (model training) · Marc Khattar · Kristy Tannoury · Angela Nabhan. Faculty of Engineering (ESIB), Université Saint-Joseph de Beyrouth, 2026.

## License

MIT
