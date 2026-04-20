# Acoustic Anomaly Detection

Clean, defendable DCASE-style pipeline with proper folder organization.

## Project Layout

```text
acoustic-anomaly-detection/
├── data/
│   └── processed/manifests/*.csv
├── src/
│   └── aad/
│       ├── __init__.py
│       ├── config.py
│       ├── preprocess.py
│       ├── dataset.py
│       ├── model.py
│       └── evaluate_utils.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── app/
│   └── main.py
└── requirements.txt
```

## Pipeline

- Load audio as mono, resample to 16 kHz, fixed duration.
- Compute log-mel features (`n_fft=1024`, `hop=512`, `n_mels=128`).
- Compute global normalization stats on **normal train** only.
- Apply z-score normalization to train/val/test with the same stats.
- Slice spectrogram into windows (`64`, stride `32`).
- Train convolutional autoencoder on normal windows (MSE).
- Score files using mean reconstruction error over windows.
- Calibrate per-machine thresholds and report AUC/pAUC.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python scripts/train.py --max-files 2000 --epochs 50 --early-stopping 8
```

Fan-only:

```bash
python scripts/train.py --machine-types fan --max-files 0 --epochs 50
```

## Evaluate

```bash
python scripts/evaluate.py --checkpoint artifacts/runs/<run_name>/best_model.pt --manifests data/processed/manifests/dcase2024_development.csv
```

## Run API

```bash
set MODEL_CHECKPOINT=artifacts/runs/<run_name>/best_model.pt
set THRESHOLDS_JSON=artifacts/runs/<run_name>/evaluation.json
uvicorn app.main:app --reload
```
# Acoustic Anomaly Detection (DCASE Pipeline)

This project implements a defendable DCASE-style anomaly detection pipeline:

- 16 kHz mono audio loading with fixed duration
- log-mel features (`n_fft=1024`, `hop=512`, `n_mels=128`)
- global z-score stats computed on **normal train only**
- spectrogram windowing (`64` frame window, stride `32`)
- convolutional autoencoder with linear output + MSE loss
- per-file anomaly score from mean window reconstruction error
- per-machine threshold calibration and AUC/pAUC evaluation
- FastAPI endpoint for `/score`

## Files

- `config.py` - pipeline defaults
- `preprocess.py` - loading, features, normalization, windowing
- `dataset.py` - manifest handling and window dataset
- `model.py` - convolutional autoencoder
- `train.py` - training + checkpoint saving
- `evaluate.py` - threshold calibration + AUC/pAUC metrics
- `api.py` - inference API

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python train.py --max-files 2000 --epochs 50 --early-stopping 8
```

Optional fan-only:

```bash
python train.py --machine-types fan --max-files 0 --epochs 50
```

## Evaluate

```bash
python evaluate.py --checkpoint artifacts/runs/<run_name>/best_model.pt --manifests data/processed/manifests/dcase2024_development.csv
```

This writes `evaluation.json` next to the checkpoint.

## Run API

```bash
set MODEL_CHECKPOINT=artifacts/runs/<run_name>/best_model.pt
set THRESHOLDS_JSON=artifacts/runs/<run_name>/evaluation.json
uvicorn api:app --reload
```

Open docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
# First-Shot Acoustic Anomaly Detection Under Domain Shift

Starter repository for an end-to-end acoustic anomaly detection product:
- ML pipeline for anomaly scoring under domain shift
- FastAPI backend for inference and explanations
- Next.js frontend for modern visualization UI

## Project structure (what lives where)

```text
acoustic-anomaly-detection/
├── backend/
│   ├── app/              # FastAPI (health, /api/v1/score, /api/v1/stats)
│   ├── ml/               # mel features, MelConvAutoencoder, inference, metrics
│   └── tests/            # pytest (unit tests — not “evaluation JSON”)
├── data/
│   ├── raw/              # unpacked DCASE / MIMII audio (large; often gitignored)
│   └── processed/manifests/   # CSVs pointing to wav paths
├── scripts/              # CLI: build_manifest, train, score_wav, eval_thresholds
├── checkpoints/mel_conv_ae/friend_run/
│   ├── best_model.pt     # your trained weights (keep; not in git)
│   └── README.txt
└── frontend/             # Next.js UI (optional for the course demo)
```

### Trained weights

Copy `best_model.pt` into `checkpoints/mel_conv_ae/friend_run/` (same folder as `.gitkeep`).  
Weight files (`*.pt`) are ignored by git — back them up via Drive/USB if needed.

## Quick Start

### Backend

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` for UI and `http://localhost:8000/docs` for API docs.

## What is already done

- **Data:** manifests for DCASE development / additional / MIMII (paths in `data/processed/manifests/`).
- **Training:** `scripts/train.py` — log-mel conv autoencoder on normal clips (z-score + checkpoint).
- **Inference:** `backend/ml/inference.py` — anomaly score (+ optional top‑k mel bands).
- **Evaluation:** `scripts/eval_thresholds.py` — per‑machine thresholds, AUC/pAUC, optional `--global-metrics`.
- **CLI scoring:** `scripts/score_wav.py`.
- **API:** `POST /api/v1/score` (set `MEL_AE_CHECKPOINT`, optional `THRESHOLDS_JSON`), `GET /api/v1/stats`.
- **Weights:** `checkpoints/mel_conv_ae/friend_run/best_model.pt` (you kept this; threshold JSONs were removed as run outputs).

## What you still have to do (course / PDF milestones)

Use this as a checklist — you do **not** need everything for a passing grade; pick what your instructor emphasized.

1. **Thresholds again (after cleanup)** — Run `eval_thresholds.py`, write a new `thresholds.json` next to the checkpoint if you need decisions + F1. Tune percentile on **train/val**, not by peeking at the final test if you want clean methodology.
2. **One improvement** — e.g. `--percentile 95`, `--topk-mels 16`, or Mahalanobis on latents — compare AUC/pAUC on **development** test.
3. **Demo the API** — `uvicorn backend.app.main:app --reload`, upload a wav to `/api/v1/score` (see `/docs`).
4. **Optional:** wire the **frontend** to the API; add a short **monitoring** note (what you’d track in production — drift, score mean).
5. **Write-up:** problem → data → method → metrics → limitations (domain shift, pooled vs per‑machine).

If you feel stuck, do **(1) thresholds + one table of per‑machine metrics on DCASE dev** and **(3) API screenshot** — that already tells a complete story.
