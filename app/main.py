from __future__ import annotations

import json
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aad.config import AudioConfig, FeatureConfig, WindowConfig
from aad.dataset import FileRecord
from aad.evaluate_utils import load_bundle, score_file

app = FastAPI(title="Acoustic Anomaly Detection API")


@lru_cache(maxsize=1)
def _bundle() -> tuple:
    ckpt_path = Path(os.environ.get("MODEL_CHECKPOINT", "artifacts/latest/best_model.pt"))
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_bundle(ckpt_path, device)
    audio_cfg = AudioConfig(**ckpt["audio_config"])
    feature_cfg = FeatureConfig(**ckpt["feature_config"])
    window_cfg = WindowConfig(**ckpt["window_config"])
    mean = float(ckpt["norm"]["mean"])
    std = float(ckpt["norm"]["std"])
    return model, device, audio_cfg, feature_cfg, window_cfg, mean, std


@lru_cache(maxsize=1)
def _thresholds() -> dict[str, float] | None:
    p = os.environ.get("THRESHOLDS_JSON", "").strip()
    if not p:
        return None
    path = Path(p)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for m, vals in data.get("per_machine", {}).items():
        if isinstance(vals, dict) and "threshold" in vals:
            out[str(m)] = float(vals["threshold"])
    return out or None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score")
async def score(
    file: UploadFile = File(...),
    machine_type: str | None = Form(None),
) -> dict:
    try:
        model, device, audio_cfg, feature_cfg, window_cfg, mean, std = _bundle()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    suffix = (file.filename or "").lower()
    if not suffix.endswith((".wav", ".flac")):
        raise HTTPException(status_code=400, detail="Upload .wav or .flac file.")

    raw = await file.read()
    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_bytes(raw)
        rec = FileRecord(
            audio_path=tmp,
            machine_type=machine_type or "unknown",
            section="unknown",
            domain="unknown",
            split="inference",
            label="unknown",
            dataset_name="uploaded",
        )
        score_value = score_file(
            model,
            rec,
            audio_cfg=audio_cfg,
            feature_cfg=feature_cfg,
            window_cfg=window_cfg,
            mean=mean,
            std=std,
            device=device,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Scoring failed: {e}") from e
    finally:
        tmp.unlink(missing_ok=True)

    th_map = _thresholds()
    decision = None
    threshold = None
    if th_map and machine_type and machine_type in th_map:
        threshold = float(th_map[machine_type])
        decision = "abnormal" if score_value > threshold else "normal"
    return {
        "filename": file.filename,
        "anomaly_score": float(score_value),
        "machine_type": machine_type,
        "threshold": threshold,
        "decision": decision,
    }
