from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_curve

from .config import AudioConfig, FeatureConfig, WindowConfig
from .dataset import FileRecord
from .model import ConvAutoencoder
from .preprocess import load_audio, waveform_to_log_mel, window_spectrogram, zscore


def partial_auc_roc(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = 0.1) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    xs = np.append(fpr[fpr <= max_fpr], max_fpr)
    ys = np.append(tpr[fpr <= max_fpr], np.interp(max_fpr, fpr, tpr))
    area = np.trapz(ys, xs)
    return float(area / max_fpr)


def load_bundle(checkpoint: Path, device: torch.device) -> tuple[ConvAutoencoder, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = ConvAutoencoder(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def score_file(
    model: ConvAutoencoder,
    rec: FileRecord,
    *,
    audio_cfg: AudioConfig,
    feature_cfg: FeatureConfig,
    window_cfg: WindowConfig,
    mean: float,
    std: float,
    device: torch.device,
) -> float:
    wav = load_audio(rec.audio_path, audio_cfg)
    mel = waveform_to_log_mel(wav, feature_cfg, sample_rate=audio_cfg.sample_rate)
    mel = zscore(mel, mean=mean, std=std)
    windows = window_spectrogram(mel, window_cfg)
    if not windows:
        return float("nan")
    scores: list[float] = []
    for w in windows:
        x = torch.from_numpy(w).unsqueeze(0).unsqueeze(0).to(device)
        out = model(x)
        mse = torch.mean((out - x) ** 2).item()
        scores.append(float(mse))
    return float(np.mean(scores))
