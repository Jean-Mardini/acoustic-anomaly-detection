from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_curve
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from .config import AudioConfig, FeatureConfig, WindowConfig
from .dataset import FileRecord
from .model import ConvAutoencoder, TransformerAutoencoder
from .preprocess import load_audio, per_file_zscore, waveform_to_log_mel, window_spectrogram, zscore


def partial_auc_roc(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = 0.1) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    xs = np.append(fpr[fpr <= max_fpr], max_fpr)
    ys = np.append(tpr[fpr <= max_fpr], np.interp(max_fpr, fpr, tpr))
    area = np.trapz(ys, xs)
    return float(area / max_fpr)


def load_bundle(checkpoint: Path, device: torch.device) -> tuple:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model_type = ckpt.get("model_type", "conv")
    if model_type == "transformer":
        model = TransformerAutoencoder(**ckpt["model_config"])
    else:
        model = ConvAutoencoder(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def collect_latents(
    model: ConvAutoencoder,
    records: list[FileRecord],
    *,
    audio_cfg: AudioConfig,
    feature_cfg: FeatureConfig,
    window_cfg: WindowConfig,
    mean: float,
    std: float,
    device: torch.device,
    per_file_norm: bool = False,
) -> np.ndarray:
    all_latents: list[np.ndarray] = []
    for rec in tqdm(records, desc="Collecting latents", unit="file", leave=False):
        try:
            wav = load_audio(rec.audio_path, audio_cfg)
            mel = waveform_to_log_mel(wav, feature_cfg, sample_rate=audio_cfg.sample_rate)
            mel = per_file_zscore(mel) if per_file_norm else zscore(mel, mean=mean, std=std)
            windows = window_spectrogram(mel, window_cfg)
        except Exception:
            continue
        for w in windows:
            x = torch.from_numpy(w).unsqueeze(0).unsqueeze(0).to(device)
            z = model.encode(x)
            z_hat, _ = model.memory(z)
            all_latents.append(z_hat.cpu().numpy())
    if not all_latents:
        raise ValueError("No latents collected from calibration files.")
    return np.concatenate(all_latents, axis=0)  # [N_windows, latent_dim]


def fit_gmm(latents: np.ndarray, n_components: int = 10) -> GaussianMixture:
    gmm = GaussianMixture(n_components=n_components, covariance_type="diag", max_iter=200, random_state=42)
    gmm.fit(latents)
    return gmm


def gmm_score_file(
    model: ConvAutoencoder,
    rec: FileRecord,
    *,
    audio_cfg: AudioConfig,
    feature_cfg: FeatureConfig,
    window_cfg: WindowConfig,
    mean: float,
    std: float,
    device: torch.device,
    gmm: GaussianMixture,
    per_file_norm: bool = False,
) -> float:
    try:
        wav = load_audio(rec.audio_path, audio_cfg)
        mel = waveform_to_log_mel(wav, feature_cfg, sample_rate=audio_cfg.sample_rate)
        mel = per_file_zscore(mel) if per_file_norm else zscore(mel, mean=mean, std=std)
        windows = window_spectrogram(mel, window_cfg)
    except Exception:
        return float("nan")
    if not windows:
        return float("nan")
    scores: list[float] = []
    for w in windows:
        x = torch.from_numpy(w).unsqueeze(0).unsqueeze(0).to(device)
        z = model.encode(x)
        z_hat, _ = model.memory(z)
        z_hat = z_hat.detach().cpu().numpy()
        score = float(-gmm.score_samples(z_hat)[0])  # negative log-likelihood
        scores.append(score)
    return float(np.max(scores))


def fit_mahalanobis(latents: np.ndarray, reg: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    mu = latents.mean(axis=0)
    cov = np.cov(latents.T) + reg * np.eye(latents.shape[1])
    inv_cov = np.linalg.inv(cov)
    return mu, inv_cov


@torch.no_grad()
def mahalanobis_score_file(
    model: ConvAutoencoder,
    rec: FileRecord,
    *,
    audio_cfg: AudioConfig,
    feature_cfg: FeatureConfig,
    window_cfg: WindowConfig,
    mean: float,
    std: float,
    device: torch.device,
    mu: np.ndarray,
    inv_cov: np.ndarray,
    per_file_norm: bool = False,
) -> float:
    try:
        wav = load_audio(rec.audio_path, audio_cfg)
        mel = waveform_to_log_mel(wav, feature_cfg, sample_rate=audio_cfg.sample_rate)
        mel = per_file_zscore(mel) if per_file_norm else zscore(mel, mean=mean, std=std)
        windows = window_spectrogram(mel, window_cfg)
    except Exception:
        return float("nan")
    if not windows:
        return float("nan")
    scores: list[float] = []
    for w in windows:
        x = torch.from_numpy(w).unsqueeze(0).unsqueeze(0).to(device)
        z = model.encode(x)
        z_hat, _ = model.memory(z)
        z_hat = z_hat.cpu().numpy()[0]
        diff = z_hat - mu
        dist = float(np.sqrt(diff @ inv_cov @ diff))
        scores.append(dist)
    return float(np.max(scores))


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
