from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

from .config import AudioConfig, FeatureConfig, WindowConfig


def load_audio(path: str | Path, audio_cfg: AudioConfig) -> np.ndarray:
    wav_path = Path(path)
    if not wav_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")
    y, _sr = librosa.load(
        wav_path,
        sr=audio_cfg.sample_rate,
        mono=audio_cfg.mono,
        duration=audio_cfg.fixed_duration_sec,
    )
    if y.size == 0:
        raise ValueError("Empty waveform.")
    if not np.isfinite(y).all():
        raise ValueError("Waveform has non-finite values.")
    rms = float(np.sqrt(np.mean(np.square(y), dtype=np.float64)))
    if rms < audio_cfg.min_rms:
        raise ValueError(f"Waveform too silent (rms={rms:.3e}).")
    return y.astype(np.float32, copy=False)


def waveform_to_log_mel(
    waveform: np.ndarray,
    feature_cfg: FeatureConfig,
    sample_rate: int,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=feature_cfg.n_fft,
        hop_length=feature_cfg.hop_length,
        n_mels=feature_cfg.n_mels,
        power=feature_cfg.power,
    )
    if feature_cfg.use_db:
        feat = librosa.power_to_db(mel, ref=np.max)
    else:
        feat = np.log(mel + feature_cfg.log_eps)
    return feat.astype(np.float32, copy=False)


def compute_global_norm_stats(
    file_paths: list[Path],
    audio_cfg: AudioConfig,
    feature_cfg: FeatureConfig,
) -> tuple[float, float]:
    sum_x = 0.0
    sum_x2 = 0.0
    n_total = 0
    for path in tqdm(file_paths, desc="Norm stats from normal train files", unit="file"):
        wav = load_audio(path, audio_cfg)
        mel = waveform_to_log_mel(wav, feature_cfg, sample_rate=audio_cfg.sample_rate)
        arr = mel.astype(np.float64, copy=False)
        sum_x += float(arr.sum())
        sum_x2 += float(np.square(arr).sum())
        n_total += int(arr.size)
    if n_total == 0:
        raise ValueError("No values available for normalization stats.")
    mean = sum_x / n_total
    var = max((sum_x2 / n_total) - (mean * mean), 0.0)
    std = float(np.sqrt(var))
    if std <= 0.0:
        std = 1e-6
    return float(mean), float(std)


def zscore(feat: np.ndarray, mean: float, std: float, eps: float = 1e-6) -> np.ndarray:
    out = (feat - mean) / (std + eps)
    return np.ascontiguousarray(out, dtype=np.float32)


def window_spectrogram(
    feat: np.ndarray,
    window_cfg: WindowConfig,
) -> list[np.ndarray]:
    if feat.ndim != 2:
        raise ValueError("Expected 2D log-mel feature matrix [mel, time].")
    n_mels, t = feat.shape
    if n_mels <= 0 or t <= 0:
        return []
    win = int(window_cfg.window_size)
    stride = int(window_cfg.stride)
    if t < win:
        pad = np.zeros((n_mels, win - t), dtype=np.float32)
        return [np.concatenate([feat.astype(np.float32, copy=False), pad], axis=1)]

    windows: list[np.ndarray] = []
    starts = list(range(0, t - win + 1, stride))
    if starts[-1] != (t - win):
        starts.append(t - win)
    for s in starts:
        windows.append(np.ascontiguousarray(feat[:, s : s + win], dtype=np.float32))
    return windows
