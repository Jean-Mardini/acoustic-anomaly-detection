"""
Log-mel spectrogram features (DCASE-style baseline settings).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import librosa

# Align with DCASE Task 2 baseline: 64 ms window, 50% hop, 128 mel bins
DEFAULT_SAMPLE_RATE = 16_000
N_MELS = 128
# Fixed time axis for conv2d autoencoder batches (pad short clips, crop long)
DEFAULT_MEL_TIME_FRAMES = 128


def frame_length_samples(sample_rate: int) -> int:
    """STFT window length for ~64 ms at the given sample rate."""
    return int(round(0.064 * sample_rate))


def waveform_to_log_mel(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    n_mels: int = N_MELS,
    n_fft: int | None = None,
    hop_length: int | None = None,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """
    Compute log-power mel spectrogram.

    Returns:
        Array shaped (n_mels, n_frames), float32, in dB (librosa power_to_db).
    """
    if n_fft is None:
        n_fft = frame_length_samples(sample_rate)
    if hop_length is None:
        hop_length = n_fft // 2
    if fmax is None:
        fmax = float(sample_rate) / 2.0

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32, copy=False)


def mel_pad_crop_time(log_mel: np.ndarray, time_frames: int = DEFAULT_MEL_TIME_FRAMES) -> np.ndarray:
    """Return log_mel with shape (n_mels, time_frames) by crop or right-pad with zeros."""
    if log_mel.ndim != 2:
        raise ValueError("log_mel must be 2D (n_mels, time).")
    n_mels, t = log_mel.shape
    if n_mels != N_MELS:
        raise ValueError(f"Expected n_mels={N_MELS}, got {n_mels}.")
    if t >= time_frames:
        out = log_mel[:, :time_frames]
    else:
        pad = np.zeros((n_mels, time_frames - t), dtype=np.float32)
        out = np.concatenate([log_mel.astype(np.float32, copy=False), pad], axis=1)
    return np.ascontiguousarray(out, dtype=np.float32)


def wav_to_fixed_mel(
    wav_path: str | Path,
    *,
    time_frames: int = DEFAULT_MEL_TIME_FRAMES,
    max_seconds: float | None = 8.0,
    target_sr: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """Load WAV, log-mel, pad/crop time to `time_frames`. Returns (n_mels, time_frames)."""
    kwargs: dict = {"sr": target_sr, "mono": True}
    if max_seconds is not None:
        kwargs["duration"] = max_seconds
    waveform, sr = librosa.load(Path(wav_path), **kwargs)
    lm = waveform_to_log_mel(waveform, sr)
    return mel_pad_crop_time(lm, time_frames)
