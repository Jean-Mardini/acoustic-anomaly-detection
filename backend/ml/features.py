"""
Log-mel spectrogram features (DCASE-style baseline settings).
"""

from __future__ import annotations

import numpy as np
import librosa

# Align with DCASE Task 2 baseline: 64 ms window, 50% hop, 128 mel bins
DEFAULT_SAMPLE_RATE = 16_000
N_MELS = 128


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
