import numpy as np

from backend.ml.features import (
    DEFAULT_MEL_TIME_FRAMES,
    DEFAULT_SAMPLE_RATE,
    N_MELS,
    frame_length_samples,
    mel_pad_crop_time,
    waveform_to_log_mel,
)


def test_frame_length_16k() -> None:
    assert frame_length_samples(16_000) == 1024


def test_log_mel_shape_and_dtype() -> None:
    sr = DEFAULT_SAMPLE_RATE
    duration = 1.0
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    waveform = 0.1 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)

    spec = waveform_to_log_mel(waveform, sr)
    assert spec.ndim == 2
    assert spec.shape[0] == N_MELS
    assert spec.shape[1] >= 1
    assert spec.dtype == np.float32
    assert np.isfinite(spec).all()


def test_mel_pad_crop_time() -> None:
    spec = np.zeros((N_MELS, 10), dtype=np.float32)
    out = mel_pad_crop_time(spec, time_frames=DEFAULT_MEL_TIME_FRAMES)
    assert out.shape == (N_MELS, DEFAULT_MEL_TIME_FRAMES)
    long = np.random.randn(N_MELS, 200).astype(np.float32)
    out2 = mel_pad_crop_time(long, time_frames=DEFAULT_MEL_TIME_FRAMES)
    assert out2.shape == (N_MELS, DEFAULT_MEL_TIME_FRAMES)
