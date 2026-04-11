import numpy as np

from backend.ml.features import DEFAULT_SAMPLE_RATE, N_MELS, frame_length_samples, waveform_to_log_mel


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
