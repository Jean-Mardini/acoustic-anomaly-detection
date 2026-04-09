import numpy as np
import pytest
from pathlib import Path

from backend.ml.ingestion import (
    build_audio_metadata,
    discover_audio_files,
    validate_waveform,
)


def test_validate_waveform_accepts_valid_mono() -> None:
    waveform = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    validated = validate_waveform(waveform)
    assert validated.dtype == np.float32
    assert validated.ndim == 1
    assert validated.shape == (3,)


def test_validate_waveform_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_waveform(np.array([], dtype=np.float32))


def test_validate_waveform_rejects_non_mono() -> None:
    stereo_like = np.array([[0.1, -0.1], [0.2, -0.2]], dtype=np.float32)
    with pytest.raises(ValueError, match="1D mono"):
        validate_waveform(stereo_like)


def test_validate_waveform_rejects_non_finite() -> None:
    bad = np.array([0.1, np.nan, 0.2], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        validate_waveform(bad)


def test_build_audio_metadata_success() -> None:
    waveform = np.array([0.2, -0.1, 0.0, 0.1], dtype=np.float32)
    metadata = build_audio_metadata(
        audio_path="machine_01.wav",
        waveform=waveform,
        sample_rate=16000,
    )
    assert metadata.filename == "machine_01.wav"
    assert metadata.sample_rate == 16000
    assert metadata.num_samples == 4
    assert metadata.duration_seconds == pytest.approx(4 / 16000)


def test_build_audio_metadata_rejects_bad_sample_rate() -> None:
    waveform = np.array([0.1, -0.1], dtype=np.float32)
    with pytest.raises(ValueError, match="positive"):
        build_audio_metadata(
            audio_path="machine_bad.wav",
            waveform=waveform,
            sample_rate=0,
        )


def test_discover_audio_files_filters_supported_extensions(tmp_path: Path) -> None:
    (tmp_path / "normal").mkdir()
    (tmp_path / "abnormal").mkdir()

    wav_file = tmp_path / "normal" / "a.wav"
    flac_file = tmp_path / "abnormal" / "b.flac"
    txt_file = tmp_path / "normal" / "notes.txt"

    wav_file.write_bytes(b"fake wav")
    flac_file.write_bytes(b"fake flac")
    txt_file.write_text("ignore this")

    discovered = discover_audio_files(tmp_path)
    assert discovered == sorted([wav_file, flac_file])


def test_discover_audio_files_rejects_invalid_root(tmp_path: Path) -> None:
    bad_dir = tmp_path / "missing_dataset"
    with pytest.raises(FileNotFoundError, match="not found"):
        discover_audio_files(bad_dir)
