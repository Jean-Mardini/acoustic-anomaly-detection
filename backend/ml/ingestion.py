from pathlib import Path
from dataclasses import dataclass

import librosa
import numpy as np

SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".flac")


@dataclass(frozen=True)
class AudioMetadata:
    filename: str
    sample_rate: int
    num_samples: int
    duration_seconds: float


def validate_waveform(waveform: np.ndarray) -> np.ndarray:
    """Validate waveform shape and values for downstream processing."""
    if waveform.size == 0:
        raise ValueError("Waveform is empty.")
    if waveform.ndim != 1:
        raise ValueError("Waveform must be a 1D mono signal.")
    if not np.isfinite(waveform).all():
        raise ValueError("Waveform contains non-finite values.")
    return waveform.astype(np.float32, copy=False)


def load_audio(
    audio_path: str | Path,
    target_sr: int = 16000,
    duration_seconds: float | None = None,
) -> tuple[np.ndarray, int]:
    """
    Load audio from disk as mono waveform with consistent sample rate.

    Returns:
        waveform: float32 waveform in range typically [-1.0, 1.0]
        sample_rate: target sample rate used for loading
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sample_rate = librosa.load(
        path=path,
        sr=target_sr,
        mono=True,
        duration=duration_seconds,
    )
    waveform = validate_waveform(waveform)
    return waveform, int(sample_rate)


def build_audio_metadata(
    audio_path: str | Path, waveform: np.ndarray, sample_rate: int
) -> AudioMetadata:
    """Create consistent metadata for loaded audio."""
    validated = validate_waveform(waveform)
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    num_samples = int(validated.shape[0])
    duration_seconds = num_samples / float(sample_rate)
    return AudioMetadata(
        filename=Path(audio_path).name,
        sample_rate=int(sample_rate),
        num_samples=num_samples,
        duration_seconds=float(duration_seconds),
    )


def discover_audio_files(
    dataset_dir: str | Path, extensions: tuple[str, ...] = SUPPORTED_AUDIO_EXTENSIONS
) -> list[Path]:
    """Discover audio files recursively from a dataset directory."""
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    if not root.is_dir():
        raise ValueError(f"Dataset path must be a directory: {root}")

    extension_set = {ext.lower() for ext in extensions}
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extension_set
    ]
    return sorted(files)
