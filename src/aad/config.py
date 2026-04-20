from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    mono: bool = True
    fixed_duration_sec: float = 10.0
    min_rms: float = 1e-5


@dataclass(frozen=True)
class FeatureConfig:
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 128
    power: float = 2.0
    log_eps: float = 1e-10
    use_db: bool = True


@dataclass(frozen=True)
class WindowConfig:
    window_size: int = 64
    stride: int = 32


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 50
    early_stopping: int = 8
    latent_dim: int = 256
    max_files: int = 0
    val_fraction: float = 0.1
    seed: int = 42
    num_workers: int = 0


def to_dict(obj: object) -> dict:
    return asdict(obj)
