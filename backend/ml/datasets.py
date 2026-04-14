"""Datasets for mel-spectrogram autoencoder training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from backend.ml.features import DEFAULT_MEL_TIME_FRAMES, wav_to_fixed_mel


def gather_train_normal_paths(
    manifest_paths: list[Path | str],
    *,
    max_files: int | None = None,
    seed: int = 42,
) -> list[Path]:
    """
    Collect audio paths from manifests: split=train, label=normal, file exists.
    Optionally shuffle and cap to max_files.
    """
    rng = np.random.default_rng(seed)
    paths: list[Path] = []
    for mp in manifest_paths:
        mp = Path(mp)
        if not mp.is_file():
            continue
        df = pd.read_csv(mp)
        if "audio_path" not in df.columns:
            continue
        m = pd.Series(True, index=df.index)
        if "split" in df.columns:
            m &= df["split"].astype(str) == "train"
        if "label" in df.columns:
            m &= df["label"].astype(str) == "normal"
        sub = df.loc[m, "audio_path"].astype(str).tolist()
        for p in sub:
            pp = Path(p)
            if pp.is_file():
                paths.append(pp.resolve())
    paths = list(dict.fromkeys(paths))
    rng.shuffle(paths)
    if max_files is not None:
        paths = paths[: int(max_files)]
    return paths


class MelSpectrogramFileDataset(Dataset):
    """One fixed-size log-mel per file (for conv AE)."""

    def __init__(
        self,
        audio_paths: list[Path],
        *,
        time_frames: int = DEFAULT_MEL_TIME_FRAMES,
        max_seconds: float | None = 8.0,
    ) -> None:
        self.paths = [Path(p) for p in audio_paths]
        self.time_frames = time_frames
        self.max_seconds = max_seconds

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        p = self.paths[idx]
        mel = wav_to_fixed_mel(p, time_frames=self.time_frames, max_seconds=self.max_seconds)
        # (1, n_mels, time) for Conv2d
        return torch.from_numpy(mel).unsqueeze(0)
