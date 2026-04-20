from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .config import AudioConfig, FeatureConfig, WindowConfig
from .preprocess import load_audio, waveform_to_log_mel, window_spectrogram, zscore


@dataclass(frozen=True)
class FileRecord:
    audio_path: Path
    machine_type: str
    section: str
    domain: str
    split: str
    label: str
    dataset_name: str


def read_manifest_rows(manifest_paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in manifest_paths]
    if not frames:
        raise ValueError("No manifest files provided.")
    df = pd.concat(frames, ignore_index=True)
    required = {"audio_path", "machine_type", "section", "domain", "split", "label", "dataset_name"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    return df


def collect_file_records(
    manifest_paths: list[Path],
    *,
    split: str,
    labels: set[str],
    machine_types: set[str] | None = None,
    max_files: int = 0,
    seed: int = 42,
) -> list[FileRecord]:
    df = read_manifest_rows(manifest_paths)
    mask = (df["split"].astype(str) == split) & df["label"].astype(str).isin(labels)
    if machine_types:
        mask &= df["machine_type"].astype(str).isin(machine_types)
    sub = df.loc[mask].copy()
    rng = np.random.default_rng(seed)
    if len(sub) > 0:
        sub = sub.iloc[rng.permutation(len(sub))]
    if max_files > 0:
        sub = sub.iloc[:max_files]

    out: list[FileRecord] = []
    for row in sub.itertuples(index=False):
        p = Path(str(row.audio_path))
        if not p.is_file():
            continue
        out.append(
            FileRecord(
                audio_path=p,
                machine_type=str(row.machine_type),
                section=str(row.section),
                domain=str(row.domain),
                split=str(row.split),
                label=str(row.label),
                dataset_name=str(row.dataset_name),
            )
        )
    return out


class WindowDataset(Dataset):
    def __init__(
        self,
        records: list[FileRecord],
        *,
        audio_cfg: AudioConfig,
        feature_cfg: FeatureConfig,
        window_cfg: WindowConfig,
        mean: float,
        std: float,
    ) -> None:
        self._items: list[tuple[np.ndarray, FileRecord, int]] = []
        for rec in tqdm(records, desc="Build windows", unit="file"):
            try:
                wav = load_audio(rec.audio_path, audio_cfg)
                mel = waveform_to_log_mel(wav, feature_cfg, sample_rate=audio_cfg.sample_rate)
                mel = zscore(mel, mean=mean, std=std)
                wins = window_spectrogram(mel, window_cfg)
            except Exception:
                continue
            for idx, w in enumerate(wins):
                self._items.append((w, rec, idx))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        w, rec, win_idx = self._items[idx]
        x = torch.from_numpy(w).unsqueeze(0)  # [1, mel, time]
        meta = {
            "audio_path": str(rec.audio_path),
            "machine_type": rec.machine_type,
            "section": rec.section,
            "domain": rec.domain,
            "split": rec.split,
            "label": rec.label,
            "dataset_name": rec.dataset_name,
            "window_index": win_idx,
        }
        return x, meta
