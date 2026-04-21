"""Pre-extract log-mel spectrograms from raw WAVs and save as .npy files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aad.config import AudioConfig, FeatureConfig
from aad.preprocess import load_audio, waveform_to_log_mel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-extract log-mel features to .npy files.")
    p.add_argument(
        "--manifests",
        nargs="*",
        default=[
            "data/processed/manifests/dcase2024_development.csv",
            "data/processed/manifests/dcase2024_additional.csv",
            "data/processed/manifests/mimii_due.csv",
        ],
    )
    p.add_argument("--machine-types", nargs="*", default=None)
    p.add_argument("--out-dir", type=Path, default=Path("data/processed/features"))
    p.add_argument("--force", action="store_true", default=False, help="Re-extract even if .npy already exists.")
    return p.parse_args()


def feature_path_for(row: pd.Series, out_dir: Path) -> Path:
    stem = Path(row["audio_path"]).stem
    return out_dir / row["dataset_name"] / row["machine_type"] / row["split"] / f"{stem}.npy"


def extract_features(df: pd.DataFrame, out_dir: Path, audio_cfg: AudioConfig, feature_cfg: FeatureConfig, force: bool = False) -> pd.DataFrame:
    rows = df.copy()
    rows["feature_path"] = rows.apply(lambda r: str(feature_path_for(r, out_dir)), axis=1)

    errors = 0
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="Extracting features", unit="file"):
        fp = Path(row["feature_path"])
        if fp.exists() and not force:
            continue
        fp.parent.mkdir(parents=True, exist_ok=True)
        try:
            wav = load_audio(row["audio_path"], audio_cfg)
            mel = waveform_to_log_mel(wav, feature_cfg, sample_rate=audio_cfg.sample_rate)
            np.save(fp, mel)
        except Exception as e:
            errors += 1
            tqdm.write(f"  SKIP {row['audio_path']}: {e}")

    if errors:
        print(f"Warning: {errors} files failed and were skipped.")

    # Keep only rows where the .npy file exists
    rows["feature_path"] = rows["feature_path"].apply(
        lambda p: p if Path(p).exists() else ""
    )
    return rows[rows["feature_path"] != ""].copy()


def compute_norm_stats(df: pd.DataFrame) -> tuple[float, float]:
    train_normal = df[(df["split"] == "train") & (df["label"] == "normal")]
    print(f"Computing norm stats from {len(train_normal)} train-normal feature files...")
    sum_x = 0.0
    sum_x2 = 0.0
    n_total = 0
    for feat_path in tqdm(train_normal["feature_path"], desc="Norm stats", unit="file"):
        mel = np.load(feat_path).astype(np.float64)
        sum_x += float(mel.sum())
        sum_x2 += float(np.square(mel).sum())
        n_total += int(mel.size)
    if n_total == 0:
        raise ValueError("No train-normal features found for norm stats.")
    mean = sum_x / n_total
    var = max((sum_x2 / n_total) - mean ** 2, 0.0)
    std = float(np.sqrt(var)) if var > 0 else 1e-6
    return float(mean), float(std)


def main() -> None:
    args = parse_args()
    audio_cfg = AudioConfig()
    feature_cfg = FeatureConfig()

    frames = [pd.read_csv(Path(p)) for p in args.manifests]
    df = pd.concat(frames, ignore_index=True)

    if args.machine_types:
        df = df[df["machine_type"].isin(set(args.machine_types))].copy()

    print(f"Total files to process: {len(df)}")
    df = extract_features(df, args.out_dir, audio_cfg, feature_cfg, force=args.force)

    # Save one features manifest per source dataset
    manifest_dir = ROOT / "data" / "processed" / "manifests"
    for dataset_name, group in df.groupby("dataset_name"):
        out_path = manifest_dir / f"{dataset_name}_features.csv"
        group.reset_index(drop=True).to_csv(out_path, index=False)
        labels = group["label"].value_counts().to_dict()
        splits = group["split"].value_counts().to_dict()
        print(f"Saved {out_path.name}  rows={len(group)}  labels={labels}  splits={splits}")

    # Compute and save norm stats
    mean, std = compute_norm_stats(df)
    tag = "_".join(sorted(args.machine_types)) if args.machine_types else "all"
    stats_path = args.out_dir / f"norm_stats_{tag}.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps({"mean": mean, "std": std, "machine_types": args.machine_types}, indent=2),
        encoding="utf-8",
    )
    print(f"Norm stats → mean={mean:.6f}  std={std:.6f}  saved to {stats_path}")
    print("Preprocessing done.")


if __name__ == "__main__":
    main()
