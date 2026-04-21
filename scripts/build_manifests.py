"""Rebuild manifest CSVs by scanning actual audio files on disk."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "manifests"
OUT.mkdir(parents=True, exist_ok=True)

COLUMNS = ["dataset_name", "machine_type", "section", "domain", "split", "label", "audio_path"]

# ── filename pattern shared by dcase2024 and mimii_due ──────────────────────
# e.g. section_00_source_train_normal_0001_pro_A.wav
#      section_00_target_test_anomaly_0039.wav
FNAME_RE = re.compile(
    r"^section_(?P<sec>\d+)_(?P<domain>source|target)_(?P<split>train|test)_(?P<label>normal|anomaly|unknown)"
)


def _parse_fname(stem: str) -> dict | None:
    m = FNAME_RE.match(stem)
    if not m:
        return None
    return {
        "section": f"section_{m['sec']}",
        "domain": m["domain"],
        "split": m["split"],
        "label": m["label"],
    }


def _folder_to_split_domain(folder: str) -> tuple[str, str]:
    """mimii_due uses folder names like train / source_test / target_test."""
    if folder == "train":
        return "train", "source"
    if folder == "source_test":
        return "test", "source"
    if folder == "target_test":
        return "test", "target"
    # dcase2024 uses plain 'test'
    return "test", "unknown"


# ── dcase2024_development ────────────────────────────────────────────────────
def build_dcase2024_development() -> pd.DataFrame:
    base = RAW / "dcase2024_development" / "unpacked"
    rows = []
    for wav in sorted(base.rglob("*.wav")):
        machine_type = wav.parts[len(base.parts)]
        parsed = _parse_fname(wav.stem)
        if parsed is None:
            continue
        rows.append({
            "dataset_name": "dcase2024_development",
            "machine_type": machine_type,
            **parsed,
            "audio_path": str(wav),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


# ── dcase2024_additional ─────────────────────────────────────────────────────
def build_dcase2024_additional() -> pd.DataFrame:
    base = RAW / "dcase2024_additional" / "unpacked"
    rows = []
    for wav in sorted(base.rglob("*.wav")):
        machine_type = wav.parts[len(base.parts)]
        parsed = _parse_fname(wav.stem)
        if parsed is None:
            continue
        rows.append({
            "dataset_name": "dcase2024_additional",
            "machine_type": machine_type,
            **parsed,
            "audio_path": str(wav),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


# ── mimii_due ────────────────────────────────────────────────────────────────
def build_mimii_due() -> pd.DataFrame:
    base = RAW / "mimii_due" / "unpacked"
    rows = []
    for wav in sorted(base.rglob("*.wav")):
        rel = wav.relative_to(base)
        machine_type = rel.parts[0]
        folder = rel.parts[1]
        split, domain_fallback = _folder_to_split_domain(folder)
        parsed = _parse_fname(wav.stem)
        if parsed is None:
            continue
        # filename domain/split should match folder; trust filename when available
        rows.append({
            "dataset_name": "mimii_due",
            "machine_type": machine_type,
            "section": parsed["section"],
            "domain": parsed["domain"],
            "split": split,
            "label": parsed["label"],
            "audio_path": str(wav),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def main() -> None:
    builders = [
        ("dcase2024_development", build_dcase2024_development),
        ("dcase2024_additional", build_dcase2024_additional),
        ("mimii_due", build_mimii_due),
    ]
    for name, fn in builders:
        print(f"Building {name}...", end=" ", flush=True)
        df = fn()
        dest = OUT / f"{name}.csv"
        df.to_csv(dest, index=False)
        label_counts = df["label"].value_counts().to_dict() if "label" in df.columns else {}
        split_counts = df["split"].value_counts().to_dict() if "split" in df.columns else {}
        print(f"{len(df)} rows | labels={label_counts} | splits={split_counts}")

    # Remove stale evaluation manifest (no data on disk)
    stale = OUT / "dcase2024_evaluation.csv"
    if stale.exists():
        stale.unlink()
        print("Removed dcase2024_evaluation.csv (no data on disk)")

    print("Done.")


if __name__ == "__main__":
    main()
