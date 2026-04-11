"""
Generate a manifest CSV from extracted audio under a dataset root.

Example (after unzipping DCASE additional training):
  python scripts/build_manifest.py ^
    --root "data/raw/dcase2024_additional/extracted/eval_data/raw" ^
    --dataset-name dcase2024_additional ^
    --out data/processed/manifests/dcase2024_additional.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.ml.manifest import build_manifest, write_manifest_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset manifest CSV.")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Folder to scan recursively for .wav/.flac (e.g. eval_data/raw).",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Label stored in CSV, e.g. dcase2024_additional",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/manifests/manifest.csv"),
        help="Output CSV path (default: data/processed/manifests/manifest.csv)",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Root is not a directory: {root}")

    records = build_manifest(root, args.dataset_name, validate_files=True)
    write_manifest_csv(records, args.out.resolve())
    print(f"Wrote {len(records)} rows to {args.out.resolve()}")


if __name__ == "__main__":
    main()
