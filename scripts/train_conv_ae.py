"""
Train convolutional autoencoder on pooled normal clips (DCASE dev + additional + MIMII).

Threshold tuning on test set is NOT done here — use dev train/val split only.

Example:
  python scripts/train_conv_ae.py --max-files 2000 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.ml.datasets import MelSpectrogramFileDataset, gather_train_normal_paths
from backend.ml.features import DEFAULT_MEL_TIME_FRAMES, N_MELS
from backend.ml.model import MelConvAutoencoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MelConvAutoencoder on normal train clips.")
    parser.add_argument(
        "--manifests",
        type=str,
        nargs="*",
        default=[
            "data/processed/manifests/dcase2024_development.csv",
            "data/processed/manifests/dcase2024_additional.csv",
            "data/processed/manifests/mimii_due.csv",
        ],
        help="Manifest CSVs to merge (train + normal only)",
    )
    parser.add_argument("--max-files", type=int, default=4000, help="Max training files total")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Holdout fraction of files")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--early-stopping", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=8.0)
    parser.add_argument("--time-frames", type=int, default=DEFAULT_MEL_TIME_FRAMES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/mel_conv_ae"))
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()
    if args.time_frames != 128:
        raise SystemExit("Only time_frames=128 is supported for MelConvAutoencoder in this version.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mpaths = [Path(p) for p in args.manifests]
    for p in mpaths:
        if not p.is_file():
            raise SystemExit(f"Manifest not found: {p.resolve()}")

    all_paths = gather_train_normal_paths(mpaths, max_files=args.max_files, seed=args.seed)
    if len(all_paths) < 50:
        raise SystemExit(f"Too few files ({len(all_paths)}). Check manifests and data/raw paths.")

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(all_paths))
    rng.shuffle(idx)
    n_val = max(1, int(len(all_paths) * args.val_fraction))
    val_idx = set(idx[:n_val].tolist())
    train_idx = [i for i in range(len(all_paths)) if i not in val_idx]

    full_ds = MelSpectrogramFileDataset(
        all_paths,
        time_frames=args.time_frames,
        max_seconds=args.max_seconds,
    )
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, list(val_idx))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model_kwargs = {
        "n_mels": N_MELS,
        "time_frames": args.time_frames,
        "latent_dim": args.latent_dim,
        "base_channels": 32,
    }
    model = MelConvAutoencoder(**model_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    run_name = args.run_name.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = (args.out / run_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "manifests": [str(p) for p in mpaths],
        "max_files": args.max_files,
        "train_files": len(train_idx),
        "val_files": len(val_idx),
        "latent_dim": args.latent_dim,
        "time_frames": args.time_frames,
        "max_seconds": args.max_seconds,
        "device": str(device),
    }
    (out_dir / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    best_val = float("inf")
    patience = 0
    best_state = None

    print(f"Train files: {len(train_idx)}  Val files: {len(val_idx)}  Device: {device}", flush=True)
    print(f"Output: {out_dir}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} train", leave=False):
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * batch.size(0)
            tr_n += batch.size(0)
        train_mse = tr_loss / max(tr_n, 1)

        model.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} val", leave=False):
                batch = batch.to(device)
                out = model(batch)
                loss = loss_fn(out, batch)
                va_loss += loss.item() * batch.size(0)
                va_n += batch.size(0)
        val_mse = va_loss / max(va_n, 1)

        print(f"Epoch {epoch:3d}  train_mse={train_mse:.6f}  val_mse={val_mse:.6f}", flush=True)

        if val_mse < best_val - 1e-9:
            best_val = val_mse
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        if patience >= args.early_stopping:
            print("Early stopping.", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = out_dir / "best_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_kwargs": model_kwargs,
            "val_mse": best_val,
        },
        ckpt_path,
    )
    print(f"Saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
