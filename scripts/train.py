from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aad.config import AudioConfig, FeatureConfig, TrainConfig, WindowConfig, to_dict
from aad.dataset import CachedWindowDataset, WindowDataset, collect_file_records
from aad.model import ConvAutoencoder
from aad.preprocess import compute_global_norm_stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train conv autoencoder for acoustic anomaly detection.")
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
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--early-stopping", type=int, default=15)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--num-memory-slots", type=int, default=100)
    p.add_argument("--memory-weight", type=float, default=0.002)
    p.add_argument("--entropy-weight", type=float, default=0.0)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/runs"))
    p.add_argument("--run-name", type=str, default="")
    p.add_argument(
        "--feature-manifests",
        nargs="*",
        default=None,
        help="Optional: pre-extracted features manifests (with feature_path column). "
             "If provided, skips raw WAV loading.",
    )
    p.add_argument(
        "--norm-stats",
        type=Path,
        default=None,
        help="Optional: path to norm_stats_*.json from preprocess.py. Skips recomputing stats.",
    )
    p.add_argument(
        "--per-file-norm",
        action="store_true",
        default=False,
        help="Normalize each spectrogram to its own mean/std instead of global stats.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_cfg = AudioConfig()
    feature_cfg = FeatureConfig()
    window_cfg = WindowConfig()
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        early_stopping=args.early_stopping,
        max_files=args.max_files,
        val_fraction=args.val_fraction,
        seed=args.seed,
        latent_dim=args.latent_dim,
    )

    machine_types = set(args.machine_types) if args.machine_types else None
    use_cache = args.feature_manifests is not None

    manifest_paths = [Path(p) for p in (args.feature_manifests if use_cache else args.manifests)]
    records = collect_file_records(
        manifest_paths,
        split="train",
        labels={"normal"},
        machine_types=machine_types,
        max_files=train_cfg.max_files,
        seed=train_cfg.seed,
    )
    if len(records) < 20:
        raise SystemExit(f"Too few valid train-normal files found: {len(records)}")

    rng = np.random.default_rng(train_cfg.seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    n_val = max(1, int(len(records) * train_cfg.val_fraction))
    val_idx = set(idx[:n_val].tolist())
    train_records = [records[i] for i in range(len(records)) if i not in val_idx]
    val_records = [records[i] for i in range(len(records)) if i in val_idx]

    if args.norm_stats is not None:
        import json as _json
        stats = _json.loads(Path(args.norm_stats).read_text())
        mean, std = float(stats["mean"]), float(stats["std"])
        print(f"Loaded norm stats: mean={mean:.6f}, std={std:.6f}")
    elif use_cache:
        from aad.preprocess import compute_global_norm_stats as _cgs
        print(f"Computing normalization stats from {len(train_records)} cached feature files...")
        sum_x, sum_x2, n_total = 0.0, 0.0, 0
        import numpy as _np
        for r in train_records:
            mel = _np.load(r.feature_path).astype(_np.float64)
            sum_x += float(mel.sum())
            sum_x2 += float(_np.square(mel).sum())
            n_total += int(mel.size)
        mean = sum_x / n_total
        var = max((sum_x2 / n_total) - mean ** 2, 0.0)
        std = float(_np.sqrt(var)) if var > 0 else 1e-6
        print(f"Normalization stats: mean={mean:.6f}, std={std:.6f}")
    else:
        print(f"Computing normalization stats from {len(train_records)} normal train files...")
        mean, std = compute_global_norm_stats(
            [r.audio_path for r in train_records],
            audio_cfg=audio_cfg,
            feature_cfg=feature_cfg,
        )
        print(f"Normalization stats: mean={mean:.6f}, std={std:.6f}")

    if use_cache:
        train_ds = CachedWindowDataset(train_records, window_cfg=window_cfg, mean=mean, std=std, per_file_norm=args.per_file_norm)
        val_ds = CachedWindowDataset(val_records, window_cfg=window_cfg, mean=mean, std=std, per_file_norm=args.per_file_norm)
    else:
        train_ds = WindowDataset(
            train_records, audio_cfg=audio_cfg, feature_cfg=feature_cfg,
            window_cfg=window_cfg, mean=mean, std=std,
        )
        val_ds = WindowDataset(
            val_records, audio_cfg=audio_cfg, feature_cfg=feature_cfg,
            window_cfg=window_cfg, mean=mean, std=std,
        )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("Window dataset is empty; check audio files and preprocessing settings.")

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=0)

    model = ConvAutoencoder(latent_dim=train_cfg.latent_dim, base_channels=64, num_memory_slots=args.num_memory_slots).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_cfg.epochs, eta_min=1e-6)
    loss_fn = nn.MSELoss()

    run_name = args.run_name.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = (args.out_dir / run_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    patience = 0
    best_state = None

    print(f"Train files={len(train_records)} val files={len(val_records)}")
    print(f"Train windows={len(train_ds)} val windows={len(val_ds)} device={device}")
    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for xb, _meta in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            out, z, z_hat, attn = model(xb)
            recon_loss = loss_fn(out, xb)
            mem_loss = loss_fn(z_hat, z.detach())
            entropy = -(attn * torch.log(attn + 1e-12)).sum(dim=1).mean()
            loss = recon_loss + args.memory_weight * mem_loss + args.entropy_weight * entropy
            loss.backward()
            opt.step()
            tr_loss += recon_loss.item() * xb.size(0)
            tr_n += xb.size(0)
        train_mse = tr_loss / max(tr_n, 1)

        model.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for xb, _meta in val_loader:
                xb = xb.to(device)
                out, z, z_hat, _ = model(xb)
                loss = loss_fn(out, xb)
                va_loss += loss.item() * xb.size(0)
                va_n += xb.size(0)
        val_mse = va_loss / max(va_n, 1)
        print(f"Epoch {epoch:03d}  train_mse={train_mse:.6f}  val_mse={val_mse:.6f}")

        scheduler.step()

        if val_mse < best_val - 1e-9:
            best_val = val_mse
            patience = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= train_cfg.early_stopping:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = out_dir / "best_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {"latent_dim": train_cfg.latent_dim, "base_channels": 64, "num_memory_slots": args.num_memory_slots},
            "norm": {"mean": mean, "std": std},
            "audio_config": to_dict(audio_cfg),
            "feature_config": to_dict(feature_cfg),
            "window_config": to_dict(window_cfg),
            "train_config": to_dict(train_cfg),
            "manifests": [str(p) for p in manifest_paths],
            "machine_types": sorted(machine_types) if machine_types else None,
            "best_val_mse": best_val,
            "per_file_norm": args.per_file_norm,
        },
        ckpt_path,
    )
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "checkpoint": str(ckpt_path),
                "best_val_mse": best_val,
                "train_files": len(train_records),
                "val_files": len(val_records),
                "train_windows": len(train_ds),
                "val_windows": len(val_ds),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
