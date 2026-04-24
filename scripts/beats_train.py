"""
BEATs + LoRA + Machine-Aware Adapters (MGA) + Dual-Level Contrastive Loss (DLCL)
Two-view SpecAugment for stronger contrastive signal.
Pooled training on all machine types — as the DCASE 2024 winners.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MODELS = ROOT / "models"
for _p in (str(SRC), str(MODELS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aad.config import AudioConfig
from aad.dataset import collect_file_records
from BEATs import BEATs, BEATsConfig


# ── LoRA ─────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 32, alpha: float = 32.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        d_in, d_out = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def inject_lora(
    model: nn.Module,
    rank: int = 32,
    alpha: float = 32.0,
    target_suffixes: tuple = ("q_proj", "v_proj", "out_proj"),
) -> int:
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(s) for s in target_suffixes):
            continue
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha))
        replaced += 1
    return replaced


def lora_state_dict(model: nn.Module) -> dict:
    return {k: v.cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}


# ── Machine-Aware Adapter (MGA) ───────────────────────────────────────────────

class MachineAwareAdapter(nn.Module):
    """One small bottleneck adapter per machine type, applied after BEATs pooling."""

    def __init__(self, machine_types: list[str], d_model: int = 768, bottleneck: int = 64):
        super().__init__()
        self.adapters = nn.ModuleDict({
            mt: nn.Sequential(
                nn.Linear(d_model, bottleneck),
                nn.ReLU(inplace=True),
                nn.Linear(bottleneck, d_model),
            )
            for mt in machine_types
        })

    def forward(self, x: torch.Tensor, machine_types: list[str]) -> torch.Tensor:
        # x: [B, D]  machine_types: list of B strings
        out = torch.stack([
            x[i] + self.adapters[mt](x[i])
            for i, mt in enumerate(machine_types)
        ])
        return out  # [B, D]


# ── Projection head ──────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 768, hidden_dim: int = 512, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ── SpecAugment on BEATs frame features ──────────────────────────────────────

def spec_augment(
    feats: torch.Tensor,
    time_mask_ratio: float = 0.15,
    feat_mask_ratio: float = 0.15,
) -> torch.Tensor:
    """Apply random time and feature masking to [B, T, D] features."""
    B, T, D = feats.shape
    out = feats.clone()
    # Time masking
    t_len = max(1, int(T * time_mask_ratio))
    t0 = torch.randint(0, max(1, T - t_len), (B,))
    for i in range(B):
        out[i, t0[i]: t0[i] + t_len, :] = 0.0
    # Feature masking
    f_len = max(1, int(D * feat_mask_ratio))
    f0 = torch.randint(0, max(1, D - f_len), (B,))
    for i in range(B):
        out[i, :, f0[i]: f0[i] + f_len] = 0.0
    return out


# ── Supervised Contrastive Loss (DLCL) ───────────────────────────────────────

class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        N = features.size(0)
        if N < 2:
            return features.sum() * 0.0

        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask.fill_diagonal_(0.0)

        sim = features @ features.T / self.temp
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)

        num_pos = mask.sum(dim=1).clamp(min=1)
        loss_per = -(mask * log_prob).sum(dim=1) / num_pos
        has_pos = mask.sum(dim=1) > 0
        if has_pos.sum() == 0:
            return features.sum() * 0.0
        return loss_per[has_pos].mean()


# ── Dataset ───────────────────────────────────────────────────────────────────

class BEATsAudioDataset(Dataset):
    def __init__(self, records, class_to_id: dict, audio_cfg: AudioConfig):
        self.records = records
        self.class_to_id = class_to_id
        self.target_len = int(audio_cfg.fixed_duration_sec * audio_cfg.sample_rate)
        self.audio_cfg = audio_cfg

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, path: str) -> np.ndarray:
        import soundfile as sf
        import warnings
        try:
            wav, _ = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception:
            import librosa
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wav, _ = librosa.load(str(path), sr=self.audio_cfg.sample_rate, mono=True)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return wav.astype(np.float32)

    def __getitem__(self, idx: int):
        for attempt in range(len(self.records)):
            rec = self.records[(idx + attempt) % len(self.records)]
            try:
                wav = self._load(rec.audio_path)
                if len(wav) < self.target_len:
                    wav = np.tile(wav, int(np.ceil(self.target_len / len(wav))))
                wav = wav[: self.target_len]
                class_id = self.class_to_id[f"{rec.machine_type}_{rec.section}"]
                return torch.from_numpy(wav), class_id, rec.machine_type
            except Exception:
                continue
        raise RuntimeError(f"Could not load any file at idx={idx}")


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BEATs + LoRA + MGA + DLCL pooled training.")
    p.add_argument("--manifests", nargs="*", default=[
        "data/processed/manifests/dcase2024_development.csv",
        "data/processed/manifests/dcase2024_additional.csv",
    ])
    p.add_argument("--machine-types", nargs="*", default=None)
    p.add_argument("--beats-ckpt", type=Path, default=Path("models/BEATs_iter3_plus_AS2M.pt"))
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--mga-bottleneck", type=int, default=64,
                   help="Bottleneck dim for Machine-Aware Adapter per machine type.")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--frame-weight", type=float, default=0.5)
    p.add_argument("--num-frames", type=int, default=10)
    p.add_argument("--time-mask", type=float, default=0.15,
                   help="SpecAugment time masking ratio.")
    p.add_argument("--feat-mask", type=float, default=0.15,
                   help="SpecAugment feature masking ratio.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Gradient accumulation steps (effective batch = batch_size × grad_accum).")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr-lora", type=float, default=2e-4)
    p.add_argument("--lr-mga", type=float, default=5e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--early-stopping", type=int, default=10)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/beats_lora"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── BEATs + LoRA ──
    print(f"Loading BEATs from {args.beats_ckpt}")
    raw_ckpt = torch.load(args.beats_ckpt, map_location="cpu")
    cfg = BEATsConfig(raw_ckpt["cfg"])
    beats = BEATs(cfg)
    beats.load_state_dict(raw_ckpt["model"])
    for p in beats.parameters():
        p.requires_grad_(False)
    n_lora = inject_lora(beats, rank=args.lora_rank, alpha=args.lora_alpha)
    print(f"LoRA injected into {n_lora} layers (rank={args.lora_rank})")
    lora_params = [p for p in beats.parameters() if p.requires_grad]
    trainable_lora = sum(p.numel() for p in lora_params)
    total = sum(p.numel() for p in beats.parameters())
    print(f"Trainable BEATs (LoRA): {trainable_lora:,} / {total:,} ({100*trainable_lora/total:.1f}%)")
    beats.to(device)

    # ── Data ──
    audio_cfg = AudioConfig()
    manifests = [Path(m) for m in args.manifests]
    machine_types_filter = set(args.machine_types) if args.machine_types else None
    records = collect_file_records(
        manifests, split="train", labels={"normal"}, machine_types=machine_types_filter
    )
    print(f"Total train normal files: {len(records)}")

    all_machine_types = sorted({r.machine_type for r in records})
    classes = sorted({f"{r.machine_type}_{r.section}" for r in records})
    class_to_id = {c: i for i, c in enumerate(classes)}
    print(f"Machine types: {len(all_machine_types)}  Pseudo-classes: {len(classes)}")

    # ── Machine-Aware Adapters ──
    mga = MachineAwareAdapter(all_machine_types, d_model=768, bottleneck=args.mga_bottleneck).to(device)
    mga_params = list(mga.parameters())
    print(f"MGA params: {sum(p.numel() for p in mga_params):,}  ({len(all_machine_types)} adapters)")

    # ── Projection head + Loss ──
    proj = ProjectionHead(in_dim=768, hidden_dim=512, out_dim=256).to(device)
    supcon = SupConLoss(temperature=args.temperature)

    # ── Train/val split ──
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    n_val = max(1, int(len(records) * args.val_fraction))
    val_set = set(idx[:n_val].tolist())
    train_recs = [records[i] for i in range(len(records)) if i not in val_set]
    val_recs = [records[i] for i in range(len(records)) if i in val_set]
    print(f"Train: {len(train_recs)}  Val: {len(val_recs)}  "
          f"Effective batch: {args.batch_size * args.grad_accum}")

    train_ds = BEATsAudioDataset(train_recs, class_to_id, audio_cfg)
    val_ds = BEATsAudioDataset(val_recs, class_to_id, audio_cfg)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # ── Optimiser ──
    opt = torch.optim.AdamW([
        {"params": lora_params,  "lr": args.lr_lora},
        {"params": mga_params,   "lr": args.lr_mga},
        {"params": proj.parameters(), "lr": args.lr_head},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    patience = 0
    start_epoch = 1
    best_lora_sd: dict | None = None
    best_mga_sd: dict | None = None
    best_proj_sd: dict | None = None

    all_trainable = lora_params + mga_params + list(proj.parameters())

    # ── Resume from checkpoint if available ──
    resume_path = args.out_dir / "checkpoint.pt"
    if resume_path.exists():
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        sd = beats.state_dict()
        sd.update(ckpt["lora_state"])
        beats.load_state_dict(sd)
        mga.load_state_dict(ckpt["mga_state"])
        proj.load_state_dict(ckpt["proj_state"])
        opt.load_state_dict(ckpt["opt_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        best_val = ckpt["best_val"]
        patience = ckpt["patience"]
        start_epoch = ckpt["epoch"] + 1
        best_lora_sd = ckpt["best_lora_sd"]
        best_mga_sd = ckpt["best_mga_sd"]
        best_proj_sd = ckpt["best_proj_sd"]
        print(f"Resumed at epoch {start_epoch}  best_val={best_val:.4f}  patience={patience}")

    for epoch in range(start_epoch, args.epochs + 1):
        # ── Train ──
        beats.train()
        mga.train()
        proj.train()
        tr_loss, tr_n = 0.0, 0
        opt.zero_grad()

        for step, (wavs, labels, mts) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False)
        ):
            wavs = wavs.to(device)
            labels = labels.to(device)
            pad = torch.zeros(wavs.size(0), wavs.size(1), dtype=torch.bool, device=device)

            # BEATs forward → [B, T, 768]
            feats, _ = beats.extract_features(wavs, padding_mask=pad)

            # Two augmented views via SpecAugment
            v1 = spec_augment(feats, args.time_mask, args.feat_mask)
            v2 = spec_augment(feats, args.time_mask, args.feat_mask)

            # Pool + MGA per view
            e1 = mga(v1.mean(dim=1), list(mts))   # [B, 768]
            e2 = mga(v2.mean(dim=1), list(mts))   # [B, 768]

            # Project both views → [2B, 256]
            p_file = proj(torch.cat([e1, e2], dim=0))
            l_file = torch.cat([labels, labels], dim=0)
            loss_file = supcon(p_file, l_file)

            # Frame-level on view1
            T = v1.size(1)
            K = min(args.num_frames, T)
            fidx = torch.randint(0, T, (K,), device=device)
            frame_raw = v1[:, fidx, :].reshape(-1, 768)        # [B*K, 768]
            frame_mts = [mt for mt in mts for _ in range(K)]
            frame_e = mga(frame_raw, frame_mts)                # [B*K, 768]
            p_frame = proj(frame_e)                            # [B*K, 256]
            l_frame = labels.unsqueeze(1).expand(-1, K).reshape(-1)
            loss_frame = supcon(p_frame, l_frame)

            loss = (loss_file + args.frame_weight * loss_frame) / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(all_trainable, 1.0)
                opt.step()
                opt.zero_grad()

            tr_loss += (loss_file + args.frame_weight * loss_frame).item()
            tr_n += 1

        # ── Validation — embed all, then score as one batch so SupCon has positive pairs ──
        beats.eval()
        mga.eval()
        proj.eval()
        val_embs: list[torch.Tensor] = []
        val_lbls: list[torch.Tensor] = []
        with torch.no_grad():
            for wavs, labels, mts in val_loader:
                wavs = wavs.to(device)
                labels = labels.to(device)
                pad = torch.zeros(wavs.size(0), wavs.size(1), dtype=torch.bool, device=device)
                feats, _ = beats.extract_features(wavs, padding_mask=pad)
                e = mga(feats.mean(dim=1), list(mts))
                val_embs.append(proj(e))
                val_lbls.append(labels)
        all_embs = torch.cat(val_embs, dim=0)
        all_lbls = torch.cat(val_lbls, dim=0)
        va = supcon(all_embs, all_lbls).item()

        tr = tr_loss / max(tr_n, 1)
        scheduler.step()
        tqdm.write(f"Epoch {epoch:03d}  train={tr:.4f}  val={va:.4f}")

        if va < best_val - 1e-6:
            best_val = va
            patience = 0
            best_lora_sd = lora_state_dict(beats)
            best_mga_sd = {k: v.cpu().clone() for k, v in mga.state_dict().items()}
            best_proj_sd = {k: v.cpu().clone() for k, v in proj.state_dict().items()}
            # Save checkpoint every time val loss improves
            torch.save({
                "epoch": epoch,
                "best_val": best_val,
                "patience": patience,
                "lora_state": lora_state_dict(beats),
                "mga_state": {k: v.cpu().clone() for k, v in mga.state_dict().items()},
                "proj_state": {k: v.cpu().clone() for k, v in proj.state_dict().items()},
                "best_lora_sd": best_lora_sd,
                "best_mga_sd": best_mga_sd,
                "best_proj_sd": best_proj_sd,
                "opt_state": opt.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }, resume_path)
            tqdm.write(f"  → checkpoint saved (best_val={best_val:.4f})")
        else:
            patience += 1
            if patience >= args.early_stopping:
                tqdm.write("Early stopping.")
                break

    # ── Save ──
    if best_lora_sd is None:
        best_lora_sd = lora_state_dict(beats)
        best_mga_sd = {k: v.cpu() for k, v in mga.state_dict().items()}
        best_proj_sd = {k: v.cpu() for k, v in proj.state_dict().items()}

    out_path = args.out_dir / "beats_lora.pt"
    torch.save({
        "lora_state": best_lora_sd,
        "mga_state": best_mga_sd,
        "proj_state": best_proj_sd,
        "beats_ckpt": str(args.beats_ckpt),
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "mga_bottleneck": args.mga_bottleneck,
        "all_machine_types": all_machine_types,
        "classes": classes,
        "best_val_loss": best_val,
    }, out_path)
    (args.out_dir / "train_config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    print(f"Saved: {out_path}  best_val={best_val:.4f}")


if __name__ == "__main__":
    main()
