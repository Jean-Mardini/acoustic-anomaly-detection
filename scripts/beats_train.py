"""
BEATs + LoRA + Dual-Level Contrastive Loss (DLCL)
Pooled training on all machine types — exactly as DCASE 2024 winners.

Step B: fine-tune BEATs with LoRA adapters on Q/V/Out projections,
        supervised contrastive loss at file-level and frame-level,
        pseudo-labels = machine_type + section.
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
from aad.preprocess import load_audio
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
    """Extract only the trainable LoRA parameters."""
    return {k: v.cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}


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


# ── Supervised Contrastive Loss ───────────────────────────────────────────────

class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # features: [N, D] L2-normalized, labels: [N] integer class IDs
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
    def __init__(
        self,
        records,
        class_to_id: dict,
        audio_cfg: AudioConfig,
        sr: int = 16_000,
    ):
        self.records = records
        self.class_to_id = class_to_id
        self.audio_cfg = audio_cfg
        self.target_len = int(audio_cfg.fixed_duration_sec * sr)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        wav = load_audio(rec.audio_path, self.audio_cfg).astype(np.float32)
        if len(wav) < self.target_len:
            wav = np.tile(wav, int(np.ceil(self.target_len / len(wav))))
        wav = wav[: self.target_len]
        class_id = self.class_to_id[f"{rec.machine_type}_{rec.section}"]
        return torch.from_numpy(wav), class_id


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BEATs + LoRA + DLCL pooled training.")
    p.add_argument(
        "--manifests", nargs="*",
        default=[
            "data/processed/manifests/dcase2024_development.csv",
            "data/processed/manifests/dcase2024_additional.csv",
        ],
    )
    p.add_argument("--machine-types", nargs="*", default=None)
    p.add_argument("--beats-ckpt", type=Path, default=Path("models/BEATs_iter3_plus_AS2M.pt"))
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--frame-weight", type=float, default=0.5)
    p.add_argument("--num-frames", type=int, default=10,
                   help="Random frames sampled per file for frame-level loss.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2,
                   help="Gradient accumulation steps (effective batch = batch_size × grad_accum).")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr-lora", type=float, default=2e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--early-stopping", type=int, default=10)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/beats_lora"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers. Keep 0 to avoid librosa multiprocessing issues.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load BEATs and inject LoRA
    print(f"Loading BEATs from {args.beats_ckpt}")
    raw_ckpt = torch.load(args.beats_ckpt, map_location="cpu")
    cfg = BEATsConfig(raw_ckpt["cfg"])
    beats = BEATs(cfg)
    beats.load_state_dict(raw_ckpt["model"])
    for p in beats.parameters():
        p.requires_grad_(False)
    n_lora = inject_lora(beats, rank=args.lora_rank, alpha=args.lora_alpha)
    print(f"Injected LoRA into {n_lora} attention projections (rank={args.lora_rank})")

    trainable = sum(p.numel() for p in beats.parameters() if p.requires_grad)
    total = sum(p.numel() for p in beats.parameters())
    print(f"Trainable BEATs params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    beats.to(device)
    proj = ProjectionHead(in_dim=768, hidden_dim=512, out_dim=256).to(device)
    supcon = SupConLoss(temperature=args.temperature)

    # Collect train normal files
    audio_cfg = AudioConfig()
    manifests = [Path(m) for m in args.manifests]
    machine_types = set(args.machine_types) if args.machine_types else None
    records = collect_file_records(
        manifests, split="train", labels={"normal"}, machine_types=machine_types
    )
    print(f"Total train normal files: {len(records)}")

    # Build pseudo-class labels: machine_type + "_" + section
    classes = sorted({f"{r.machine_type}_{r.section}" for r in records})
    class_to_id = {c: i for i, c in enumerate(classes)}
    print(f"Pseudo-classes (machine × section): {len(classes)}")

    # Train / val split
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    n_val = max(1, int(len(records) * args.val_fraction))
    val_set = set(idx[:n_val].tolist())
    train_recs = [records[i] for i in range(len(records)) if i not in val_set]
    val_recs = [records[i] for i in range(len(records)) if i in val_set]
    print(f"Train: {len(train_recs)} files  Val: {len(val_recs)} files")

    train_ds = BEATsAudioDataset(train_recs, class_to_id, audio_cfg)
    val_ds = BEATsAudioDataset(val_recs, class_to_id, audio_cfg)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Optimiser: separate LRs for LoRA and projection head
    lora_params = [p for p in beats.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr_lora},
            {"params": proj.parameters(), "lr": args.lr_head},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-6
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    patience = 0
    best_lora_sd: dict | None = None
    best_proj_sd: dict | None = None

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        beats.train()
        proj.train()
        tr_loss, tr_n = 0.0, 0
        opt.zero_grad()
        for step, (wavs, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False)
        ):
            wavs = wavs.to(device)
            labels = labels.to(device)
            padding = torch.zeros(wavs.size(0), wavs.size(1), dtype=torch.bool, device=device)

            feats, _ = beats.extract_features(wavs, padding_mask=padding)  # [B, T, 768]

            # File-level: mean pool → project → SupCon
            file_emb = proj(feats.mean(dim=1))  # [B, 256]
            loss_file = supcon(file_emb, labels)

            # Frame-level: random K frames per file → project → SupCon
            T = feats.size(1)
            K = min(args.num_frames, T)
            fidx = torch.randint(0, T, (K,), device=device)
            frame_emb = proj(feats[:, fidx, :].reshape(-1, 768))  # [B*K, 256]
            frame_labels = labels.unsqueeze(1).expand(-1, K).reshape(-1)
            loss_frame = supcon(frame_emb, frame_labels)

            loss = (loss_file + args.frame_weight * loss_frame) / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(lora_params + list(proj.parameters()), 1.0)
                opt.step()
                opt.zero_grad()

            tr_loss += (loss_file + args.frame_weight * loss_frame).item()
            tr_n += 1

        # ── Validation ──
        beats.eval()
        proj.eval()
        va_loss, va_n = 0.0, 0
        with torch.no_grad():
            for wavs, labels in val_loader:
                wavs = wavs.to(device)
                labels = labels.to(device)
                padding = torch.zeros(wavs.size(0), wavs.size(1), dtype=torch.bool, device=device)
                feats, _ = beats.extract_features(wavs, padding_mask=padding)
                file_emb = proj(feats.mean(dim=1))
                va_loss += supcon(file_emb, labels).item()
                va_n += 1

        tr = tr_loss / max(tr_n, 1)
        va = va_loss / max(va_n, 1)
        scheduler.step()
        print(f"Epoch {epoch:03d}  train={tr:.4f}  val={va:.4f}")

        if va < best_val - 1e-6:
            best_val = va
            patience = 0
            best_lora_sd = lora_state_dict(beats)
            best_proj_sd = {k: v.cpu().clone() for k, v in proj.state_dict().items()}
        else:
            patience += 1
            if patience >= args.early_stopping:
                print("Early stopping.")
                break

    # Save checkpoint
    if best_lora_sd is None:
        best_lora_sd = lora_state_dict(beats)
        best_proj_sd = {k: v.cpu() for k, v in proj.state_dict().items()}

    out_path = args.out_dir / "beats_lora.pt"
    torch.save(
        {
            "lora_state": best_lora_sd,
            "proj_state": best_proj_sd,
            "beats_ckpt": str(args.beats_ckpt),
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "classes": classes,
            "best_val_loss": best_val,
        },
        out_path,
    )
    (args.out_dir / "train_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2)
    )
    trainable_lora = sum(p.numel() for p in beats.parameters() if p.requires_grad)
    print(f"Saved: {out_path}  (best_val={best_val:.4f}, LoRA params={trainable_lora:,})")


if __name__ == "__main__":
    main()
