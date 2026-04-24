"""
BEATs anomaly scoring — per-machine GMM on file-level embeddings.

Step A: frozen BEATs (no --lora-ckpt)
Step B: BEATs + LoRA (--lora-ckpt artifacts/beats_lora/beats_lora.pt)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MODELS = ROOT / "models"
SCRIPTS = ROOT / "scripts"
for _p in (str(SRC), str(MODELS), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aad.config import AudioConfig
from aad.dataset import collect_file_records
from aad.evaluate_utils import partial_auc_roc
from aad.preprocess import load_audio
from BEATs import BEATs, BEATsConfig
from beats_train import LoRALinear, inject_lora, MachineAwareAdapter  # noqa: F401


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BEATs GMM anomaly scoring.")
    p.add_argument("--beats-ckpt", type=Path, default=Path("models/BEATs_iter3_plus_AS2M.pt"))
    p.add_argument(
        "--lora-ckpt", type=Path, default=None,
        help="Path to beats_lora.pt from beats_train.py. Omit for frozen BEATs (Step A).",
    )
    p.add_argument(
        "--manifests", nargs="*",
        default=["data/processed/manifests/dcase2024_development.csv"],
    )
    p.add_argument("--machine-types", nargs="*", default=None)
    p.add_argument("--calibrate-split", default="train")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--gmm-components", type=int, default=32)
    p.add_argument("--max-fpr", type=float, default=0.1)
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


@torch.no_grad()
def embed_file(
    wav_np: np.ndarray,
    beats: BEATs,
    device: torch.device,
    target_len: int = 160_000,
    mga: MachineAwareAdapter | None = None,
    machine_type: str | None = None,
) -> np.ndarray:
    import soundfile as sf
    wav = wav_np.astype(np.float32)
    if len(wav) < target_len:
        wav = np.tile(wav, int(np.ceil(target_len / len(wav))))
    wav = wav[:target_len]
    wav_t = torch.from_numpy(wav).unsqueeze(0).to(device)
    pad = torch.zeros(1, wav_t.size(1), dtype=torch.bool, device=device)
    feats, _ = beats.extract_features(wav_t, padding_mask=pad)
    emb = feats.mean(dim=1)  # [1, 768]
    if mga is not None and machine_type is not None:
        emb = mga(emb, [machine_type])
    return emb.squeeze(0).cpu().numpy()  # [768]


def load_beats(beats_ckpt: Path, lora_ckpt: Path | None) -> tuple:
    raw = torch.load(beats_ckpt, map_location="cpu")
    cfg = BEATsConfig(raw["cfg"])
    beats = BEATs(cfg)
    beats.load_state_dict(raw["model"])
    mga = None

    if lora_ckpt is not None:
        lora = torch.load(lora_ckpt, map_location="cpu")
        for p in beats.parameters():
            p.requires_grad_(False)
        inject_lora(beats, rank=lora["lora_rank"], alpha=lora["lora_alpha"])
        sd = beats.state_dict()
        sd.update(lora["lora_state"])
        beats.load_state_dict(sd)
        if "mga_state" in lora:
            mga = MachineAwareAdapter(
                lora["all_machine_types"], d_model=768,
                bottleneck=lora.get("mga_bottleneck", 64),
            )
            mga.load_state_dict(lora["mga_state"])
        mode = f"BEATs+LoRA+MGA(rank={lora['lora_rank']})"
    else:
        mode = "BEATs-frozen"

    return beats, mga, mode


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    beats, mga, mode = load_beats(args.beats_ckpt, args.lora_ckpt)
    beats.to(device).eval()
    if mga is not None:
        mga.to(device).eval()
    print(f"Mode: {mode}")

    audio_cfg = AudioConfig()
    target_len = int(audio_cfg.fixed_duration_sec * audio_cfg.sample_rate)

    manifests = [Path(m) for m in args.manifests]
    machine_types = set(args.machine_types) if args.machine_types else None

    cal_records = collect_file_records(
        manifests, split=args.calibrate_split,
        labels={"normal"}, machine_types=machine_types,
    )
    eval_records = collect_file_records(
        manifests, split=args.eval_split,
        labels={"normal", "anomaly"}, machine_types=machine_types,
    )

    machines = sorted({r.machine_type for r in cal_records})
    result: dict = {"mode": mode, "scorer": "gmm", "per_machine": {}}

    for mach in machines:
        cal_files = [r for r in cal_records if r.machine_type == mach]
        eval_files = [r for r in eval_records if r.machine_type == mach]

        # ── Calibration embeddings ──
        print(f"\n[{mach}] Embedding {len(cal_files)} calibration files...")
        cal_emb: list[np.ndarray] = []
        for rec in tqdm(cal_files, desc=f"Cal {mach}", leave=False):
            try:
                wav = load_audio(rec.audio_path, audio_cfg)
                cal_emb.append(embed_file(wav, beats, device, target_len, mga=mga, machine_type=mach))
            except Exception:
                continue
        if len(cal_emb) < 5:
            result["per_machine"][mach] = {"note": "Too few calibration embeddings"}
            continue
        cal_mat = np.stack(cal_emb)

        # ── Fit GMM ──
        n_comp = min(args.gmm_components, max(1, len(cal_mat) // 2))
        gmm = GaussianMixture(
            n_components=n_comp, covariance_type="full", max_iter=300, random_state=42
        )
        gmm.fit(cal_mat)
        print(f"[{mach}] GMM fitted ({n_comp} components on {len(cal_mat)} embeddings)")

        # ── Score eval files ──
        y_true: list[int] = []
        y_score: list[float] = []
        for rec in tqdm(eval_files, desc=f"Eval {mach}", leave=False):
            try:
                wav = load_audio(rec.audio_path, audio_cfg)
                emb = embed_file(wav, beats, device, target_len, mga=mga, machine_type=mach)
                sc = float(-gmm.score_samples(emb.reshape(1, -1))[0])
                if np.isfinite(sc):
                    y_true.append(1 if rec.label == "anomaly" else 0)
                    y_score.append(sc)
            except Exception:
                continue

        if len(y_true) == 0 or len(set(y_true)) < 2:
            result["per_machine"][mach] = {"note": "Need both normal and anomaly eval files"}
            continue

        yt = np.array(y_true)
        ys = np.array(y_score)
        auc = float(roc_auc_score(yt, ys))
        pauc = partial_auc_roc(yt, ys, max_fpr=args.max_fpr)
        print(f"[{mach}] AUC={auc:.4f}  pAUC={pauc:.4f}")
        result["per_machine"][mach] = {
            "auc_roc": auc,
            f"pauc_fpr_le_{args.max_fpr}": pauc,
            "n_cal": len(cal_mat),
            "n_eval": len(y_true),
        }

    aucs = [v["auc_roc"] for v in result["per_machine"].values() if "auc_roc" in v]
    result["mean_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    print(f"\n=== MEAN AUC: {result['mean_auc']:.4f} ===")

    suffix = "lora" if args.lora_ckpt else "frozen"
    out = args.out_json or Path(f"artifacts/beats_{suffix}_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
