"""
Export BEATs frozen GMM scorers for use in a demo app.

Fits one GMM per machine type on training embeddings, then saves:
  export_dir/
    gmms/
      {machine_type}.pkl   ← fitted GMM scorer
    embeddings_info.json   ← machine types + GMM params
    README.txt             ← usage instructions

Usage:
  python3 scripts/beats_export_scorers.py \
    --manifests data/processed/manifests/dcase2024_development.csv \
    --beats-ckpt models/BEATs_iter3_plus_AS2M.pt \
    --out-dir artifacts/beats_frozen_export
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifests", nargs="*",
                   default=["data/processed/manifests/dcase2024_development.csv"])
    p.add_argument("--beats-ckpt", type=Path,
                   default=Path("models/BEATs_iter3_plus_AS2M.pt"))
    p.add_argument("--gmm-components", type=int, default=32)
    p.add_argument("--out-dir", type=Path,
                   default=Path("artifacts/beats_frozen_export"))
    return p.parse_args()


@torch.no_grad()
def embed_file(wav_np: np.ndarray, beats: BEATs, device: torch.device,
               target_len: int = 160_000) -> np.ndarray:
    wav = wav_np.astype(np.float32)
    if len(wav) < target_len:
        wav = np.tile(wav, int(np.ceil(target_len / len(wav))))
    wav = wav[:target_len]
    wav_t = torch.from_numpy(wav).unsqueeze(0).to(device)
    pad = torch.zeros(1, wav_t.size(1), dtype=torch.bool, device=device)
    feats, _ = beats.extract_features(wav_t, padding_mask=pad)
    return feats.mean(dim=1).squeeze(0).cpu().numpy()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load frozen BEATs
    print(f"Loading BEATs from {args.beats_ckpt}")
    raw = torch.load(args.beats_ckpt, map_location="cpu")
    cfg = BEATsConfig(raw["cfg"])
    beats = BEATs(cfg)
    beats.load_state_dict(raw["model"])
    beats.to(device).eval()

    audio_cfg = AudioConfig()
    target_len = int(audio_cfg.fixed_duration_sec * audio_cfg.sample_rate)
    manifests = [Path(m) for m in args.manifests]

    records = collect_file_records(manifests, split="train", labels={"normal"})
    machines = sorted({r.machine_type for r in records})
    print(f"Machine types: {machines}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gmms").mkdir(exist_ok=True)

    machine_info = {}

    for mach in machines:
        cal_files = [r for r in records if r.machine_type == mach]
        print(f"\n[{mach}] Embedding {len(cal_files)} files...")

        embs = []
        for rec in tqdm(cal_files, desc=mach, leave=False):
            try:
                wav = load_audio(rec.audio_path, audio_cfg)
                embs.append(embed_file(wav, beats, device, target_len))
            except Exception:
                continue

        if len(embs) < 5:
            print(f"[{mach}] Too few embeddings, skipping.")
            continue

        mat = np.stack(embs)
        n_comp = min(args.gmm_components, max(1, len(mat) // 2))
        gmm = GaussianMixture(n_components=n_comp, covariance_type="diag",
                              max_iter=300, random_state=42)
        gmm.fit(mat)
        print(f"[{mach}] GMM fitted ({n_comp} components on {len(mat)} embeddings)")

        gmm_path = args.out_dir / "gmms" / f"{mach}.pkl"
        with open(gmm_path, "wb") as f:
            pickle.dump(gmm, f)

        # Calibrate threshold on normal training scores
        train_scores = -gmm.score_samples(mat)  # higher = more anomalous
        threshold_95 = float(np.percentile(train_scores, 95))
        threshold_99 = float(np.percentile(train_scores, 99))
        print(f"[{mach}] threshold_95={threshold_95:.4f}  threshold_99={threshold_99:.4f}")

        machine_info[mach] = {
            "gmm_path": f"gmms/{mach}.pkl",
            "n_components": n_comp,
            "n_train_embeddings": len(mat),
            "threshold_95": threshold_95,
            "threshold_99": threshold_99,
        }

    # Save metadata
    meta = {
        "beats_ckpt": str(args.beats_ckpt.name),
        "gmm_components": args.gmm_components,
        "embedding_dim": 768,
        "target_len_samples": target_len,
        "sample_rate": audio_cfg.sample_rate,
        "machines": machine_info,
    }
    (args.out_dir / "embeddings_info.json").write_text(json.dumps(meta, indent=2))

    # Write README
    readme = """# BEATs Frozen Anomaly Detection — Export

## Files
- gmms/{machine}.pkl  : fitted GMM scorer per machine type
- embeddings_info.json: metadata (embedding dim, sample rate, etc.)
- BEATs_iter3_plus_AS2M.pt: pretrained BEATs checkpoint (copy here manually, 345MB)
- BEATs.py, backbone.py, modules.py: BEATs model code (copy from models/)

## Usage in your app
```python
import pickle, numpy as np, torch
from BEATs import BEATs, BEATsConfig

# Load BEATs
raw = torch.load("BEATs_iter3_plus_AS2M.pt", map_location="cpu")
beats = BEATs(BEATsConfig(raw["cfg"]))
beats.load_state_dict(raw["model"])
beats.eval()

# Load GMM for a machine type
with open("gmms/fan.pkl", "rb") as f:
    gmm = pickle.load(f)

# Score a 10s audio clip (numpy float32, 16kHz)
wav_t = torch.from_numpy(wav).unsqueeze(0)
pad = torch.zeros(1, wav_t.size(1), dtype=torch.bool)
with torch.no_grad():
    feats, _ = beats.extract_features(wav_t, padding_mask=pad)
emb = feats.mean(dim=1).numpy()
anomaly_score = float(-gmm.score_samples(emb)[0])  # higher = more anomalous
```
"""
    (args.out_dir / "README.txt").write_text(readme)

    print(f"\nExported to {args.out_dir}")
    print("Next: copy BEATs_iter3_plus_AS2M.pt and BEATs.py/backbone.py/modules.py into that folder.")


if __name__ == "__main__":
    main()
