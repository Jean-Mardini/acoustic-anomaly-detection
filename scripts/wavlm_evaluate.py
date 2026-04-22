"""
WavLM frozen feature extraction + anomaly scoring.
Step A: frozen WavLM embeddings + GMM/LOF/Mahalanobis per machine.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aad.config import AudioConfig
from aad.dataset import collect_file_records
from aad.evaluate_utils import fit_mahalanobis, partial_auc_roc
from aad.preprocess import load_audio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WavLM frozen features + anomaly scoring.")
    p.add_argument("--manifests", nargs="*", default=["data/processed/manifests/dcase2024_development.csv"])
    p.add_argument("--machine-types", nargs="*", default=None)
    p.add_argument("--calibrate-split", default="train")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--scorer", choices=["gmm", "lof", "mahalanobis"], default="gmm")
    p.add_argument("--gmm-components", type=int, default=32)
    p.add_argument("--lof-neighbors", type=int, default=20)
    p.add_argument("--model-name", default="microsoft/wavlm-base-plus")
    p.add_argument("--model-cache", type=Path, default=Path("models/wavlm"))
    p.add_argument("--max-fpr", type=float, default=0.1)
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


@torch.no_grad()
def extract_embedding(wav: np.ndarray, model, extractor, device: torch.device, sr: int = 16000) -> np.ndarray:
    inputs = extractor(wav, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    # Mean pool over time frames → [768]
    emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    return emb


def embed_files(records, model, extractor, audio_cfg, device, desc="Embedding") -> np.ndarray:
    embeddings = []
    for rec in tqdm(records, desc=desc, unit="file", leave=False):
        try:
            wav = load_audio(rec.audio_path, audio_cfg)
            emb = extract_embedding(wav, model, extractor, device)
            embeddings.append(emb)
        except Exception:
            continue
    return np.stack(embeddings) if embeddings else np.zeros((0, 768))


def fit_scorer(embeddings: np.ndarray, scorer: str, gmm_components: int, lof_neighbors: int):
    if scorer == "gmm":
        n = min(gmm_components, max(1, len(embeddings) // 2))
        gmm = GaussianMixture(n_components=n, covariance_type="full", max_iter=300, random_state=42)
        gmm.fit(embeddings)
        return gmm
    if scorer == "lof":
        lof = LocalOutlierFactor(n_neighbors=min(lof_neighbors, len(embeddings) - 1), novelty=True, contamination=0.1)
        lof.fit(embeddings)
        return lof
    # mahalanobis
    return fit_mahalanobis(embeddings)


def score_embedding(emb: np.ndarray, scorer_obj, scorer: str) -> float:
    if scorer == "gmm":
        return float(-scorer_obj.score_samples(emb.reshape(1, -1))[0])
    if scorer == "lof":
        return float(-scorer_obj.score_samples(emb.reshape(1, -1))[0])
    mu, inv_cov = scorer_obj
    diff = emb - mu
    return float(np.sqrt(diff @ inv_cov @ diff))


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading WavLM: {args.model_name}")
    extractor = AutoFeatureExtractor.from_pretrained(args.model_name, cache_dir=args.model_cache)
    model = AutoModel.from_pretrained(args.model_name, cache_dir=args.model_cache)
    model.to(device).eval()

    audio_cfg = AudioConfig()
    manifests = [Path(p) for p in args.manifests]
    machine_types = set(args.machine_types) if args.machine_types else None

    cal_records = collect_file_records(manifests, split=args.calibrate_split, labels={"normal"}, machine_types=machine_types)
    eval_records = collect_file_records(manifests, split=args.eval_split, labels={"normal", "anomaly"}, machine_types=machine_types)

    machines = sorted({r.machine_type for r in cal_records})
    result: dict = {"model": args.model_name, "scorer": args.scorer, "per_machine": {}}

    for mach in machines:
        cal_files = [r for r in cal_records if r.machine_type == mach]
        eval_files = [r for r in eval_records if r.machine_type == mach]

        print(f"\n[{mach}] Embedding {len(cal_files)} calibration files...")
        cal_emb = embed_files(cal_files, model, extractor, audio_cfg, device, desc=f"Cal {mach}")
        if len(cal_emb) < 5:
            result["per_machine"][mach] = {"note": "Too few calibration embeddings"}
            continue

        print(f"[{mach}] Fitting {args.scorer} on {len(cal_emb)} embeddings...")
        scorer_obj = fit_scorer(cal_emb, args.scorer, args.gmm_components, args.lof_neighbors)

        print(f"[{mach}] Scoring {len(eval_files)} eval files...")
        y_true, y_score = [], []
        for rec in tqdm(eval_files, desc=f"Eval {mach}", unit="file", leave=False):
            try:
                wav = load_audio(rec.audio_path, audio_cfg)
                emb = extract_embedding(wav, model, extractor, device)
                sc = score_embedding(emb, scorer_obj, args.scorer)
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
            "n_cal": len(cal_emb),
            "n_eval": len(y_true),
        }

    # Summary
    aucs = [v["auc_roc"] for v in result["per_machine"].values() if "auc_roc" in v]
    result["mean_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    print(f"\n=== MEAN AUC: {result['mean_auc']:.4f} ===")

    out = args.out_json or Path(f"artifacts/wavlm_{args.scorer}_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
