from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aad.config import AudioConfig, FeatureConfig, WindowConfig
from aad.dataset import collect_file_records
from aad.evaluate_utils import collect_latents, fit_gmm, fit_mahalanobis, gmm_score_file, load_bundle, mahalanobis_score_file, partial_auc_roc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate checkpoint with per-machine thresholds and AUC/pAUC.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--manifests",
        nargs="*",
        default=["data/processed/manifests/dcase2024_development.csv"],
    )
    p.add_argument("--machine-types", nargs="*", default=None)
    p.add_argument("--calibrate-split", default="train")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--percentile", type=float, default=99.0)
    p.add_argument("--max-fpr", type=float, default=0.1)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--scorer", choices=["mahalanobis", "gmm", "domain_gmm", "tta_gmm"], default="mahalanobis")
    p.add_argument("--gmm-components", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_bundle(args.checkpoint, device)

    audio_cfg = AudioConfig(**ckpt["audio_config"])
    feature_cfg = FeatureConfig(**ckpt["feature_config"])
    window_cfg = WindowConfig(**ckpt["window_config"])
    mean = float(ckpt["norm"]["mean"])
    std = float(ckpt["norm"]["std"])
    per_file_norm = bool(ckpt.get("per_file_norm", False))
    machine_types = set(args.machine_types) if args.machine_types else None
    manifests = [Path(p) for p in args.manifests]

    cal_normals = collect_file_records(
        manifests,
        split=args.calibrate_split,
        labels={"normal"},
        machine_types=machine_types,
    )
    eval_records = collect_file_records(
        manifests,
        split=args.eval_split,
        labels={"normal", "anomaly"},
        machine_types=machine_types,
    )
    if not cal_normals:
        raise SystemExit("No calibration normal records found.")
    if not eval_records:
        raise SystemExit("No evaluation records found.")

    machines = sorted({r.machine_type for r in cal_normals})
    result: dict = {
        "checkpoint": str(args.checkpoint.resolve()),
        "percentile": args.percentile,
        "calibrate_split": args.calibrate_split,
        "eval_split": args.eval_split,
        "max_fpr": args.max_fpr,
        "per_machine": {},
    }

    for mach in machines:
        cal_files = [r for r in cal_normals if r.machine_type == mach]
        eval_files = [r for r in eval_records if r.machine_type == mach]

        shared = dict(audio_cfg=audio_cfg, feature_cfg=feature_cfg, window_cfg=window_cfg,
                      mean=mean, std=std, device=device, per_file_norm=per_file_norm)

        print(f"[{mach}] Collecting latents from {len(cal_files)} calibration files...")
        latents = collect_latents(model, cal_files, **shared)
        if len(latents) < 10:
            result["per_machine"][mach] = {"note": "Too few valid calibration latents."}
            continue

        if args.scorer == "gmm":
            scorer = fit_gmm(latents, n_components=args.gmm_components)

        elif args.scorer == "domain_gmm":
            # Separate GMM per domain, score = min(score_source, score_target)
            source_files = [r for r in cal_files if r.domain == "source"]
            target_files = [r for r in cal_files if r.domain == "target"]
            print(f"[{mach}] domain_gmm: {len(source_files)} source / {len(target_files)} target cal files")
            latents_src = collect_latents(model, source_files, **shared) if source_files else latents
            n_tgt = min(args.gmm_components, max(1, len(target_files) * 4))
            latents_tgt = collect_latents(model, target_files, **shared) if target_files else latents
            gmm_src = fit_gmm(latents_src, n_components=args.gmm_components)
            gmm_tgt = fit_gmm(latents_tgt, n_components=n_tgt)

        elif args.scorer == "tta_gmm":
            # Test-time adaptation: fit GMM on all test files (no labels used)
            print(f"[{mach}] tta_gmm: fitting GMM on {len(eval_files)} test files...")
            test_latents = collect_latents(model, eval_files, **shared)
            scorer = fit_gmm(test_latents, n_components=args.gmm_components)

        else:
            mu, inv_cov = fit_mahalanobis(latents)

        def score_rec(rec, _scorer=args.scorer):
            if _scorer == "gmm" or _scorer == "tta_gmm":
                return gmm_score_file(model, rec, **shared, gmm=scorer)
            if _scorer == "domain_gmm":
                s_src = gmm_score_file(model, rec, **shared, gmm=gmm_src)
                s_tgt = gmm_score_file(model, rec, **shared, gmm=gmm_tgt)
                if np.isfinite(s_src) and np.isfinite(s_tgt):
                    return min(s_src, s_tgt)
                return s_src if np.isfinite(s_src) else s_tgt
            return mahalanobis_score_file(model, rec, **shared, mu=mu, inv_cov=inv_cov)

        cal_scores: list[float] = []
        for rec in tqdm(cal_files, desc=f"Calibrate {mach}", unit="file", leave=False):
            sc = score_rec(rec)
            if np.isfinite(sc):
                cal_scores.append(sc)
        if len(cal_scores) < 5:
            result["per_machine"][mach] = {"note": "Too few valid calibration files."}
            continue
        thr = float(np.percentile(np.asarray(cal_scores), args.percentile))

        y_true: list[int] = []
        y_score: list[float] = []
        for rec in tqdm(eval_files, desc=f"Evaluate {mach}", unit="file", leave=False):
            sc = score_rec(rec)
            if not np.isfinite(sc):
                continue
            y_true.append(1 if rec.label == "anomaly" else 0)
            y_score.append(sc)

        if len(y_true) == 0 or len(set(y_true)) < 2:
            result["per_machine"][mach] = {
                "threshold": thr,
                "n_calibration_normals": len(cal_scores),
                "n_eval_scored": len(y_true),
                "note": "Need both normal and abnormal eval labels.",
            }
            continue
        yt = np.asarray(y_true, dtype=int)
        ys = np.asarray(y_score, dtype=np.float64)
        yp = (ys > thr).astype(int)
        auc = float(roc_auc_score(yt, ys))
        pauc = partial_auc_roc(yt, ys, max_fpr=args.max_fpr)
        bacc = float(balanced_accuracy_score(yt, yp))
        p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="binary", pos_label=1, zero_division=0)
        result["per_machine"][mach] = {
            "threshold": thr,
            "n_calibration_normals": len(cal_scores),
            "n_eval_scored": int(len(yt)),
            "auc_roc": auc,
            f"pauc_fpr_le_{args.max_fpr}": pauc,
            "balanced_accuracy_at_threshold": bacc,
            "precision_abnormal": float(p),
            "recall_abnormal": float(r),
            "f1_abnormal": float(f1),
        }

    out_json = args.out_json or (args.checkpoint.parent / "evaluation.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
