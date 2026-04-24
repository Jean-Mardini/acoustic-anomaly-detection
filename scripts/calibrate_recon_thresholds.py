from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aad.config import AudioConfig, FeatureConfig, WindowConfig
from aad.dataset import collect_file_records
from aad.evaluate_utils import load_bundle, partial_auc_roc, score_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibrate reconstruction thresholds per machine by optimizing F1 (recall-aware)."
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifests", nargs="*", default=["data/processed/manifests/dcase2024_development.csv"])
    p.add_argument("--split", default="test")
    p.add_argument("--machine-types", nargs="*", default=None)
    p.add_argument("--max-fpr", type=float, default=0.1)
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def _best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict[str, float]]:
    uniq = np.unique(y_score)
    if len(uniq) == 1:
        thr = float(uniq[0])
        y_pred = (y_score > thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
        return thr, {"precision_abnormal": float(p), "recall_abnormal": float(r), "f1_abnormal": float(f1)}

    mids = (uniq[:-1] + uniq[1:]) / 2.0
    candidates = np.concatenate(([uniq[0] - 1e-9], mids, [uniq[-1] + 1e-9]))

    best_thr = float(candidates[0])
    best_tuple = (-1.0, -1.0, -1.0)
    best_metrics = {"precision_abnormal": 0.0, "recall_abnormal": 0.0, "f1_abnormal": 0.0}
    for thr in candidates:
        y_pred = (y_score > thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            pos_label=1,
            zero_division=0,
        )
        # prioritize F1, then recall, then precision
        cur = (float(f1), float(r), float(p))
        if cur > best_tuple:
            best_tuple = cur
            best_thr = float(thr)
            best_metrics = {
                "precision_abnormal": float(p),
                "recall_abnormal": float(r),
                "f1_abnormal": float(f1),
            }
    return best_thr, best_metrics


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

    manifests = [Path(p) for p in args.manifests]
    machine_types = set(args.machine_types) if args.machine_types else None
    eval_records = collect_file_records(
        manifests,
        split=args.split,
        labels={"normal", "anomaly"},
        machine_types=machine_types,
    )
    if not eval_records:
        raise SystemExit("No evaluation records found.")

    machines = sorted({r.machine_type for r in eval_records})
    result: dict = {
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "objective": "maximize_f1_then_recall_then_precision",
        "max_fpr": args.max_fpr,
        "per_machine": {},
    }

    for mach in machines:
        records = [r for r in eval_records if r.machine_type == mach]
        y_true: list[int] = []
        y_score: list[float] = []
        for rec in tqdm(records, desc=f"Score {mach}", unit="file", leave=False):
            try:
                sc = score_file(
                    model,
                    rec,
                    audio_cfg=audio_cfg,
                    feature_cfg=feature_cfg,
                    window_cfg=window_cfg,
                    mean=mean,
                    std=std,
                    device=device,
                )
            except Exception:
                continue
            if not np.isfinite(sc):
                continue
            y_true.append(1 if rec.label == "anomaly" else 0)
            y_score.append(float(sc))

        if len(y_true) == 0 or len(set(y_true)) < 2:
            result["per_machine"][mach] = {
                "note": "Need both normal and anomaly samples for threshold optimization.",
                "n_eval_scored": len(y_true),
            }
            continue

        yt = np.asarray(y_true, dtype=int)
        ys = np.asarray(y_score, dtype=np.float64)
        thr, prf = _best_threshold(yt, ys)
        auc = float(roc_auc_score(yt, ys))
        pauc = float(partial_auc_roc(yt, ys, max_fpr=args.max_fpr))
        result["per_machine"][mach] = {
            "threshold": float(thr),
            "n_eval_scored": int(len(yt)),
            "auc_roc": auc,
            f"pauc_fpr_le_{args.max_fpr}": pauc,
            "precision_abnormal": prf["precision_abnormal"],
            "recall_abnormal": prf["recall_abnormal"],
            "f1_abnormal": prf["f1_abnormal"],
        }

    out_json = args.out_json or (args.checkpoint.parent / "thresholds_optimized_recon.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
