"""
Calibrate per-machine-type thresholds on TRAIN normals and evaluate on TEST.

Uses reconstruction MSE from MelConvAutoencoder. Saves thresholds.json for deployment.

Example:
  python scripts/eval_thresholds.py ^
    --checkpoint checkpoints/mel_conv_ae/run/best_model.pt ^
    --manifest data/processed/manifests/dcase2024_development.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.ml.inference import load_mel_conv_ae, score_wav_file
from backend.ml.metrics_extra import partial_auc_roc, roc_auc_safe


def _score_paths(
    model,
    device: torch.device,
    paths: list[Path],
    *,
    max_seconds: float,
    time_frames: int,
) -> list[float]:
    out: list[float] = []
    for p in tqdm(paths, desc="Scoring", leave=False):
        if not p.is_file():
            continue
        mse, _, _ = score_wav_file(
            model,
            p,
            device=device,
            time_frames=time_frames,
            max_seconds=max_seconds,
        )
        if np.isfinite(mse):
            out.append(float(mse))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-machine thresholds + AUC/pAUC on dev test.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/manifests/dcase2024_development.csv"),
        help="Primary manifest (needs train normals + test normal/abnormal per machine)",
    )
    parser.add_argument("--calibrate-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Threshold = this percentile of CALIBRATION (train) normal scores per machine",
    )
    parser.add_argument("--max-seconds", type=float, default=8.0)
    parser.add_argument("--max-files-per-machine-cal", type=int, default=0, help="0 = all")
    parser.add_argument("--max-files-per-machine-test", type=int, default=0, help="0 = all")
    parser.add_argument("--pauc-max-fpr", type=float, default=0.1)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Where to write thresholds (default: next to checkpoint)",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not args.manifest.is_file():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_mel_conv_ae(args.checkpoint, device=device)
    mk = ckpt.get("model_kwargs", {})
    time_frames = int(mk.get("time_frames", 128))

    df = pd.read_csv(args.manifest)
    for col in ("machine_type", "split", "label", "audio_path"):
        if col not in df.columns:
            raise SystemExit(f"Manifest missing column: {col}")

    machines = sorted(df["machine_type"].astype(str).unique().tolist())
    result: dict = {
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "calibrate_split": args.calibrate_split,
        "eval_split": args.eval_split,
        "percentile": args.percentile,
        "max_seconds": args.max_seconds,
        "pauc_max_fpr": args.pauc_max_fpr,
        "per_machine": {},
    }

    print(
        f"Device: {device}  |  Calibrate on {args.calibrate_split} normals  "
        f"|  Evaluate on {args.eval_split}  |  percentile={args.percentile}",
        flush=True,
    )

    for m in machines:
        cal = df[
            (df["machine_type"].astype(str) == m)
            & (df["split"].astype(str) == args.calibrate_split)
            & (df["label"].astype(str) == "normal")
        ]
        ev = df[
            (df["machine_type"].astype(str) == m)
            & (df["split"].astype(str) == args.eval_split)
            & (df["label"].astype(str).isin({"normal", "abnormal"}))
        ]
        cal_paths = [Path(p) for p in cal["audio_path"].tolist()]
        test_paths = [(Path(r["audio_path"]), str(r["label"])) for _, r in ev.iterrows()]

        if args.max_files_per_machine_cal > 0:
            cal_paths = cal_paths[: args.max_files_per_machine_cal]
        if args.max_files_per_machine_test > 0:
            test_paths = test_paths[: args.max_files_per_machine_test]

        cal_paths = [p for p in cal_paths if p.is_file()]
        test_paths = [(p, lab) for p, lab in test_paths if p.is_file()]

        if len(cal_paths) < 5:
            print(f"  [{m}] SKIP calibrate: only {len(cal_paths)} train normals found.", flush=True)
            continue
        if len(test_paths) < 4:
            print(f"  [{m}] SKIP eval: only {len(test_paths)} test rows.", flush=True)
            continue

        print(f"  [{m}] cal n={len(cal_paths)}  test n={len(test_paths)}", flush=True)
        cal_scores = _score_paths(model, device, cal_paths, max_seconds=args.max_seconds, time_frames=time_frames)
        if len(cal_scores) < 5:
            print(f"  [{m}] SKIP: too few valid cal scores.", flush=True)
            continue
        thr = float(np.percentile(cal_scores, args.percentile))

        y_true: list[int] = []
        scores: list[float] = []
        for p, lab in tqdm(test_paths, desc=f"test {m}", leave=False):
            mse, _, _ = score_wav_file(
                model,
                p,
                device=device,
                time_frames=time_frames,
                max_seconds=args.max_seconds,
            )
            if not np.isfinite(mse):
                continue
            y_true.append(1 if lab == "abnormal" else 0)
            scores.append(float(mse))

        if len(set(y_true)) < 2:
            print(f"  [{m}] SKIP metrics: need both classes in test.", flush=True)
            result["per_machine"][m] = {"threshold": thr, "note": "insufficient test labels"}
            continue

        yt = np.array(y_true, dtype=int)
        sc = np.array(scores, dtype=np.float64)
        y_pred = (sc > thr).astype(int)

        auc_full = roc_auc_safe(yt, sc)
        pauc = partial_auc_roc(yt, sc, max_fpr=args.pauc_max_fpr)
        bacc = balanced_accuracy_score(yt, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            yt, y_pred, average="binary", pos_label=1, zero_division=0
        )

        result["per_machine"][m] = {
            "threshold": thr,
            "n_calibration_normals": len(cal_scores),
            "n_eval_scored": int(len(scores)),
            "auc_roc": auc_full,
            f"pauc_fpr_le_{args.pauc_max_fpr}": pauc,
            "balanced_accuracy_at_threshold": float(bacc),
            "precision_abnormal": float(prec),
            "recall_abnormal": float(rec),
            "f1_abnormal": float(f1),
        }
        print(
            f"    threshold={thr:.6f}  AUC={auc_full:.4f}  pAUC@{args.pauc_max_fpr}={pauc:.4f}  "
            f"BAcc={bacc:.4f}  F1(abn)={f1:.4f}",
            flush=True,
        )

    out_path = args.out_json
    if out_path is None:
        out_path = args.checkpoint.parent / "thresholds.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
