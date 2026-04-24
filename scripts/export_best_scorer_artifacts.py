from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import LocalOutlierFactor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aad.config import AudioConfig, FeatureConfig, WindowConfig
from aad.dataset import collect_file_records
from aad.evaluate_utils import collect_latents, fit_gmm, fit_mahalanobis, load_bundle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export scorer artifacts for each run's best evaluation file only.")
    p.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    p.add_argument("--dcase-manifest", type=Path, default=Path("data/processed/manifests/dcase2024_development.csv"))
    p.add_argument("--mimii-manifest", type=Path, default=Path("data/processed/manifests/dcase2024_development.csv"))
    p.add_argument("--calibrate-split", default="train")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--gmm-components", type=int, default=10)
    p.add_argument("--lof-neighbors", type=int, default=20)
    p.add_argument("--max-cal-files", type=int, default=300)
    p.add_argument("--max-eval-files", type=int, default=600)
    p.add_argument("--limit", type=int, default=0, help="Optional limit for quick runs.")
    return p.parse_args()


def clean_machine(run_name: str, dataset: str) -> str:
    value = run_name.removesuffix("_best_v1").removesuffix("_transformer_v1")
    if dataset == "mimii" and value.lower().startswith("mimii_"):
        value = value[6:]
    value = re.sub(r"_v\d+$", "", value)
    return value


def score_key_from_eval(data: dict[str, Any]) -> tuple[float, float, float]:
    per_machine = data.get("per_machine", {})
    if not isinstance(per_machine, dict) or not per_machine:
        return (0.0, 0.0, 0.0)
    aucs: list[float] = []
    paucs: list[float] = []
    f1s: list[float] = []
    for vals in per_machine.values():
        if not isinstance(vals, dict):
            continue
        aucs.append(float(vals.get("auc_roc", 0.0)))
        paucs.append(float(vals.get("pauc_fpr_le_0.1", 0.0)))
        f1s.append(float(vals.get("f1_abnormal", 0.0)))
    if not aucs:
        return (0.0, 0.0, 0.0)
    return (sum(aucs) / len(aucs), sum(paucs) / len(paucs), sum(f1s) / len(f1s))


def pick_best_eval(run_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    best_path: Path | None = None
    best_payload: dict[str, Any] | None = None
    best_key = (-1.0, -1.0, -1.0)
    for p in sorted(run_dir.glob("evaluation*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = score_key_from_eval(payload)
        if key > best_key:
            best_key = key
            best_path = p
            best_payload = payload
    if best_path is None or best_payload is None:
        return None
    return best_path, best_payload


def method_from_eval_filename(name: str) -> str:
    if name == "evaluation.json":
        return "reconstruction"
    if "domain_gmm" in name:
        return "domain_gmm"
    if "tta_gmm" in name:
        return "tta_gmm"
    if "gmm" in name:
        return "gmm"
    if "mahalanobis" in name:
        return "mahalanobis"
    if "lof" in name:
        return "lof"
    return "unknown"


def _localized_manifest(manifest_path: Path) -> Path:
    """Create a temp manifest with audio paths rewritten to local ROOT when needed."""
    df = pd.read_csv(manifest_path)
    if "audio_path" not in df.columns:
        return manifest_path
    prefix = "/home/charbel-mezeraani/acoustic-anomaly-detection/"
    root_str = str(ROOT).replace("\\", "/").rstrip("/") + "/"
    s = df["audio_path"].astype(str)
    # only rewrite known legacy absolute prefix; keep other rows untouched
    s = s.str.replace(prefix, root_str, regex=False)
    s = s.str.replace("\\", "/", regex=False)
    df["audio_path"] = s
    fd, tmp_name = tempfile.mkstemp(prefix=f"{manifest_path.stem}_localized_", suffix=".csv")
    try:
        os.close(fd)
    except Exception:
        pass
    Path(tmp_name).unlink(missing_ok=True)
    out = Path(tmp_name)
    df.to_csv(out, index=False)
    return out


def main() -> None:
    args = parse_args()
    root = (ROOT / args.artifacts_root).resolve()
    if not root.exists():
        raise SystemExit(f"Artifacts root not found: {root}")

    ckpts = sorted(root.glob("*/*/*/best_model.pt"))
    if args.limit > 0:
        ckpts = ckpts[: args.limit]
    if not ckpts:
        raise SystemExit("No checkpoints found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary: list[dict[str, Any]] = []
    temp_manifests: list[Path] = []
    localized_dcase = _localized_manifest(args.dcase_manifest)
    localized_mimii = _localized_manifest(args.mimii_manifest)
    if localized_dcase != args.dcase_manifest:
        temp_manifests.append(localized_dcase)
    if localized_mimii != args.mimii_manifest:
        temp_manifests.append(localized_mimii)

    try:
        for ckpt_path in ckpts:
            run_dir = ckpt_path.parent
            dataset, architecture, run_name = ckpt_path.parts[-4], ckpt_path.parts[-3], ckpt_path.parts[-2]
            machine = clean_machine(run_name, dataset)

            best = pick_best_eval(run_dir)
            if best is None:
                summary.append({"run_dir": str(run_dir), "status": "skip_no_eval"})
                continue
            best_eval_path, best_eval_payload = best
            method = method_from_eval_filename(best_eval_path.name)
            per_machine = best_eval_payload.get("per_machine", {}).get(machine, {})
            threshold = float(per_machine["threshold"]) if isinstance(per_machine, dict) and "threshold" in per_machine else None

            model, ckpt = load_bundle(ckpt_path, device)
            audio_cfg = AudioConfig(**ckpt["audio_config"])
            feature_cfg = FeatureConfig(**ckpt["feature_config"])
            window_cfg = WindowConfig(**ckpt["window_config"])
            mean = float(ckpt["norm"]["mean"])
            std = float(ckpt["norm"]["std"])
            per_file_norm = bool(ckpt.get("per_file_norm", False))

            manifest = localized_mimii if dataset == "mimii" else localized_dcase
            if not manifest.is_file():
                summary.append({"run_dir": str(run_dir), "status": "skip_manifest_missing", "manifest": str(manifest)})
                continue

            machine_types = {machine}
            cal_normals = collect_file_records(
                [manifest],
                split=args.calibrate_split,
                labels={"normal"},
                machine_types=machine_types,
                max_files=args.max_cal_files,
            )
            eval_records = collect_file_records(
                [manifest],
                split=args.eval_split,
                labels={"normal", "anomaly"},
                machine_types=machine_types,
                max_files=args.max_eval_files,
            )
            shared = dict(
                audio_cfg=audio_cfg,
                feature_cfg=feature_cfg,
                window_cfg=window_cfg,
                mean=mean,
                std=std,
                device=device,
                per_file_norm=per_file_norm,
            )

            artifact_files: list[str] = []
            status = "ok"
            if method == "reconstruction":
                # no external scorer object required; checkpoint itself is the scorer
                status = "ok_reconstruction_no_extra_artifact"
            elif method == "gmm":
                try:
                    latents = collect_latents(model, cal_normals, **shared)
                except Exception:
                    status = "skip_no_latents_gmm"
                    latents = None
                if latents is not None:
                    gmm = fit_gmm(latents, n_components=args.gmm_components)
                    out = run_dir / "scorer_gmm.joblib"
                    joblib.dump(gmm, out)
                    artifact_files.append(out.name)
            elif method == "mahalanobis":
                try:
                    latents = collect_latents(model, cal_normals, **shared)
                except Exception:
                    status = "skip_no_latents_mahalanobis"
                    latents = None
                if latents is not None:
                    mu, inv_cov = fit_mahalanobis(latents)
                    out = run_dir / "scorer_mahalanobis.npz"
                    np.savez(out, mu=mu, inv_cov=inv_cov)
                    artifact_files.append(out.name)
            elif method == "lof":
                try:
                    latents = collect_latents(model, cal_normals, **shared)
                except Exception:
                    status = "skip_no_latents_lof"
                    latents = None
                if latents is None:
                    pass
                elif len(latents) < 3:
                    status = "skip_too_few_latents_for_lof"
                else:
                    n_neighbors = min(args.lof_neighbors, max(2, len(latents) - 1))
                    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=0.1)
                    lof.fit(latents)
                    out = run_dir / "scorer_lof.joblib"
                    joblib.dump(lof, out)
                    artifact_files.append(out.name)
            elif method == "domain_gmm":
                source_files = [r for r in cal_normals if r.domain == "source"]
                target_files = [r for r in cal_normals if r.domain == "target"]
                try:
                    latents_src = (
                        collect_latents(model, source_files, **shared)
                        if source_files
                        else collect_latents(model, cal_normals, **shared)
                    )
                    latents_tgt = (
                        collect_latents(model, target_files, **shared)
                        if target_files
                        else collect_latents(model, cal_normals, **shared)
                    )
                except Exception:
                    status = "skip_no_latents_domain_gmm"
                    latents_src, latents_tgt = None, None
                if latents_src is not None and latents_tgt is not None:
                    gmm_src = fit_gmm(latents_src, n_components=args.gmm_components)
                    n_tgt = min(args.gmm_components, max(1, len(target_files) * 4)) if target_files else args.gmm_components
                    gmm_tgt = fit_gmm(latents_tgt, n_components=n_tgt)
                    out = run_dir / "scorer_domain_gmm.joblib"
                    joblib.dump({"source": gmm_src, "target": gmm_tgt}, out)
                    artifact_files.append(out.name)
            elif method == "tta_gmm":
                try:
                    test_latents = collect_latents(model, eval_records, **shared)
                except Exception:
                    status = "skip_no_latents_tta_gmm"
                    test_latents = None
                if test_latents is not None:
                    gmm = fit_gmm(test_latents, n_components=args.gmm_components)
                    out = run_dir / "scorer_tta_gmm.joblib"
                    joblib.dump(gmm, out)
                    artifact_files.append(out.name)
            else:
                status = f"skip_unknown_method_{method}"

            meta = {
                "method": method,
                "best_eval_file": best_eval_path.name,
                "threshold": threshold,
                "machine_type": machine,
                "dataset": dataset,
                "architecture": architecture,
                "calibrate_split": args.calibrate_split,
                "eval_split": args.eval_split,
                "artifact_files": artifact_files,
            }
            meta_path = run_dir / "scorer_meta.json"
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            summary.append(
                {
                    "run_dir": str(run_dir),
                    "status": status,
                    "method": method,
                    "best_eval_file": best_eval_path.name,
                    "artifact_files": artifact_files,
                }
            )
            print(f"[{status}] {run_dir} -> {method} ({best_eval_path.name})")

        out_summary = root / "best_scorer_export_summary.json"
        out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote summary: {out_summary}")
    finally:
        for p in temp_manifests:
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
