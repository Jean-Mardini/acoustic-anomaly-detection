from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MODELS = ROOT / "models"
ARTIFACTS_ROOT = ROOT / "artifacts"
BEATS_EXPORT_ROOT = ROOT / "beats_frozen_export"
MIN_RECALL = float(os.environ.get("MIN_RECALL_ABNORMAL", "0.15"))
MIN_F1 = float(os.environ.get("MIN_F1_ABNORMAL", "0.15"))
ENABLE_QUALITY_GATE = os.environ.get("ENABLE_QUALITY_GATE", "0").strip() not in {"0", "false", "False"}
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(MODELS) not in sys.path:
    sys.path.insert(0, str(MODELS))

from aad.config import AudioConfig, FeatureConfig, WindowConfig
from aad.dataset import FileRecord
from aad.evaluate_utils import gmm_score_file, load_bundle, mahalanobis_score_file, score_file
from aad.preprocess import load_audio

app = FastAPI(title="Acoustic Anomaly Detection API")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def _clean_machine(machine: str, dataset: str) -> str:
    value = machine.strip()
    if dataset == "mimii" and value.lower().startswith("mimii_"):
        value = value[6:]
    value = re.sub(r"_v\d+$", "", value)
    return value


def _score_key_from_eval(data: dict[str, Any]) -> tuple[float, float, float]:
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


def _pick_best_eval(run_dir: Path) -> dict[str, Any] | None:
    best_payload: dict[str, Any] | None = None
    best_name = ""
    best_key = (-1.0, -1.0, -1.0)
    for p in sorted(run_dir.glob("evaluation*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = _score_key_from_eval(payload)
        if key > best_key:
            best_key = key
            best_payload = payload
            best_name = p.name
    if best_payload is None:
        return None
    return {
        "eval_file": best_name,
        "eval_key": best_key,
        "payload": best_payload,
    }


def _method_from_eval_file(eval_file: str | None) -> str:
    if not eval_file:
        return "reconstruction"
    name = eval_file.lower()
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
    return "reconstruction"


def _pick_optimized_threshold(run_dir: Path, machine: str) -> tuple[float, str] | None:
    p = run_dir / "thresholds_optimized_recon.json"
    if not p.is_file():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    per_machine = payload.get("per_machine", {})
    if not isinstance(per_machine, dict):
        return None
    vals = per_machine.get(machine, {})
    if not isinstance(vals, dict) or "threshold" not in vals:
        return None
    return float(vals["threshold"]), p.name


def _resolve_beats_ckpt() -> Path:
    configured = os.environ.get("BEATS_CHECKPOINT", "").strip()
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            ROOT / "models" / "BEATs_iter3_plus_AS2M.pt",
            ROOT / "beats_frozen_export" / "BEATs_iter3_plus_AS2M.pt",
            ROOT.parent / "BEATs_iter3_plus_AS2M.pt",
        ]
    )
    for path in candidates:
        p = path.expanduser()
        if p.is_file():
            return p
    raise FileNotFoundError(
        "BEATs checkpoint not found. Set BEATS_CHECKPOINT or place BEATs_iter3_plus_AS2M.pt "
        "in models/, beats_frozen_export/, or the repo parent folder."
    )


def _load_beats_export_meta() -> dict[str, Any] | None:
    p = BEATS_EXPORT_ROOT / "embeddings_info.json"
    if not p.is_file():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_beats_eval_meta() -> dict[str, Any] | None:
    p = BEATS_EXPORT_ROOT / "beats_frozen_results.json"
    if not p.is_file():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


@lru_cache(maxsize=1)
def _catalog() -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    datasets: dict[str, dict[str, Any]] = {}

    if ARTIFACTS_ROOT.exists():
        for ckpt in sorted(ARTIFACTS_ROOT.glob("*/*/*/best_model.pt")):
            try:
                dataset, architecture, run_name = ckpt.parts[-4], ckpt.parts[-3], ckpt.parts[-2]
            except Exception:
                continue
            machine_raw = run_name.removesuffix("_best_v1").removesuffix("_transformer_v1")
            machine = _clean_machine(machine_raw, dataset)
            best_eval = _pick_best_eval(ckpt.parent)
            model_entry: dict[str, Any] = {
                "dataset": dataset,
                "architecture": architecture,
                "machine_type": machine,
                "run_name": run_name,
                "checkpoint": _rel(ckpt),
                "best_eval_file": None,
                "best_method": "reconstruction",
                "best_eval_metrics": None,
                "quality_gate": {
                    "enabled": ENABLE_QUALITY_GATE,
                    "min_recall_abnormal": MIN_RECALL,
                    "min_f1_abnormal": MIN_F1,
                },
                "is_deployable": True,
                "quality_reason": None,
                "threshold": None,
                "threshold_source": None,
            }
            if best_eval:
                payload = best_eval["payload"]
                metrics = payload.get("per_machine", {}).get(machine, {})
                if not isinstance(metrics, dict):
                    metrics = {}
                model_entry["best_eval_file"] = best_eval["eval_file"]
                model_entry["best_method"] = _method_from_eval_file(best_eval["eval_file"])
                model_entry["best_eval_metrics"] = {
                    "auc_roc": float(metrics.get("auc_roc", 0.0)),
                    "pauc_fpr_le_0.1": float(metrics.get("pauc_fpr_le_0.1", 0.0)),
                    "precision_abnormal": float(metrics.get("precision_abnormal", 0.0)),
                    "recall_abnormal": float(metrics.get("recall_abnormal", 0.0)),
                    "f1_abnormal": float(metrics.get("f1_abnormal", 0.0)),
                }
                if ENABLE_QUALITY_GATE:
                    rec = float(metrics.get("recall_abnormal", 0.0))
                    f1 = float(metrics.get("f1_abnormal", 0.0))
                    if rec < MIN_RECALL or f1 < MIN_F1:
                        model_entry["is_deployable"] = False
                        model_entry["quality_reason"] = (
                            f"Low reliability for deployment: recall={rec:.3f}, f1={f1:.3f}"
                        )
                if "threshold" in metrics:
                    model_entry["threshold"] = float(metrics["threshold"])
                    model_entry["threshold_source"] = best_eval["eval_file"]

            # Use reconstruction-optimized thresholds only when reconstruction is the active method.
            # For LOF/GMM/Mahalanobis methods, keep thresholds from their matching evaluation file.
            if model_entry.get("best_method") == "reconstruction":
                opt_threshold = _pick_optimized_threshold(ckpt.parent, machine)
                if opt_threshold is not None:
                    model_entry["threshold"] = opt_threshold[0]
                    model_entry["threshold_source"] = opt_threshold[1]

            models.append(model_entry)
            datasets.setdefault(machine, {}).setdefault(architecture, []).append(model_entry)

    beats_meta = _load_beats_export_meta()
    beats_eval = _load_beats_eval_meta()
    beats_eval_per_machine = beats_eval.get("per_machine", {}) if isinstance(beats_eval, dict) else {}
    if beats_meta:
        machine_info = beats_meta.get("machines", {})
        if isinstance(machine_info, dict):
            for machine, vals in machine_info.items():
                if not isinstance(vals, dict):
                    continue
                gmm_rel = str(vals.get("gmm_path", f"gmms/{machine}.pkl"))
                gmm_path = BEATS_EXPORT_ROOT / Path(gmm_rel)
                if not gmm_path.is_file():
                    continue
                threshold = float(vals.get("threshold_99", vals.get("threshold_95", np.nan)))
                eval_metrics = beats_eval_per_machine.get(machine, {}) if isinstance(beats_eval_per_machine, dict) else {}
                if not isinstance(eval_metrics, dict):
                    eval_metrics = {}
                model_entry = {
                    "dataset": "dcase2024_development",
                    "architecture": "beats_frozen",
                    "machine_type": str(machine),
                    "run_name": "beats_frozen_export",
                    "checkpoint": "beats_frozen://export",
                    "best_eval_file": "beats_frozen_results.json" if eval_metrics else "embeddings_info.json",
                    "best_method": "gmm",
                    "best_eval_metrics": {
                        "auc_roc": float(eval_metrics.get("auc_roc", 0.0)),
                        "pauc_fpr_le_0.1": float(eval_metrics.get("pauc_fpr_le_0.1", 0.0)),
                        "precision_abnormal": 0.0,
                        "recall_abnormal": 0.0,
                        "f1_abnormal": 0.0,
                    },
                    "quality_gate": {
                        "enabled": False,
                        "min_recall_abnormal": MIN_RECALL,
                        "min_f1_abnormal": MIN_F1,
                    },
                    "is_deployable": True,
                    "quality_reason": None,
                    "threshold": None if not np.isfinite(threshold) else threshold,
                    "threshold_source": "embeddings_info.json:threshold_99",
                    "beats_gmm_path": _rel(gmm_path),
                }
                models.append(model_entry)
                datasets.setdefault(str(machine), {}).setdefault("beats_frozen", []).append(model_entry)

    for machine, arch_map in datasets.items():
        for architecture, entries in arch_map.items():
            entries.sort(
                key=lambda item: (
                    float((item.get("best_eval_metrics") or {}).get("auc_roc", 0.0)),
                    float((item.get("best_eval_metrics") or {}).get("pauc_fpr_le_0.1", 0.0)),
                    float((item.get("best_eval_metrics") or {}).get("recall_abnormal", 0.0)),
                    float((item.get("best_eval_metrics") or {}).get("f1_abnormal", 0.0)),
                ),
                reverse=True,
            )
            deployable = [item for item in entries if item.get("is_deployable", True)]
            pick_pool = deployable if deployable else entries
            for item in entries:
                item["recommended_for_combo"] = item is pick_pool[0]
            if len(pick_pool) > 1:
                pick_pool[0]["dataset_reason"] = "highest_auc_then_pauc_then_recall_then_f1"
    return {"models": models, "datasets": datasets}


@lru_cache(maxsize=16)
def _bundle_for_checkpoint(rel_checkpoint: str) -> tuple[Any, Any, AudioConfig, FeatureConfig, WindowConfig, float, float]:
    ckpt_path = ROOT / rel_checkpoint
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_bundle(ckpt_path, device)
    audio_cfg = AudioConfig(**ckpt["audio_config"])
    feature_cfg = FeatureConfig(**ckpt["feature_config"])
    window_cfg = WindowConfig(**ckpt["window_config"])
    mean = float(ckpt["norm"]["mean"])
    std = float(ckpt["norm"]["std"])
    return model, device, audio_cfg, feature_cfg, window_cfg, mean, std


def _resolve_model(machine_type: str, architecture: str, dataset: str | None) -> dict[str, Any]:
    cat = _catalog()
    machine_map = cat["datasets"].get(machine_type)
    if not machine_map:
        raise HTTPException(status_code=404, detail=f"No models found for machine_type='{machine_type}'.")
    arch_items = machine_map.get(architecture)
    if not arch_items:
        raise HTTPException(status_code=404, detail=f"No models found for architecture='{architecture}'.")
    if dataset:
        for item in arch_items:
            if item["dataset"] == dataset:
                if item.get("is_deployable") is False:
                    raise HTTPException(
                        status_code=409,
                        detail=item.get("quality_reason")
                        or "Selected model is below quality gate thresholds.",
                    )
                return item
        raise HTTPException(
            status_code=404,
            detail=f"No model found for machine_type='{machine_type}', architecture='{architecture}', dataset='{dataset}'.",
        )
    for item in arch_items:
        if item.get("is_deployable", True):
            return item
    raise HTTPException(
        status_code=409,
        detail=(
            f"No deployable model for machine_type='{machine_type}', architecture='{architecture}'. "
            "Try another architecture or relax quality gate thresholds."
        ),
    )


@lru_cache(maxsize=64)
def _load_scorer(rel_run_dir: str, method: str) -> Any:
    run_dir = ROOT / rel_run_dir
    if method in ("reconstruction", "unknown"):
        return None
    if method == "gmm":
        p = run_dir / "scorer_gmm.joblib"
        if not p.is_file():
            raise FileNotFoundError(f"Missing scorer artifact: {p}")
        return joblib.load(p)
    if method == "tta_gmm":
        p = run_dir / "scorer_tta_gmm.joblib"
        if not p.is_file():
            raise FileNotFoundError(f"Missing scorer artifact: {p}")
        return joblib.load(p)
    if method == "domain_gmm":
        p = run_dir / "scorer_domain_gmm.joblib"
        if not p.is_file():
            raise FileNotFoundError(f"Missing scorer artifact: {p}")
        return joblib.load(p)
    if method == "lof":
        p = run_dir / "scorer_lof.joblib"
        if not p.is_file():
            raise FileNotFoundError(f"Missing scorer artifact: {p}")
        return joblib.load(p)
    if method == "mahalanobis":
        p = run_dir / "scorer_mahalanobis.npz"
        if not p.is_file():
            raise FileNotFoundError(f"Missing scorer artifact: {p}")
        arr = np.load(p)
        return {"mu": arr["mu"], "inv_cov": arr["inv_cov"]}
    raise FileNotFoundError(f"Unsupported scorer method '{method}'")


@lru_cache(maxsize=1)
def _beats_runtime() -> dict[str, Any]:
    try:
        from BEATs import BEATs, BEATsConfig
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing BEATs dependency: torchaudio. Install it with `pip install torchaudio`."
        ) from e

    ckpt = _resolve_beats_ckpt()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(ckpt, map_location="cpu")
    beats = BEATs(BEATsConfig(raw["cfg"]))
    beats.load_state_dict(raw["model"])
    beats.to(device).eval()
    audio_cfg = AudioConfig()
    target_len = int(audio_cfg.fixed_duration_sec * audio_cfg.sample_rate)
    return {"model": beats, "device": device, "audio_cfg": audio_cfg, "target_len": target_len, "ckpt": ckpt}


@torch.no_grad()
def _score_beats_frozen(audio_path: Path, machine_type: str, gmm_rel_path: str) -> float:
    runtime = _beats_runtime()
    gmm_path = ROOT / gmm_rel_path
    if not gmm_path.is_file():
        raise FileNotFoundError(f"Missing BEATs scorer artifact: {gmm_path}")
    gmm = joblib.load(gmm_path)
    wav = load_audio(audio_path, runtime["audio_cfg"]).astype(np.float32)
    target_len = int(runtime["target_len"])
    if len(wav) < target_len:
        wav = np.tile(wav, int(np.ceil(target_len / max(1, len(wav)))))
    wav = wav[:target_len]
    wav_t = torch.from_numpy(wav).unsqueeze(0).to(runtime["device"])
    pad = torch.zeros(1, wav_t.size(1), dtype=torch.bool, device=runtime["device"])
    feats, _ = runtime["model"].extract_features(wav_t, padding_mask=pad)
    emb = feats.mean(dim=1).cpu().numpy()
    return float(-gmm.score_samples(emb)[0])


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Acoustic Anomaly Detection Studio</title>
  <style>
    :root {
      --bg: #070b14;
      --panel: #101827;
      --panel-soft: #0d1422;
      --text: #ecf2fb;
      --muted: #9fb2d2;
      --line: #22324d;
      --brand: #4cc9f0;
      --brand-2: #46e3b7;
      --ok: #22c55e;
      --alert: #ef4444;
      --warn: #f59e0b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Inter, Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(1300px 700px at 15% -10%, #17233b 0%, transparent 55%),
        radial-gradient(900px 500px at 100% 0%, #112a36 0%, transparent 50%),
        var(--bg);
      min-height: 100vh;
      font-size: 22px;
      zoom: 1.25;
    }
    .container {
      width: 100%;
      max-width: none;
      margin: 0;
      padding: 12px 14px 24px;
    }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: linear-gradient(180deg, #141f33, #0f1828);
      padding: 14px 16px;
      position: sticky;
      top: 12px;
      z-index: 20;
      backdrop-filter: blur(5px);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 11px;
      min-width: 0;
    }
    .brand-mark {
      width: 38px;
      height: 38px;
      border-radius: 10px;
      background: conic-gradient(from 210deg, #47d9ff, #5ce5bf, #47d9ff);
      box-shadow: 0 0 26px rgba(76, 201, 240, 0.25);
      flex: 0 0 auto;
    }
    .brand h1 { margin: 0; font-size: 1.3rem; }
    .brand p {
      margin: 2px 0 0;
      color: var(--muted);
      font-size: 0.87rem;
    }
    .nav { display: flex; gap: 8px; flex-wrap: wrap; }
    .nav button {
      border: 1px solid #304462;
      background: #0b1422;
      color: #d9e8ff;
      border-radius: 999px;
      padding: 8px 14px;
      font-weight: 600;
      cursor: pointer;
    }
    .nav button.active {
      border-color: #3acff1;
      color: #072638;
      background: linear-gradient(90deg, var(--brand), var(--brand-2));
    }
    .page { display: none; margin-top: 16px; }
    .page.active { display: block; }
    .hero {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 22px;
      background:
        linear-gradient(rgba(7, 12, 22, 0.45), rgba(7, 12, 22, 0.70)),
        url('https://images.unsplash.com/photo-1581094794329-c8112a89af12?auto=format&fit=crop&w=1800&q=70') center/cover no-repeat;
      min-height: 380px;
      display: flex;
      flex-direction: column;
      justify-content: end;
    }
    .hero h2 { margin: 0; font-size: 2.35rem; }
    .hero p { margin: 8px 0 0; color: #d5e5fb; max-width: 920px; line-height: 1.5; font-size: 1.05rem; }
    .hero-actions { margin-top: 14px; display: flex; gap: 10px; flex-wrap: wrap; }
    .cta {
      border-radius: 10px;
      border: 0;
      padding: 11px 14px;
      font-weight: 700;
      cursor: pointer;
    }
    .cta.primary {
      color: #062032;
      background: linear-gradient(90deg, var(--brand), var(--brand-2));
    }
    .cta.secondary {
      color: #dfebff;
      border: 1px solid #355175;
      background: #101c2f;
    }
    .section-title { margin: 0 0 12px; font-size: 1.45rem; }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 14px;
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: linear-gradient(180deg, var(--panel), var(--panel-soft));
      padding: 16px;
    }
    .machine-card {
      min-height: 240px;
      position: relative;
      overflow: hidden;
      padding: 0;
    }
    .machine-card img {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
      transform: scale(1.02);
      transition: transform .35s ease;
    }
    .machine-card:hover img { transform: scale(1.10); }
    .machine-overlay {
      position: absolute;
      inset: 0;
      background: linear-gradient(180deg, rgba(10,16,28,0.25), rgba(10,16,28,0.92));
      display: flex;
      flex-direction: column;
      justify-content: end;
      padding: 14px;
    }
    .machine-overlay h4 { margin: 0; font-size: 1.03rem; }
    .machine-overlay p { margin: 5px 0 0; color: #b9c9e1; font-size: 0.9rem; }
    .inference-grid {
      display: grid;
      grid-template-columns: 1.25fr 1fr;
      gap: 14px;
    }
    label { display: block; margin: 10px 0 6px; font-weight: 600; color: #d8e6fb; font-size: 1rem; }
    select, input[type=file], input[type=number] {
      width: 100%;
      padding: 11px 12px;
      margin-bottom: 8px;
      border-radius: 10px;
      border: 1px solid #2a3d5d;
      background: #0b1321;
      color: var(--text);
    }
    .btn {
      width: 100%; padding: 11px; border: none; border-radius: 10px;
      cursor: pointer; font-weight: 700; color: #072638;
      background: linear-gradient(90deg, var(--brand), var(--brand-2));
      margin-top: 6px;
    }
    .btn:disabled { opacity: 0.65; cursor: not-allowed; }
    .result {
      margin-top: 10px;
      border: 1px solid #2a3f61;
      border-radius: 10px;
      padding: 12px;
      background: #0a1220;
    }
    .result-line { margin: 4px 0; color: #cfddf3; font-size: 1.03rem; }
    .state {
      display: inline-block; font-size: 0.82rem; font-weight: 700; border-radius: 999px; padding: 5px 10px;
      border: 1px solid #314563; background: #132138; color: #d5e4fa;
    }
    .state.normal { color: #dcfce7; border-color: #1f6f45; background: rgba(34,197,94,0.18); }
    .state.abnormal { color: #fee2e2; border-color: #812626; background: rgba(239,68,68,0.2); }
    .state.uncertain { color: #fef3c7; border-color: #7a5419; background: rgba(245,158,11,0.18); }
    pre {
      margin: 8px 0 0; max-height: 300px; overflow: auto;
      background: #08101d; color: #d6e4fb; border: 1px solid #243551; border-radius: 10px; padding: 10px;
      font-size: 0.82rem; line-height: 1.4;
    }
    .meta { color: var(--muted); font-size: 0.96rem; margin-top: 6px; }
    .about-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 12px;
    }
    @media (max-width: 960px) {
      .inference-grid { grid-template-columns: 1fr; }
      .hero h2 { font-size: 1.6rem; }
      .topbar { position: static; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="topbar">
      <div class="brand">
        <div class="brand-mark"></div>
        <div>
          <h1>Acoustic AAD Studio</h1>
          <p>Industrial machine health intelligence</p>
        </div>
      </div>
      <div class="nav" id="navTabs">
        <button data-page="home" class="active">Home</button>
        <button data-page="machines">Machines</button>
        <button data-page="inference">Inference</button>
        <button data-page="about">About</button>
      </div>
    </div>

    <section class="page active" id="page-home">
      <div class="hero">
        <h2>Pro-grade Acoustic Anomaly Monitoring</h2>
        <p>Browse machine families, inspect available model combinations, and run live anomaly inference in one visual workspace designed for demos and operations teams.</p>
        <div class="hero-actions">
          <button class="cta primary" id="goInference">Open Inference</button>
          <button class="cta secondary" id="goMachines">Explore Machines</button>
        </div>
      </div>
    </section>

    <section class="page" id="page-machines">
      <h3 class="section-title">Machine Gallery</h3>
      <div class="cards" id="machineCards"></div>
    </section>

    <section class="page" id="page-inference">
      <h3 class="section-title">Inference Workbench</h3>
      <div class="inference-grid">
        <div class="card">
          <label>Machine Type</label>
          <select id="machine"></select>
          <label>Architecture</label>
          <select id="arch"></select>
          <label>Dataset (Auto picks recommended)</label>
          <select id="dataset"></select>
          <label>Audio file (.wav / .flac)</label>
          <input id="audio" type="file" accept=".wav,.flac" />
          <button class="btn" id="run">Run Inference</button>
          <div class="meta" id="selectionMeta">Load a file and run scoring.</div>
        </div>
        <div class="card">
          <div id="resultState" class="state uncertain">Waiting</div>
          <div class="result">
            <div class="result-line"><strong>Score:</strong> <span id="scoreValue">-</span></div>
            <div class="result-line"><strong>Threshold:</strong> <span id="thresholdValue">-</span></div>
            <div class="result-line"><strong>Decision:</strong> <span id="decisionValue">-</span></div>
            <div class="result-line"><strong>Method:</strong> <span id="methodValue">-</span></div>
            <div class="result-line"><strong>Threshold Source:</strong> <span id="thresholdSourceValue">-</span></div>
          </div>
        </div>
      </div>
    </section>

    <section class="page" id="page-about">
      <h3 class="section-title">Project Context</h3>
      <div class="about-grid">
        <div class="card">
          <h4 style="margin:0 0 6px;">Model Selection</h4>
          <p class="meta">Machine + architecture picks the best dataset run using your catalog ranking logic.</p>
        </div>
        <div class="card">
          <h4 style="margin:0 0 6px;">Method-aware Scoring</h4>
          <p class="meta">Inference uses the selected method (reconstruction, GMM, LOF, Mahalanobis, etc.) with matching thresholds.</p>
        </div>
        <div class="card">
          <h4 style="margin:0 0 6px;">Demo Reliability</h4>
          <p class="meta">Designed for quick demos with readable outcomes and direct insight into score vs threshold behavior.</p>
        </div>
      </div>
    </section>
  </div>
  <script>
    const machineImageByKeyword = {
      toycar: '/static/machines/toycar.jpg',
      toytrain: '/static/machines/toytrain.jpg',
      valve: '/static/machines/valve.jpg',
      fan: '/static/machines/fan.jpg',
      pump: '/static/machines/pump.jpg',
      slider: '/static/machines/slider.jpg',
      gearbox: '/static/machines/gearbox.jpg',
      bearing: '/static/machines/bearing.jpg'
    };
    const machineDefaultImage = '/static/machines/default.jpg';
    let catalog = {};

    function asNum(v) {
      if (v === null || v === undefined || Number.isNaN(Number(v))) return '-';
      return Number(v).toFixed(6);
    }
    function setState(label, cls) {
      const el = document.getElementById('resultState');
      el.textContent = label;
      el.className = `state ${cls || 'uncertain'}`;
    }
    function showPage(pageName) {
      document.querySelectorAll('.page').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('#navTabs button').forEach(el => el.classList.remove('active'));
      const page = document.getElementById(`page-${pageName}`);
      const tab = document.querySelector(`#navTabs button[data-page="${pageName}"]`);
      if (page) page.classList.add('active');
      if (tab) tab.classList.add('active');
    }
    function imageForMachine(machine) {
      const key = machine.toLowerCase();
      for (const k of Object.keys(machineImageByKeyword)) {
        if (key.includes(k)) return machineImageByKeyword[k];
      }
      return machineDefaultImage;
    }
    function renderMachineCards() {
      const holder = document.getElementById('machineCards');
      const machines = Object.keys(catalog.datasets || {}).sort();
      if (!machines.length) {
        holder.innerHTML = '<div class="card"><p class="meta">No machine models detected in artifacts catalog.</p></div>';
        return;
      }
      holder.innerHTML = machines.map(m => {
        const archMap = (catalog.datasets || {})[m] || {};
        const archs = Object.keys(archMap);
        const runs = archs.reduce((acc, a) => acc + (archMap[a] || []).length, 0);
        return `
          <div class="card machine-card" data-machine="${m}">
            <img src="${imageForMachine(m)}" alt="${m}" />
            <div class="machine-overlay">
              <h4>${m}</h4>
              <p>${archs.length} architecture(s) | ${runs} run(s)</p>
            </div>
          </div>
        `;
      }).join('');
      holder.querySelectorAll('.machine-card').forEach(card => {
        card.addEventListener('click', () => {
          const machine = card.dataset.machine;
          showPage('inference');
          document.getElementById('machine').value = machine;
          refreshArch();
        });
      });
    }
    async function loadCatalog() {
      const res = await fetch('/models/catalog');
      catalog = await res.json();
      const machines = Object.keys(catalog.datasets || {}).sort();
      document.getElementById('machine').innerHTML = machines.map(m => `<option value="${m}">${m}</option>`).join('');
      renderMachineCards();
      refreshArch();
    }
    function refreshArch() {
      const machine = document.getElementById('machine').value;
      const archMap = (catalog.datasets || {})[machine] || {};
      const arches = Object.keys(archMap).sort();
      document.getElementById('arch').innerHTML = arches.map(a => `<option value="${a}">${a}</option>`).join('');
      refreshDataset();
    }
    function refreshDataset() {
      const machine = document.getElementById('machine').value;
      const arch = document.getElementById('arch').value;
      const entries = (((catalog.datasets || {})[machine] || {})[arch] || []);
      const opts = ['<option value="">Auto (recommended)</option>']
        .concat(entries.map(e => `<option value="${e.dataset}">${e.dataset}${e.recommended_for_combo ? ' (recommended)' : ''}</option>`));
      document.getElementById('dataset').innerHTML = opts.join('');
      const rec = entries.find(e => e.recommended_for_combo) || entries[0];
      if (rec) {
        const met = rec.best_eval_metrics || {};
        document.getElementById('selectionMeta').textContent =
          `Recommended: ${rec.run_name} (${rec.best_method}) | AUC ${asNum(met.auc_roc)}`;
      } else {
        document.getElementById('selectionMeta').textContent = 'No models for this selection.';
      }
    }
    async function runScore() {
      const fileInput = document.getElementById('audio');
      if (!fileInput.files.length) return;
      const btn = document.getElementById('run');
      btn.disabled = true;
      btn.textContent = 'Running...';
      setState('Running', 'uncertain');
      const fd = new FormData();
      fd.append('file', fileInput.files[0]);
      fd.append('machine_type', document.getElementById('machine').value);
      fd.append('architecture', document.getElementById('arch').value);
      const ds = document.getElementById('dataset').value;
      if (ds) fd.append('dataset', ds);
      try {
        const res = await fetch('/score', { method: 'POST', body: fd });
        const txt = await res.text();
        let out = null;
        try { out = JSON.parse(txt); } catch (_) { out = null; }
        if (!res.ok || !out) {
          setState('Error', 'abnormal');
          document.getElementById('scoreValue').textContent = '-';
          document.getElementById('thresholdValue').textContent = '-';
          document.getElementById('decisionValue').textContent = '-';
          document.getElementById('methodValue').textContent = '-';
          document.getElementById('thresholdSourceValue').textContent = '-';
          const detail = out && out.detail ? String(out.detail) : `HTTP ${res.status}`;
          document.getElementById('selectionMeta').textContent = `Inference failed: ${detail}`;
          return;
        }
        const decision = out.decision || 'uncertain';
        setState(decision.toUpperCase(), decision === 'abnormal' ? 'abnormal' : (decision === 'normal' ? 'normal' : 'uncertain'));
        document.getElementById('scoreValue').textContent = asNum(out.anomaly_score);
        document.getElementById('thresholdValue').textContent = asNum(out.threshold);
        document.getElementById('decisionValue').textContent = decision;
        document.getElementById('methodValue').textContent = (out.selected_model || {}).best_method || '-';
        document.getElementById('thresholdSourceValue').textContent = out.threshold_source || '-';
      } finally {
        btn.disabled = false;
        btn.textContent = 'Run Inference';
      }
    }

    document.querySelectorAll('#navTabs button').forEach(btn => {
      btn.addEventListener('click', () => showPage(btn.dataset.page));
    });
    document.getElementById('goInference').addEventListener('click', () => showPage('inference'));
    document.getElementById('goMachines').addEventListener('click', () => showPage('machines'));
    document.getElementById('machine').addEventListener('change', refreshArch);
    document.getElementById('arch').addEventListener('change', refreshDataset);
    document.getElementById('run').addEventListener('click', runScore);
    loadCatalog();
  </script>
</body>
</html>
"""


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models/catalog")
def models_catalog() -> dict[str, Any]:
    return _catalog()


@app.post("/score")
async def score(
    file: UploadFile = File(...),
    machine_type: str = Form(...),
    architecture: str = Form(...),
    dataset: str | None = Form(None),
    threshold_override: float | None = Form(None),
) -> dict[str, Any]:
    model_item = _resolve_model(machine_type=machine_type, architecture=architecture, dataset=dataset)

    suffix = (file.filename or "").lower()
    if not suffix.endswith((".wav", ".flac")):
        raise HTTPException(status_code=400, detail="Upload .wav or .flac file.")

    raw = await file.read()
    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    tmp = Path(tmp_name)
    method = str(model_item.get("best_method") or "reconstruction")
    try:
        tmp.write_bytes(raw)
        if architecture == "beats_frozen":
            gmm_rel = model_item.get("beats_gmm_path")
            if not gmm_rel:
                raise HTTPException(status_code=409, detail="Missing BEATs GMM path in model catalog.")
            try:
                score_value = _score_beats_frozen(tmp, machine_type=machine_type, gmm_rel_path=gmm_rel)
            except ModuleNotFoundError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
        else:
            try:
                model, device, audio_cfg, feature_cfg, window_cfg, mean, std = _bundle_for_checkpoint(model_item["checkpoint"])
            except FileNotFoundError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
            run_dir_rel = str(Path(model_item["checkpoint"]).parent).replace("\\", "/")
            scorer_obj = None
            if method != "reconstruction":
                try:
                    scorer_obj = _load_scorer(run_dir_rel, method)
                except FileNotFoundError as e:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"Best method '{method}' selected by {model_item.get('best_eval_file')}, "
                            f"but scorer artifact is missing. {e}"
                        ),
                    ) from e
            rec = FileRecord(
                audio_path=tmp,
                machine_type=machine_type,
                section="unknown",
                domain="unknown",
                split="inference",
                label="unknown",
                dataset_name=model_item["dataset"],
            )
            shared = dict(
                audio_cfg=audio_cfg,
                feature_cfg=feature_cfg,
                window_cfg=window_cfg,
                mean=mean,
                std=std,
                device=device,
            )
            if method in ("gmm", "tta_gmm"):
                score_value = gmm_score_file(model, rec, **shared, gmm=scorer_obj)
            elif method == "domain_gmm":
                s_src = gmm_score_file(model, rec, **shared, gmm=scorer_obj["source"])
                s_tgt = gmm_score_file(model, rec, **shared, gmm=scorer_obj["target"])
                if np.isfinite(s_src) and np.isfinite(s_tgt):
                    score_value = min(float(s_src), float(s_tgt))
                else:
                    score_value = float(s_src) if np.isfinite(s_src) else float(s_tgt)
            elif method == "mahalanobis":
                score_value = mahalanobis_score_file(
                    model, rec, **shared, mu=scorer_obj["mu"], inv_cov=scorer_obj["inv_cov"]
                )
            elif method == "lof":
                z = model.encode
                # Reuse gmm_score_file-style preprocessing for LOF by collecting one file's latents inline.
                from aad.preprocess import load_audio, waveform_to_log_mel, window_spectrogram, zscore

                wav = load_audio(rec.audio_path, audio_cfg)
                mel = waveform_to_log_mel(wav, feature_cfg, sample_rate=audio_cfg.sample_rate)
                mel = zscore(mel, mean=mean, std=std)
                windows = window_spectrogram(mel, window_cfg)
                if not windows:
                    score_value = float("nan")
                else:
                    scores: list[float] = []
                    for w in windows:
                        x = torch.from_numpy(w).unsqueeze(0).unsqueeze(0).to(device)
                        latent = z(x)
                        z_hat, _ = model.memory(latent)
                        vec = z_hat.detach().cpu().numpy()
                        scores.append(float(-scorer_obj.score_samples(vec)[0]))
                    score_value = float(np.max(scores))
            else:
                score_value = score_file(model, rec, **shared)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Scoring failed: {e}") from e
    finally:
        try:
            tmp.unlink(missing_ok=True)
        finally:
            pass

    threshold = threshold_override if threshold_override is not None else model_item.get("threshold")
    decision = None
    if threshold is not None:
        decision = "abnormal" if float(score_value) > float(threshold) else "normal"
    return {
        "filename": file.filename,
        "anomaly_score": float(score_value),
        "decision": decision,
        "threshold": threshold,
        "threshold_source": (
            "override" if threshold_override is not None else model_item.get("threshold_source")
        ),
        "selected_model": {
            "machine_type": machine_type,
            "architecture": architecture,
            "dataset": model_item["dataset"],
            "run_name": model_item["run_name"],
            "checkpoint": model_item["checkpoint"],
            "best_eval_file": model_item["best_eval_file"],
            "best_method": model_item.get("best_method"),
            "best_eval_metrics": model_item["best_eval_metrics"],
        },
    }
