"""
Build dataset manifests (CSV) from extracted DCASE 2024 and MIMII DUE layouts.

Expects extracted .wav files under folders that include split names (train/test)
and machine-type folder names, as in official dataset READMEs.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from backend.ml.dataset_schema import DatasetRecord, validate_record
from backend.ml.ingestion import discover_audio_files

# Normalized (alphanumeric lower) -> canonical display name
_MACHINE_ALIASES: dict[str, str] = {
    # DCASE 2024 development
    "bearing": "bearing",
    "fan": "fan",
    "gearbox": "gearbox",
    "slider": "slider",
    "toycar": "ToyCar",
    "toytrain": "ToyTrain",
    "valve": "valve",
    # DCASE 2024 additional / evaluation
    "3dprinter": "3DPrinter",
    "aircompressor": "AirCompressor",
    "brushlessmotor": "BrushlessMotor",
    "hairdryer": "HairDryer",
    "hoveringdrone": "HoveringDrone",
    "roboticarm": "RoboticArm",
    "scanner": "Scanner",
    "toothbrush": "ToothBrush",
    "toycircuit": "ToyCircuit",
    # MIMII DUE
    "pump": "pump",
}

_RE_LABELED = re.compile(
    r"^section_(?P<section>\d+)_(?P<domain>source|target)_(?P<split>train|test)_"
    r"(?P<label>normal|abnormal|anomaly)_(?P<clip>\d+)(?:_.*)?\.wav$",
    re.IGNORECASE,
)

_RE_EVAL_SIMPLE = re.compile(
    r"^section_(?P<section>\d+)_(?P<idx>\d+)\.wav$",
    re.IGNORECASE,
)


def _normalize_label(raw: str) -> str:
    if raw.lower() == "anomaly":
        return "abnormal"
    return raw.lower()


def _infer_machine_type(path: Path) -> str | None:
    """Infer machine type from path: .../<machine>/<train|test>/file.wav."""
    parts = path.parts
    for i, part in enumerate(parts):
        if part.lower() in ("train", "test"):
            if i > 0:
                key = "".join(ch for ch in parts[i - 1].lower() if ch.isalnum())
                if key in _MACHINE_ALIASES:
                    return _MACHINE_ALIASES[key]
    for part in parts:
        key = "".join(ch for ch in part.lower() if ch.isalnum())
        if key in _MACHINE_ALIASES:
            return _MACHINE_ALIASES[key]
    return None


def parse_wav_path(path: Path, dataset_name: str) -> DatasetRecord:
    """Parse one .wav path into a DatasetRecord."""
    path = path.resolve()
    machine_type = _infer_machine_type(path)
    if machine_type is None:
        raise ValueError(f"Could not infer machine type from path: {path}")

    name = path.name
    if _RE_EVAL_SIMPLE.fullmatch(name):
        m = _RE_EVAL_SIMPLE.fullmatch(name)
        assert m is not None
        section = f"section_{m.group('section')}"
        return DatasetRecord(
            dataset_name=dataset_name,
            machine_type=machine_type,
            section=section,
            domain="unknown",
            split="test",
            label="unknown",
            audio_path=path,
        )

    m = _RE_LABELED.match(name)
    if not m:
        raise ValueError(f"Unrecognized filename pattern: {name}")

    section = f"section_{m.group('section')}"
    domain = m.group("domain").lower()
    split = m.group("split").lower()
    label = _normalize_label(m.group("label"))

    return DatasetRecord(
        dataset_name=dataset_name,
        machine_type=machine_type,
        section=section,
        domain=domain,
        split=split,
        label=label,
        audio_path=path,
    )


def build_manifest(
    root: str | Path,
    dataset_name: str,
    *,
    validate_files: bool = True,
) -> list[DatasetRecord]:
    """
    Discover audio under root and build manifest records.

    Skips files that fail parsing (caller can inspect logs later; strict mode optional).
    """
    root_path = Path(root)
    files = discover_audio_files(root_path)
    records: list[DatasetRecord] = []
    for wav in files:
        try:
            rec = parse_wav_path(wav, dataset_name)
            if validate_files:
                validate_record(rec)
            else:
                # Schema-only check without file existence
                if rec.domain not in {"source", "target", "unknown"}:
                    raise ValueError(f"Invalid domain: {rec.domain}")
                if rec.split not in {"train", "dev", "test"}:
                    raise ValueError(f"Invalid split: {rec.split}")
                if rec.label not in {"normal", "abnormal", "unknown"}:
                    raise ValueError(f"Invalid label: {rec.label}")
            records.append(rec)
        except (ValueError, FileNotFoundError):
            continue
    return sorted(records, key=lambda r: str(r.audio_path))


def write_manifest_csv(records: list[DatasetRecord], out_path: str | Path) -> None:
    """Write manifest to CSV (header row)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_name",
        "machine_type",
        "section",
        "domain",
        "split",
        "label",
        "audio_path",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "dataset_name": r.dataset_name,
                    "machine_type": r.machine_type,
                    "section": r.section,
                    "domain": r.domain,
                    "split": r.split,
                    "label": r.label,
                    "audio_path": r.audio_path.as_posix(),
                }
            )
