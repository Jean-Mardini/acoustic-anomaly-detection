from pathlib import Path

import pytest

from backend.ml.manifest import build_manifest, parse_wav_path, write_manifest_csv


def test_parse_labeled_train_clip(tmp_path: Path) -> None:
    wav = (
        tmp_path
        / "eval_data"
        / "raw"
        / "fan"
        / "train"
        / "section_00_source_train_normal_0001_n_0.wav"
    )
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"")

    rec = parse_wav_path(wav, "dcase2024_additional")
    assert rec.machine_type == "fan"
    assert rec.section == "section_00"
    assert rec.domain == "source"
    assert rec.split == "train"
    assert rec.label == "normal"


def test_parse_eval_clip(tmp_path: Path) -> None:
    wav = tmp_path / "eval_data" / "raw" / "3DPrinter" / "test" / "section_00_0001.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"")

    rec = parse_wav_path(wav, "dcase2024_evaluation")
    assert rec.machine_type == "3DPrinter"
    assert rec.section == "section_00"
    assert rec.domain == "unknown"
    assert rec.split == "test"
    assert rec.label == "unknown"


def test_anomaly_label_maps_to_abnormal(tmp_path: Path) -> None:
    wav = (
        tmp_path
        / "fan"
        / "test"
        / "section_00_target_test_anomaly_0042_x.wav"
    )
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"")

    rec = parse_wav_path(wav, "dcase2024_development")
    assert rec.label == "abnormal"


def test_write_manifest_csv_roundtrip(tmp_path: Path) -> None:
    wav = tmp_path / "ToyCar" / "train" / "section_00_target_train_normal_0003.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"")

    records = build_manifest(tmp_path, "dcase2024_development", validate_files=True)
    assert len(records) == 1

    out = tmp_path / "manifest.csv"
    write_manifest_csv(records, out)
    text = out.read_text(encoding="utf-8")
    assert "ToyCar" in text
    assert "section_00" in text


def test_build_manifest_skips_unrecognized_names(tmp_path: Path) -> None:
    bad = tmp_path / "fan" / "train" / "random_name.wav"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"")

    records = build_manifest(tmp_path, "x", validate_files=True)
    assert records == []
