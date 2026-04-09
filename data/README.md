# Dataset Card (Step 1.4)

This document defines what each dataset folder means before training.
It helps avoid split leakage and domain mixing mistakes.

## Target Datasets

- DCASE 2024 Task 2
- DCASE 2024 additional training data
- MIMII DUE

## Recommended Local Layout

```text
data/
  raw/
    dcase2024/
      train/
      dev/
      test/
    dcase2024_additional/
      train/
    mimii_due/
      train/
      test/
  processed/
    manifests/
    features/
```

## Folder Meaning Contract

- `train/`: clips used for model fitting only.
- `dev/`: clips used for threshold/calibration only.
- `test/`: clips used only for final evaluation.
- `source/`: recordings from source domain conditions.
- `target/`: recordings from target domain conditions.
- `normal/`: expected healthy machine behavior.
- `abnormal` or `anomaly`: anomalous behavior clips (when provided).

## Required Metadata Fields (per clip)

- `dataset_name` (e.g., dcase2024, mimii_due)
- `machine_type` (fan, pump, valve, etc.)
- `section` (section_00, section_01, ...)
- `domain` (source or target)
- `split` (train, dev, test)
- `label` (normal, abnormal, or unknown)
- `audio_path` (absolute or project-relative path)

## Ingestion Safety Rules

- Never tune thresholds on `test/`.
- Never mix `test/` into training features.
- Keep domain and section fields in manifests.
- Track clip duration and sample rate for every clip.
- Accept only known audio formats (`.wav`, `.flac`) unless explicitly updated.

## When to place dataset files

Place raw dataset files under `data/raw/...` before Step 2 feature extraction.
You can start placing files now, but we will parse them in the next ingestion step.
