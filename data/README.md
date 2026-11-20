# Data

Raw audio is not stored in the repo. Download the three datasets and unzip them so the folders look like this:

```text
data/raw/
  dcase2024_development/unpacked/<machine>/{train,test}/*.wav
  dcase2024_additional/unpacked/<machine>/train/*.wav
  mimii_due/unpacked/<machine>/{train,source_test,target_test}/*.wav
```

- DCASE 2024 Task 2 development set and additional training data: https://dcase.community/challenge2024/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring
- MIMII-DUE: https://zenodo.org/records/4740355

Then run `python scripts/build_manifests.py`. It reads the file names (for example `section_00_source_test_anomaly_0001_....wav`) and writes one CSV per dataset to `data/processed/manifests/` with these columns:

| Column | Meaning |
|---|---|
| `dataset_name` | `dcase2024_development`, `dcase2024_additional` or `mimii_due` |
| `machine_type` | fan, gearbox, valve, ToyCar, ... |
| `section` | `section_00`, `section_01`, ... |
| `domain` | `source` or `target` |
| `split` | `train` or `test` |
| `label` | `normal`, `anomaly` or `unknown` |
| `audio_path` | path to the clip |

The manifests in the repo were built on another machine, so their `audio_path` values are absolute paths from that machine. Rebuild them after downloading the data.

Rules we followed:

- Training uses normal clips from the `train` split only.
- Domain and section are kept for every clip so source and target clips are never mixed up between splits.
- Thresholds for the deployed BEATs scorers are computed from normal training clips (99th percentile), not from test clips.
