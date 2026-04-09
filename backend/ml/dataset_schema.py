from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetRecord:
    dataset_name: str
    machine_type: str
    section: str
    domain: str
    split: str
    label: str
    audio_path: Path


VALID_DOMAINS = {"source", "target", "unknown"}
VALID_SPLITS = {"train", "dev", "test"}
VALID_LABELS = {"normal", "abnormal", "unknown"}


def validate_record(record: DatasetRecord) -> DatasetRecord:
    """Validate a single dataset manifest record."""
    if record.domain not in VALID_DOMAINS:
        raise ValueError(f"Invalid domain: {record.domain}")
    if record.split not in VALID_SPLITS:
        raise ValueError(f"Invalid split: {record.split}")
    if record.label not in VALID_LABELS:
        raise ValueError(f"Invalid label: {record.label}")
    if not record.audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {record.audio_path}")
    return record
