"""Audit the supplied label CSV and define the shared data entry points."""

import csv
import json
from collections import Counter
from pathlib import Path

from .config import LABEL_MAPPING, PROFILE_COLUMNS, RAW_LABELS_PATH


def audit_labels(path: str | Path = RAW_LABELS_PATH) -> dict:
    """Inspect the ID/label file without requiring third-party libraries."""
    path = Path(path)
    seen_ids: set[str] = set()
    counts: Counter[str] = Counter()
    rows = missing_ids = missing_labels = duplicate_ids = 0

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing_columns = {"id", "account_type"} - set(columns)
        if missing_columns:
            raise ValueError(f"Missing label columns: {sorted(missing_columns)}")

        for row in reader:
            rows += 1
            identifier = (row["id"] or "").strip()
            label = (row["account_type"] or "").strip()
            missing_ids += not identifier
            missing_labels += not label
            if identifier:
                duplicate_ids += identifier in seen_ids
                seen_ids.add(identifier)
            counts[label] += 1

    return {
        "file": path.name,
        "rows": rows,
        "columns": columns,
        "label_counts": dict(counts),
        "missing_ids": missing_ids,
        "missing_labels": missing_labels,
        "duplicate_ids": duplicate_ids,
        "unknown_labels": sorted(set(counts) - set(LABEL_MAPPING) - {""}),
        "missing_profile_columns": sorted(set(PROFILE_COLUMNS) - set(columns)),
    }


def load_profiles(path: str | Path):
    """Read a normalized, enriched profile CSV; IDs remain strings.

    This is a separate input from the supplied ID/label file. The caller must
    resolve metadata collection, column mapping and account identity first.
    """
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        columns = next(csv.reader(handle), [])
    missing = set(PROFILE_COLUMNS) - set(columns)
    if missing:
        raise ValueError(f"Profile data is incomplete; missing columns: {sorted(missing)}")

    import pandas as pd

    return pd.read_csv(path, dtype={"user_id": "string"})


def split_features(features):
    """TODO: return train/validation/test DataFrames and persist their user IDs.

    Follow the source's official split when available; otherwise use the shared
    fractions and seed in config.py. Keep each user in exactly one split.
    """
    raise NotImplementedError("Chưa triển khai chia tập dùng chung.")


if __name__ == "__main__":
    print(json.dumps(audit_labels(), ensure_ascii=False, indent=2))
