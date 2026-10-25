"""Read and validate source profiles without changing or dropping source rows."""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    COLUMN_MAPPING, COUNT_COLUMNS, LABEL_MAPPING, NUMERIC_COLUMNS,
    NUMERIC_MISSING_MARKERS, RAW_LABELS_PATH, RAW_PROFILE_COLUMNS,
)


def audit_labels(path: str | Path = RAW_LABELS_PATH) -> dict:
    """Inspect source IDs, labels and the required raw profile column names."""
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
        "missing_profile_columns": sorted(set(RAW_PROFILE_COLUMNS) - set(columns)),
    }


def load_profiles(path: str | Path = RAW_LABELS_PATH) -> tuple[pd.DataFrame, dict]:
    """Return normalized profiles and quality counts; preserve literal bio text.

    Identity/schema errors stop processing. Invalid numeric cells become missing
    and are counted in the report; no rows are removed. Source age is required,
    so observed_at is neither required nor fabricated.
    """
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        columns = next(csv.reader(handle), [])
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate column names in source CSV.")
    missing = set(RAW_PROFILE_COLUMNS) - set(columns)
    if missing:
        raise ValueError(f"Profile data is incomplete; missing columns: {sorted(missing)}")

    # Default pandas NA tokens would turn the literal biography 'None' into NA.
    raw = pd.read_csv(path, dtype="string", keep_default_na=False, encoding="utf-8-sig")
    return normalize_profiles(raw)


def clean_numeric(series: pd.Series, *, integer: bool = False) -> tuple[pd.Series, dict]:
    """Convert a nonnegative numeric column, retaining missing/invalid as pd.NA."""
    text = series.astype("string").str.strip()
    missing = text.isna() | text.str.lower().isin(NUMERIC_MISSING_MARKERS)
    numeric = pd.to_numeric(text.mask(missing), errors="coerce").astype("Float64")
    finite = pd.Series(
        np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)), index=series.index
    )
    non_numeric = ~missing & numeric.isna()
    non_finite = numeric.notna() & ~finite
    negative = finite & numeric.lt(0).fillna(False)
    fractional = (
        finite & ~negative & numeric.mod(1).ne(0).fillna(False)
        if integer else pd.Series(False, index=series.index)
    )
    invalid = non_numeric | non_finite | negative | fractional
    cleaned = numeric.mask(missing | invalid)
    stats = {
        "missing": int(missing.sum()), "non_numeric": int(non_numeric.sum()),
        "non_finite": int(non_finite.sum()), "negative": int(negative.sum()),
        "fractional": int(fractional.sum()), "invalid_total": int(invalid.sum()),
        "output_missing": int(cleaned.isna().sum()),
        "valid_zero": int(cleaned.eq(0).sum()),
    }
    return cleaned, stats


def normalize_profiles(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate raw source columns and normalize a copy (never mutate the input)."""
    missing = set(RAW_PROFILE_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Profile data is incomplete; missing columns: {sorted(missing)}")
    if raw.empty:
        raise ValueError("Source CSV contains no profile rows.")
    profiles = raw.copy()
    dropped_index = []
    for column in profiles.columns:
        if column == "" or str(column).startswith("Unnamed:"):
            values = pd.to_numeric(profiles[column], errors="coerce")
            if not np.array_equal(values.to_numpy(), np.arange(len(profiles))):
                raise ValueError(f"Unverified source index column: {column!r}; refusing to drop it.")
            dropped_index.append(column)
    profiles = profiles.drop(columns=dropped_index).rename(columns=COLUMN_MAPPING)
    if profiles.columns.duplicated().any():
        raise ValueError("Column mapping creates duplicate normalized column names.")
    identifiers = profiles["user_id"].astype("string")
    labels = profiles["account_type"].astype("string")
    missing_ids = identifiers.isna() | identifiers.str.strip().eq("").fillna(False)
    missing_labels = labels.isna() | labels.str.strip().eq("").fillna(False)
    duplicate = identifiers.duplicated(keep=False) & ~missing_ids
    invalid_labels = ~labels.isin(LABEL_MAPPING)
    grouped = pd.DataFrame({"id": identifiers, "label": labels}).loc[~missing_ids]
    conflicts = grouped.groupby("id")["label"].nunique(dropna=False).gt(1)
    identity = {
        "missing_ids": int(missing_ids.sum()),
        "duplicate_id_rows": int(duplicate.sum()),
        "duplicate_ids": int(identifiers[duplicate].nunique()),
        "missing_labels": int(missing_labels.sum()),
        "invalid_labels": int(invalid_labels.sum()),
        "conflicting_ids": int(conflicts.sum()),
    }
    if any(identity.values()):
        raise ValueError("Invalid profile identity/labels: " + json.dumps(identity))
    profiles["user_id"] = identifiers
    profiles["label"] = labels.map(LABEL_MAPPING).astype("int64")
    numeric_quality = {}
    for column in NUMERIC_COLUMNS:
        profiles[column], numeric_quality[column] = clean_numeric(
            profiles[column], integer=column in COUNT_COLUMNS
        )
    bio = profiles["description"].astype("string")
    null_bio = bio.isna()
    empty_bio = bio.eq("").fillna(False)
    whitespace_bio = bio.notna() & ~empty_bio & bio.str.strip().eq("").fillna(False)
    profiles["description"] = bio.mask(empty_bio)
    created = profiles["created_at"].astype("string")
    parsed = pd.to_datetime(created, errors="coerce", utc=True)
    date_missing = created.isna() | created.str.strip().eq("").fillna(False)
    quality = {
        "rows_read": len(raw), "rows_retained": len(profiles),
        "rows_removed": 0, "removal_reasons": [],
        "source_columns": list(raw.columns),
        "dropped_source_index_columns": dropped_index, "identity": identity,
        "label_counts": {str(k): int(v) for k, v in labels.value_counts().items()},
        "numeric_columns": numeric_quality,
        "bio": {
            "null": int(null_bio.sum()), "empty_string": int(empty_bio.sum()),
            "null_or_empty": int((null_bio | empty_bio).sum()),
            "whitespace_only_retained": int(whitespace_bio.sum()),
            "literal_None_retained": int(bio.eq("None").sum()),
        },
        "created_at": {
            "missing": int(date_missing.sum()),
            "unparseable": int((parsed.isna() & ~date_missing).sum()),
            "used_to_recompute_age": False,
        },
    }
    return profiles, quality


def split_features(features):
    """TODO: return train/validation/test DataFrames and persist their user IDs.

    Follow the source's official split when available; otherwise use the shared
    fractions and seed in config.py. Keep each user in exactly one split.
    """
    raise NotImplementedError("Chưa triển khai chia tập dùng chung.")


if __name__ == "__main__":
    print(json.dumps(audit_labels(), ensure_ascii=False, indent=2))
