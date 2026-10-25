"""Extract four profile features: python -m src.features."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    FEATURE_COLUMNS, FEATURE_OUTPUT_COLUMNS, FEATURES_PATH, FEATURE_SUMMARY_PATH,
    NUMERIC_COLUMNS, POSTING_ROUNDING_DECIMALS, RAW_LABELS_PATH,
)
from .data import load_profiles


def build_features(profiles: pd.DataFrame) -> pd.DataFrame:
    """Extract from load_profiles output; invalid inputs must already be missing.

    Zero denominators use one; missing denominators remain missing. Negative
    ages/counts must have been masked by the loader, never clipped to one.
    """
    required = {"user_id", "label", "description", *NUMERIC_COLUMNS}
    missing = required - set(profiles.columns)
    if missing:
        raise ValueError(f"Missing normalized columns: {sorted(missing)}")
    for column in NUMERIC_COLUMNS:
        values = profiles[column].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any() or (values < 0).any():
            raise ValueError(f"Unclean numeric values in {column}; use load_profiles first.")
    features = pd.DataFrame(index=profiles.index)
    features["user_id"] = profiles["user_id"].astype("string")
    features["follower_following_ratio"] = (
        profiles["followers_count"] / profiles["following_count"].clip(lower=1)
    )
    features["posting_frequency"] = (
        profiles["statuses_count"] / profiles["account_age_days"].clip(lower=1)
    )
    features["account_age_days"] = profiles["account_age_days"]
    features["bio_length"] = profiles["description"].astype("string").fillna("").str.len().astype("int64")
    features["label"] = profiles["label"].astype("int64")
    # Arithmetic overflow also becomes missing rather than exporting infinity.
    features = features.loc[:, list(FEATURE_OUTPUT_COLUMNS)]
    for column in FEATURE_COLUMNS:
        values = features[column].to_numpy(dtype=float, na_value=np.nan)
        features[column] = features[column].mask(np.isinf(values))
    return features


def summarize_features(profiles: pd.DataFrame, features: pd.DataFrame, quality: dict) -> dict:
    """Add formulas, source comparison and explicit processing assumptions."""
    tolerance = 0.5 * 10 ** -POSTING_ROUNDING_DECIMALS
    comparable = features["posting_frequency"].notna() & profiles["average_tweets_per_day"].notna()
    differences = (
        features.loc[comparable, "posting_frequency"]
        - profiles.loc[comparable, "average_tweets_per_day"]
    ).abs()
    within = differences.le(tolerance + 1e-12)
    arithmetic_overflow = {}
    for name, numerator, denominator in (
        ("follower_following_ratio", "followers_count", "following_count"),
        ("posting_frequency", "statuses_count", "account_age_days"),
    ):
        valid_inputs = profiles[numerator].notna() & profiles[denominator].notna()
        arithmetic_overflow[name] = int((valid_inputs & features[name].isna()).sum())
    return {
        "data_quality": quality,
        "output": {
            "rows": len(features), "columns": list(features.columns),
            "model_feature_columns": list(FEATURE_COLUMNS),
            "missing_by_feature": {c: int(features[c].isna().sum()) for c in FEATURE_COLUMNS},
            "arithmetic_overflow_to_missing": arithmetic_overflow,
            "infinite_feature_values": int(np.isinf(features[list(FEATURE_COLUMNS)].to_numpy(dtype=float, na_value=np.nan)).sum()),
        },
        "formulas": {
            "follower_following_ratio": "followers_count / max(following_count, 1)",
            "posting_frequency": "statuses_count / max(account_age_days, 1)",
            "account_age_days": "Validated account_age_days supplied by the source",
            "bio_length": "len(description), with empty/null biography length 0",
        },
        "assumptions": {
            "numeric_invalid": "Missing, non-numeric, non-finite or negative values become missing; fractional counts also become missing. Rows are retained. No numeric imputation or scaling.",
            "zero_denominators": "Valid following=0 or age=0 uses denominator 1; missing/invalid inputs remain missing.",
            "account_age": "Historical source-supplied age; no observed_at is available. No use of today's date or download date, and no fabricated observation timestamp.",
            "posting_frequency": "Lifetime average posts/day, not recent posting activity.",
            "bio": "Empty/null bio has length 0; unavailable bio cannot be distinguished from truly empty bio. Literal None/NA/nan text is preserved. Whitespace and newlines count as characters; Unicode length counts code points, not grapheme clusters.",
            "identity": "Missing/duplicate IDs and invalid/conflicting labels stop processing; no silent row removal. ID and label are excluded from model inputs.",
        },
        "posting_frequency_comparison": {
            "source_column": "average_tweets_per_day",
            "assumed_rounding_decimals": POSTING_ROUNDING_DECIMALS,
            "absolute_tolerance": tolerance, "floating_point_margin": 1e-12,
            "comparable_rows": int(comparable.sum()),
            "uncomparable_rows": int((~comparable).sum()),
            "within_rounding_tolerance": int(within.sum()),
            "beyond_rounding_tolerance": int((~within).sum()),
            "max_absolute_difference": float(differences.max()) if len(differences) else None,
            "mean_absolute_difference": float(differences.mean()) if len(differences) else None,
            "interpretation": "Comparison only: differences beyond assumed rounding tolerance are reported, not automatically treated as data errors.",
        },
    }


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run_pipeline(
    source_path: str | Path = RAW_LABELS_PATH,
    output_path: str | Path = FEATURES_PATH,
    summary_path: str | Path = FEATURE_SUMMARY_PATH,
) -> tuple[pd.DataFrame, dict]:
    """Validate, extract, verify ID/label alignment and persist CSV plus JSON."""
    source_path, output_path, summary_path = map(Path, (source_path, output_path, summary_path))
    resolved = [p.resolve() for p in (source_path, output_path, summary_path)]
    if len(set(resolved)) != 3:
        raise ValueError("Source, features and summary paths must be different; source cannot be overwritten.")
    before = file_sha256(source_path)
    profiles, quality = load_profiles(source_path)
    features = build_features(profiles)
    if not features["user_id"].equals(profiles["user_id"]) or not features["label"].equals(profiles["label"]):
        raise ValueError("Output IDs/labels do not match normalized source rows.")
    summary = summarize_features(profiles, features, quality)
    after = file_sha256(source_path)
    if before != after:
        raise ValueError("Source CSV changed while the pipeline was running; outputs were not written.")
    summary["source"] = {
        "file": source_path.name, "sha256_before": before,
        "sha256_after": after, "unchanged": True,
    }
    summary["verification"] = {"id_label_alignment": True, "row_order_preserved": True}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False, encoding="utf-8", na_rep="")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return features, summary


if __name__ == "__main__":
    result, report = run_pipeline()
    print(f"Features: {FEATURES_PATH} ({len(result)} rows, {len(result.columns)} columns)")
    print(f"Quality report: {FEATURE_SUMMARY_PATH}")
    print(f"Rows removed: {report['data_quality']['rows_removed']}; source unchanged: {report['source']['unchanged']}")
