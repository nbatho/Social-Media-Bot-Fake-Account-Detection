"""Project paths and conventions shared by all notebooks."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_LABELS_PATH = DATASET_DIR / "twitter_human_bots_dataset.csv"
PROCESSED_DIR = DATASET_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
LABEL_MAPPING = {"human": 0, "bot": 1}
SPLIT_FRACTIONS = {"train": 0.6, "validation": 0.2, "test": 0.2}

# Normalized input schema, to be populated from verified profile metadata.
PROFILE_COLUMNS = (
    "user_id",
    "account_type",
    "followers_count",
    "following_count",
    "statuses_count",
    "created_at",
    "description",
    "observed_at",
)

FEATURE_COLUMNS = (
    "follower_following_ratio",
    "posting_frequency",
    "account_age_days",
    "bio_length",
)
