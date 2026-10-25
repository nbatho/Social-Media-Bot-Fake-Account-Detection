"""Project paths and conventions shared by all notebooks."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_LABELS_PATH = DATASET_DIR / "twitter_human_bots_dataset.csv"
PROCESSED_DIR = DATASET_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FEATURES_PATH = PROCESSED_DIR / "features.csv"
FEATURE_SUMMARY_PATH = REPORTS_DIR / "feature_summary.json"

RANDOM_STATE = 42
LABEL_MAPPING = {"human": 0, "bot": 1}
SPLIT_FRACTIONS = {"train": 0.6, "validation": 0.2, "test": 0.2}

RAW_PROFILE_COLUMNS = (
    "id",
    "account_type",
    "followers_count",
    "friends_count",
    "statuses_count",
    "created_at",
    "description",
    "account_age_days",
    "average_tweets_per_day",
)
COLUMN_MAPPING = {"id": "user_id", "friends_count": "following_count"}
PROFILE_COLUMNS = tuple(COLUMN_MAPPING.get(c, c) for c in RAW_PROFILE_COLUMNS)
COUNT_COLUMNS = ("followers_count", "following_count", "statuses_count")
NUMERIC_COLUMNS = (*COUNT_COLUMNS, "account_age_days", "average_tweets_per_day")
# Numeric markers never apply to biography text or IDs.
NUMERIC_MISSING_MARKERS = ("", "na", "n/a", "nan", "null", "none")
POSTING_ROUNDING_DECIMALS = 3

FEATURE_COLUMNS = (
    "follower_following_ratio",
    "posting_frequency",
    "account_age_days",
    "bio_length",
)
FEATURE_OUTPUT_COLUMNS = ("user_id", *FEATURE_COLUMNS, "label")
