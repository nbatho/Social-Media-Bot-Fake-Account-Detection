"""Đường dẫn và quy ước dùng chung cho mọi notebook."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "dataset" / "twitter_human_bots_dataset.csv"
FEATURES_PATH = PROJECT_ROOT / "dataset" / "processed" / "features.csv"
SPLIT_PATH = PROJECT_ROOT / "dataset" / "processed" / "split.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
LABEL_MAPPING = {"human": 0, "bot": 1}
SPLIT_NAMES = ("train", "validation", "test")  # 60 / 20 / 20

FEATURE_COLUMNS = [
    "follower_following_ratio",
    "posting_frequency",
    "account_age_days",
    "bio_length",
]
