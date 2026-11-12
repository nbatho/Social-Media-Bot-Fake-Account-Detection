"""Đọc dữ liệu, tạo 4 đặc trưng và chia tập train/validation/test."""

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURES_PATH, LABEL_MAPPING, RANDOM_STATE, RAW_DATA_PATH, SPLIT_NAMES, SPLIT_PATH

NUMERIC_COLUMNS = ["followers_count", "friends_count", "statuses_count", "account_age_days"]


def load_profiles(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Đọc CSV; dừng nếu ID/nhãn lỗi. Số không hợp lệ hoặc âm thành giá trị thiếu."""
    # keep_default_na=False: giữ nguyên bio là chữ "None"/"NA" thay vì biến thành thiếu.
    df = pd.read_csv(path, dtype={"id": "string", "description": "string"}, keep_default_na=False)
    missing = {"id", "account_type", "description", *NUMERIC_COLUMNS} - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột: {sorted(missing)}")
    if df["id"].str.strip().eq("").any() or df["id"].duplicated().any():
        raise ValueError("ID rỗng hoặc bị trùng.")
    if not df["account_type"].isin(LABEL_MAPPING).all():
        raise ValueError(f"Nhãn phải là {list(LABEL_MAPPING)}.")
    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce")
        df[column] = values.mask(values < 0)
    df["label"] = df["account_type"].map(LABEL_MAPPING)
    return df.rename(columns={"id": "user_id"})


def build_features(profiles: pd.DataFrame) -> pd.DataFrame:
    """Tính 4 đặc trưng. Mẫu số bằng 0 thì dùng 1; mẫu số thiếu thì kết quả thiếu."""
    features = pd.DataFrame({"user_id": profiles["user_id"]})
    features["follower_following_ratio"] = profiles["followers_count"] / profiles["friends_count"].clip(lower=1)
    features["posting_frequency"] = profiles["statuses_count"] / profiles["account_age_days"].clip(lower=1)
    features["account_age_days"] = profiles["account_age_days"]
    features["bio_length"] = profiles["description"].str.len()  # bio rỗng có độ dài 0
    features["label"] = profiles["label"]
    return features


def split_features(features: pd.DataFrame, path=SPLIT_PATH):
    """Trả về (train, validation, test). Lần đầu chia 60/20/20 và lưu split.csv; các lần sau dùng lại."""
    if path.exists():
        split = pd.read_csv(path, dtype={"user_id": "string"}).set_index("user_id")["split"]
        if set(split.index) != set(features["user_id"]):
            raise ValueError(f"{path} không khớp với dữ liệu hiện tại.")
    else:
        rest, test = train_test_split(features, test_size=0.2, stratify=features["label"], random_state=RANDOM_STATE)
        train, val = train_test_split(rest, test_size=0.25, stratify=rest["label"], random_state=RANDOM_STATE)
        split = pd.concat([pd.Series(name, index=part["user_id"]) for name, part in
                           zip(SPLIT_NAMES, (train, val, test))]).rename("split")
        path.parent.mkdir(parents=True, exist_ok=True)
        split.rename_axis("user_id").reset_index().to_csv(path, index=False)
    assigned = features["user_id"].map(split)
    return tuple(features[assigned == name] for name in SPLIT_NAMES)


def run_pipeline():
    """Chạy bằng: python -m src.data  → lưu dataset/processed/features.csv."""
    features = build_features(load_profiles())
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_PATH, index=False)
    return features


if __name__ == "__main__":
    result = run_pipeline()
    print(f"Đã lưu {FEATURES_PATH} ({len(result)} dòng)")
