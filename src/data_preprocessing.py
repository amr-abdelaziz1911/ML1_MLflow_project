import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


WRIST_INDEX = 0
MIDDLE_FINGER_TIP_INDEX = 12


def load_hand_landmarks(csv_path: str) -> pd.DataFrame:
    """Load the hand landmarks CSV into a DataFrame."""
    return pd.read_csv(csv_path)


def get_feature_and_label_columns(df: pd.DataFrame, label_col: str = "gesture"):
    """Split landmark feature columns and label column names."""
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset columns.")

    feature_cols = [c for c in df.columns if c != label_col]
    return feature_cols, label_col


def recenter_and_normalize_landmarks(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Recenters (x, y) so that the wrist is the origin and scales by the
    distance to the middle finger tip. z is left as-is.
    Assumes columns are named x_0, y_0, z_0, ..., x_20, y_20, z_20.
    """
    df = df.copy()

    wrist_x = df[f"x_{WRIST_INDEX}"]
    wrist_y = df[f"y_{WRIST_INDEX}"]

    # Recenter x, y by subtracting wrist coordinates
    for i in range(21):
        df[f"x_{i}"] = df[f"x_{i}"] - wrist_x
        df[f"y_{i}"] = df[f"y_{i}"] - wrist_y

    # Scale by distance to middle finger tip
    middle_x = df[f"x_{MIDDLE_FINGER_TIP_INDEX}"]
    middle_y = df[f"y_{MIDDLE_FINGER_TIP_INDEX}"]
    scale = np.sqrt(middle_x**2 + middle_y**2)
    scale = scale.replace(0, 1e-6)

    for i in range(21):
        df[f"x_{i}"] = df[f"x_{i}"] / scale
        df[f"y_{i}"] = df[f"y_{i}"] / scale

    return df[feature_cols]


def train_test_split_landmarks(
    df: pd.DataFrame,
    label_col: str = "gesture",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Return train/test split with preprocessing applied to features."""
    feature_cols, label_col = get_feature_and_label_columns(df, label_col)
    X_processed = recenter_and_normalize_landmarks(df, feature_cols)
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test

