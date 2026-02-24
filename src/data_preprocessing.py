import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List


# Indices in MediaPipe space (0-based) – still useful for live video,
# but the CSV columns are named x1..x21, y1..y21, z1..z21.
WRIST_INDEX = 0
MIDDLE_FINGER_TIP_INDEX = 12

LANDMARK_COUNT = 21


def load_hand_landmarks(csv_path: str) -> pd.DataFrame:
    """Load the hand landmarks CSV into a DataFrame."""
    return pd.read_csv(csv_path)


def get_feature_and_label_columns(df: pd.DataFrame, label_col: str = "label"):
    """
    Split landmark feature columns and label column names.
    The default label column in the provided CSV is 'label'.
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset columns.")

    feature_cols = [c for c in df.columns if c != label_col]
    return feature_cols, label_col


def recenter_and_normalize_landmarks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recenters (x, y) so that the wrist is the origin and scales by the
    distance to the middle finger tip. z is left as-is.

    Assumes columns are named:
    x1, y1, z1, ..., x21, y21, z21
    """
    df = df.copy()

    # Wrist is landmark 1 in this CSV (MediaPipe index 0).
    wrist_x = df["x1"]
    wrist_y = df["y1"]

    # Recenter x, y by subtracting wrist coordinates
    for i in range(1, LANDMARK_COUNT + 1):
        df[f"x{i}"] = df[f"x{i}"] - wrist_x
        df[f"y{i}"] = df[f"y{i}"] - wrist_y

    # Middle finger tip is landmark 13 in this CSV (MediaPipe index 12).
    middle_x = df["x13"]
    middle_y = df["y13"]
    scale = np.sqrt(middle_x**2 + middle_y**2)
    scale = scale.replace(0, 1e-6)

    for i in range(1, LANDMARK_COUNT + 1):
        df[f"x{i}"] = df[f"x{i}"] / scale
        df[f"y{i}"] = df[f"y{i}"] / scale

    # Use all landmark coordinates as features, in a stable order.
    feature_cols: List[str] = []
    for i in range(1, LANDMARK_COUNT + 1):
        feature_cols.extend([f"x{i}", f"y{i}", f"z{i}"])

    return df[feature_cols]


def train_test_split_landmarks(
    df: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Return train/test split with preprocessing applied to features."""
    _, label_col = get_feature_and_label_columns(df, label_col)
    X_processed = recenter_and_normalize_landmarks(df)
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test

