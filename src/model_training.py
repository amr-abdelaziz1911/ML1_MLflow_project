from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

try:
    from .data_preprocessing import load_hand_landmarks, train_test_split_landmarks
except ImportError:  # fallback when run as plain script
    from data_preprocessing import load_hand_landmarks, train_test_split_landmarks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlflow_utils import run_experiment


DATA_PATH = PROJECT_ROOT / "data" / "hand_landmarks.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def get_models():
    """Return a dictionary of models to train."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, n_jobs=-1, random_state=42
        ),
        "SVM-RBF": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=15),
    }


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    print(f"=== {name} ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model, metrics


def save_metrics_table(metrics_list):
    REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    df_metrics = pd.DataFrame(metrics_list)
    csv_path = REPORTS_DIR / "metrics_summary.csv"
    df_metrics.to_csv(csv_path, index=False)
    print(f"Saved metrics summary to {csv_path}")

    plt.figure(figsize=(8, 5))
    metrics_to_plot = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    df_melted = df_metrics.melt(id_vars="model", value_vars=metrics_to_plot, var_name="metric")
    sns.barplot(data=df_melted, x="model", y="value", hue="metric")
    plt.ylim(0, 1.0)
    plt.title("Model comparison (macro metrics)")
    plt.tight_layout()
    fig_path = FIGURES_DIR / "model_comparison.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved model comparison figure to {fig_path}")

    return df_metrics


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Place 'hand_landmarks.csv' in the data/ folder."
        )

    df = load_hand_landmarks(str(DATA_PATH))
    X_train, X_test, y_train, y_test = train_test_split_landmarks(df)

    models = get_models()
    metrics_list = []
    trained_models = {}

    for name, model in models.items():
        trained_model, metrics = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        metrics_list.append(metrics)
        trained_models[name] = trained_model

        # Log this run to MLflow (research branch only).
        run_experiment(
            experiment_name="hand_gesture_classification",
            run_name=f"{name}_run",
            dataset_path=str(DATA_PATH),
            dataset_n_rows=len(df),
            model_name=name,
            model=trained_model,
            model_params=getattr(trained_model, "get_params", lambda: {})(),
            metric_values={
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
            },
            register_as=None,  # you can set a registry name for the best model later
        )

    df_metrics = save_metrics_table(metrics_list)

    # Choose best model by f1_macro
    best_row = df_metrics.sort_values("f1_macro", ascending=False).iloc[0]
    best_name = best_row["model"]
    best_model = trained_models[best_name]

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Saved best model ({best_name}) to {model_path}")


if __name__ == "__main__":
    main()

