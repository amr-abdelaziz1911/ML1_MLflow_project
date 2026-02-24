## Hand Gesture Classification (HaGRID + MediaPipe Landmarks)

This project trains machine learning models to classify hand gestures using MediaPipe hand landmarks extracted from the HaGRID dataset. The input is a CSV file of hand landmarks (21 points × \(x,y,z\)) and labels; the output is a trained classifier and a demo script that overlays predictions on a video.

### Project structure

- `data/`
  - `hand_landmarks.csv` – input dataset (not included in repo; place it here).
- `src/`
  - `data_preprocessing.py` – loading, recentering/normalizing landmark data, and train/test split.
  - `model_training.py` – training multiple ML models, evaluating, and saving the best model and comparison plots.
  - `video_inference.py` – runs the trained model on a video, uses MediaPipe to get landmarks per frame, and writes an annotated output video.
- `models/`
  - `best_model.joblib` – saved best-performing model (created after training).
- `reports/`
  - `metrics_summary.csv` – per-model metrics.
  - `figures/model_comparison.png` – comparison chart between models.
- `notebooks/`
  - `ML1_hand_gesture_classification.ipynb` – Colab-ready notebook implementing the full pipeline (no MLflow on `main`).

On the `research` branch, there will also be:

- `mlflow_utils.py` – helper functions wrapping all MLflow usage.
- Updated notebook and training script that import and use `mlflow_utils`.
- `mlruns/` – MLflow runs, metrics, parameters, and artifacts.
- `reports/mlflow_screenshots/` – screenshots from MLflow UI (runs, charts, registry).

### Dataset

Place your CSV file as:

- `data/hand_landmarks.csv`

The file is expected to contain:

- Landmark features: columns like `x_0, y_0, z_0, ..., x_20, y_20, z_20`.
- A target label column called `gesture` with one of the 18 HaGRID gesture names.

The script will:

1. Recenter \(x, y\) coordinates so that the wrist landmark (index 0) is the origin.
2. Scale \(x, y\) coordinates by dividing by the distance to the middle finger tip (index 12).
3. Leave \(z\) coordinates as-is.

### How to run (local)

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place `hand_landmarks.csv` under `data/`.
4. Train and evaluate models:

```bash
python -m src.model_training
```

This will:

- Train at least three models (e.g., Random Forest, SVM, KNN).
- Save `models/best_model.joblib`.
- Write metrics to `reports/metrics_summary.csv`.
- Save a comparison plot to `reports/figures/model_comparison.png`.

5. Run the video demo:

```bash
python -m src.video_inference --input_video path/to/input.mp4 --output_video outputs/annotated_output.mp4
```

This will:

- Read the video frame by frame.
- Use MediaPipe Hands to detect landmarks.
- Use the trained classifier to predict the gesture.
- Stabilize predictions over a sliding window using the mode.
- Draw landmarks and the predicted label in the top-left corner.

### Colab usage

Upload the repository to GitHub, then open the notebook `notebooks/ML1_hand_gesture_classification.ipynb` in Google Colab. The notebook:

- Mounts your Drive or fetches the CSV from GitHub.
- Mirrors the training pipeline from the scripts.
- Produces the same metrics and plots as the local run.

### MLflow (research branch only)

On the `research` branch:

- All MLflow functions live in `mlflow_utils.py`.
- The notebook and training script use these helpers to:
  - Log experiments, runs, parameters, metrics, models, and artifacts.
  - Compare runs and choose the best model from the UI.
  - Register the selected model into the MLflow Model Registry.

The `research` branch will also contain:

- The `mlruns/` directory.
- Screenshots in `reports/mlflow_screenshots/`.
- An extended README section explaining which model was chosen and why, with a comparison table/plot.


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>accuracy</th>
      <th>precision_macro</th>
      <th>recall_macro</th>
      <th>f1_macro</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RandomForest</td>
      <td>0.979163</td>
      <td>0.978995</td>
      <td>0.979027</td>
      <td>0.978956</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SVM-RBF</td>
      <td>0.978189</td>
      <td>0.978131</td>
      <td>0.978161</td>
      <td>0.977977</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KNN</td>
      <td>0.973126</td>
      <td>0.973140</td>
      <td>0.972973</td>
      <td>0.972896</td>
    </tr>
  </tbody>
</table>
</div>

