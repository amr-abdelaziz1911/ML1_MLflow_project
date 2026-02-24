import argparse
from collections import deque, Counter
from pathlib import Path
from typing import Deque, List, Optional

import cv2
import joblib
import numpy as np

# Use the new MediaPipe Tasks API for hand landmarks.
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

try:
    from .data_preprocessing import WRIST_INDEX, MIDDLE_FINGER_TIP_INDEX
except ImportError:  # fallback when run as plain script
    from data_preprocessing import WRIST_INDEX, MIDDLE_FINGER_TIP_INDEX


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def preprocess_landmarks_for_model(landmarks: List) -> np.ndarray:
    """Apply the same recenter + scale transform as in training."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    wrist = coords[WRIST_INDEX, :2]
    coords[:, 0:2] -= wrist

    middle = coords[MIDDLE_FINGER_TIP_INDEX, 0:2]
    scale = np.linalg.norm(middle)
    if scale < 1e-6:
        scale = 1e-6
    coords[:, 0:2] /= scale

    features = coords.flatten()[None, :]
    return features


def sliding_window_mode(preds_window: Deque[str]) -> str:
    counter = Counter(preds_window)
    return counter.most_common(1)[0][0]


def run_video_inference(
    input_video: str,
    output_video: str,
    model_path: Optional[str] = None,
    window_size: int = 7,
):
    if model_path is None:
        model_path = MODELS_DIR / "best_model.joblib"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    clf = joblib.load(model_path)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Ensure the hand landmarker model file exists (download if needed).
    model_path = PROJECT_ROOT / "hand_landmarker.task"
    if not model_path.exists():
        url = (
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )
        print("Downloading hand_landmarker.task model... This is a one-time download.")
        urllib.request.urlretrieve(url, model_path)

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    preds_window: Deque[str] = deque(maxlen=window_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe Tasks expects RGB images.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            features = preprocess_landmarks_for_model(hand_landmarks)
            pred = clf.predict(features)[0]
            preds_window.append(str(pred))
            smoothed_pred = sliding_window_mode(preds_window)

            # Draw landmarks (simple circles instead of the old connections API)
            for lm in hand_landmarks:
                x = int(lm.x * width)
                y = int(lm.y * height)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            cv2.rectangle(frame, (10, 10), (260, 60), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"{smoothed_pred}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        out.write(frame)

    cap.release()
    out.release()


def parse_args():
    parser = argparse.ArgumentParser(description="Run hand-gesture classifier on a video.")
    parser.add_argument("--input_video", type=str, required=True, help="Path to input video.")
    parser.add_argument(
        "--output_video",
        type=str,
        required=True,
        help="Path to save annotated output video.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional custom path to trained model.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=7,
        help="Sliding window size for mode-based smoothing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_video_inference(
        input_video=args.input_video,
        output_video=args.output_video,
        model_path=args.model_path,
        window_size=args.window_size,
    )

