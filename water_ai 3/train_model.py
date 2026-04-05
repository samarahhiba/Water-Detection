from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# data/ contains the clean vs dirty water images.
DATA_ROOT = Path(__file__).resolve().parent / 'data' / 'water images'

# model/ stores the trained classifier and the evaluation summary.
MODEL_DIR = Path(__file__).resolve().parent / 'model'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / 'clean_water_detector.joblib'
METRICS_PATH = MODEL_DIR / 'metrics.json'


def extract_features(image_path: str | Path) -> np.ndarray:
    """
    Turn one image into a numeric feature vector.

    Why this exists:
    a machine learning model cannot read a raw image path directly,
    so this function converts the image into measurable signals.

    The features here are intentionally lightweight so the project stays easy to train locally.
    """
    # Read image, force 3 channels, and resize so every sample has a consistent shape.
    img = Image.open(image_path).convert('RGB').resize((64, 64))
    # convert image to numpy array and scale pixel values to 0..1
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # build a grayscale version for brightness / texture measurements
    gray = arr.mean(axis=2)
    # store all extracted features here
    feats: list[float] = []

    # --- color summary features ---
    # For each RGB channel, collect a few basic stats + a histogram.
    # This helps the model notice overall color trends and distribution shifts.
    for c in range(3):
        ch = arr[:, :, c]
        feats += [
            float(ch.mean()),
            float(ch.std()),
            float(np.percentile(ch, 10)),
            float(np.percentile(ch, 50)),
            float(np.percentile(ch, 90)),
        ]
        hist, _ = np.histogram(ch, bins=8, range=(0, 1), density=True)
        feats += hist.astype(float).tolist()

    # --- grayscale brightness / contrast features ---
    # Useful for murkiness, dullness, and overall image intensity changes.
    feats += [
        float(gray.mean()),
        float(gray.std()),
        float(np.percentile(gray, 10)),
        float(np.percentile(gray, 50)),
        float(np.percentile(gray, 90)),
    ]

    # --- edge / texture features ---
    # The idea here: dirtier water images can have more clutter, particles, or rough texture.
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    edge = np.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2)
    feats += [
        float(edge.mean()),
        float(edge.std()),
        float(np.percentile(edge, 90)),
        float((edge > 0.15).mean()),
    ]

    # --- saturation features ---
    # Captures how vivid / flat the colors are overall.
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    sat = np.where(maxc == 0, 0, (maxc - minc) / np.maximum(maxc, 1e-6))
    feats += [float(sat.mean()), float(sat.std())]

    # --- tiny thumbnail features ---
    # This keeps a very low-resolution memory of the whole image layout.
    # It is not deep learning, but it gives the classifier a bit more structure.
    small = np.asarray(Image.open(image_path).convert('RGB').resize((8, 8)), dtype=np.float32).reshape(-1) / 255.0
    feats += small.astype(float).tolist()

    return np.array(feats, dtype=np.float32)


def load_split(split: str):
    """Load either the train split or the test split into X features and y labels."""
    X, y = [], []

    # I mapped clean -> 1 and dirty -> 0 so predict_proba(...)[0, 1] means clean probability.
    for label, folder in [(1, 'Clean-samples'), (0, 'Dirty-samples')]:
        folder_path = DATA_ROOT / split / folder
        for img_path in sorted(folder_path.glob('*.jpg')):
            X.append(extract_features(img_path))
            y.append(label)

    return np.vstack(X), np.array(y, dtype=np.int64)


def main() -> None:
    # Step 1: build the training and testing datasets.
    X_train, y_train = load_split('train')
    X_test, y_test = load_split('test')

    # Step 2: standardize the features, then fit a simple interpretable classifier.
    # Logistic regression is lightweight, fast, and good for a first practical baseline.
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=5000, random_state=42))
    ])
    model.fit(X_train, y_train)

    # Step 3: evaluate on held-out test data.
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(
            y_test,
            y_pred,
            target_names=['dirty', 'clean'],
            output_dict=True,
            zero_division=0,
        ),
        # Keeping raw probabilities is useful if I want to inspect confidence later.
        'test_probabilities_clean': y_proba.round(4).tolist(),
    }

    # Step 4: save  for reuse.
    # clean_water_detector.joblib = deployed model file
    # metrics.json = lightweight record of how this trained version performed
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f'Saved model to {MODEL_PATH}')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
