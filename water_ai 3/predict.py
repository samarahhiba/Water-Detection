from __future__ import annotations # lets me use newer type hint syntax cleanly


import argparse # used so this file can also be run from the terminal with an image path

from pathlib import Path # makes file paths cleaner and more portable than hardcoded strings

import joblib # joblib loads the saved sklearn model pipeline from disk


from train_model import extract_features # reuse the exact same feature extraction logic from training
# inferenec must match trainig


# path to the trained model artifact produced by train_model.py
MODEL_PATH = Path(__file__).resolve().parent / 'model' / 'clean_water_detector.joblib'


def predict_image(image_path: str | Path) -> dict:
    """
    Full prediction flow:
    1. load the trained classifier from disk
    2. turn the image into the same feature vector used during training
    3. ask the model for class probabilities
    4. package the result for the UI / API
    """
    # load the saved pipeline from disk
    # this avoids retraining every time the app starts
    model = joblib.load(MODEL_PATH)

    
    # turn one image into the model's expected feature vector
    # reshape(1, -1) makes it 2D because sklearn expects rows of samples
    features = extract_features(image_path).reshape(1, -1)

    # index 1 = probability of the positive class, which I mapped to "clean" during training.
    clean_prob = float(model.predict_proba(features)[0, 1])
    # convert the probability into a final label
    label = 'clean' if clean_prob >= 0.5 else 'dirty'

    # package everything in a form that is easy to understand
    return {
        'image': str(image_path),
        'prediction': label,
        'clean_probability': round(clean_prob, 4),
        'dirty_probability': round(1 - clean_prob, 4),
    }


def main() -> None:
    # lets me test the model directly from terminal like:
    # python predict.py some_image.jpg
    parser = argparse.ArgumentParser(description='Predict whether a water image is clean or dirty.')
        
    # required command-line argument = path to image file
    parser.add_argument('image', help='Path to an image file')
        
    # parse command-line inputs
    args = parser.parse_args()

    #  test from terminal without opening the web app.
    result = predict_image(args.image)
    print(result)

    
# only run CLI testing mode if this file is executed directly
if __name__ == '__main__':
    main()
