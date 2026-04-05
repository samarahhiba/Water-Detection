# Clean Water Detector AI

This project is a small binary image classification setup for telling the difference between clean water and dirty water using the images included in the kaggle dataset.

## What it does
- Trains on `train/Clean-samples` vs `train/Dirty-samples`
- Evaluates on `test/Clean-samples` vs `test/Dirty-samples`
- Saves a reusable model
- Provides both:
  - a CLI predictor
  - a simple Flask web app

## Dataset layout
The dataset inside this project is already placed under:

```text
data/water images/
  train/
    Clean-samples/
    Dirty-samples/
  test/
    Clean-samples/
    Dirty-samples/
```

## Quick start

```bash
cd water_ai\ 3
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Then open:

```text
http://127.0.0.1:5001
```

## CLI usage
You can also test individual images from the terminal:

```bash
python predict.py "data/water images/test/Clean-samples/29.jpg"
python predict.py "data/water images/test/Dirty-samples/15.jpg"
```

## Notes
- This is a lightweight classical ML detector, which is a good fit for a very small dataset.
- Because the dataset is tiny, the reported score can look very strong while still being brittle on new real-world images.
- For better generalization, add many more examples of clean and dirty water from different angles, lighting conditions, and backgrounds.

## A couple of practical notes

This project uses a lightweight classical machine learning approach instead of a heavier deep learning setup. That makes sense here because the dataset is pretty small.

One thing to keep in mind is that with a tiny dataset, results can sometimes look better than they really are. A high score on the test set does not always mean the model will behave well on new images in real conditions. I have personally taken iamges that are not being detected as clean when they obviously are

To make it more reliable, the dataset should eventually include a lot more clean and dirty water examples with different:

lighting
camera angles
backgrounds
water colors and textures

## Citations 
I really don't like working too long on html files in time constraint environs. I was a bit confused about the AI restrictions. I haven't done a hackathon in 3 years. Not sure what changed. Don't judge if there happens to be an AI restriction
https://chatgpt.com/c/69d2034d-1d5c-8327-ae79-acf920746cbe

i basically used most of this for app.py line 29 - 319
