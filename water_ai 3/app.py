from __future__ import annotations
# lets me use newer type hint syntax cleanly

import os

import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request
from werkzeug.utils import secure_filename

from predict import predict_image

# --- project folders ---
# BASE_DIR points to the backend folder this file lives in.
BASE_DIR = Path(__file__).resolve().parent

# Every uploaded image gets copied into uploads/ first.
# I kept this folder local so it is easy to inspect what was tested during demos.
UPLOAD_DIR = BASE_DIR / 'uploads'
UPLOAD_DIR.mkdir(exist_ok=True)

# Small allowlist on purpose:
# keeps the app from trying to process random unsupported file types.
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # keep uploads lightweight for quick inference

# ai usage here heavily 
HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Water Detector | Clean Water Detector AI</title>
  <style>
    :root {
      --bg1: #fff6fb;
      --bg2: #fdf2ff;
      --card: rgba(255, 255, 255, 0.82);
      --border: rgba(236, 72, 153, 0.18);
      --text: #4a1d3f;
      --muted: #8b5d7a;
      --accent: #ec4899;
      --accent-2: #f472b6;
      --accent-3: #c084fc;
      --good-bg: #fff1f7;
      --good-border: #f9a8d4;
      --shadow: 0 18px 45px rgba(236, 72, 153, 0.12);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, #ffe4f1 0%, transparent 28%),
        radial-gradient(circle at top right, #f5d9ff 0%, transparent 25%),
        linear-gradient(135deg, var(--bg1), var(--bg2));
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 32px 16px;
    }

    .shell {
      width: 100%;
      max-width: 980px;
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 22px;
    }

    .card {
      background: var(--card);
      backdrop-filter: blur(10px);
      border: 1px solid var(--border);
      border-radius: 28px;
      box-shadow: var(--shadow);
      padding: 28px;
    }

    .hero h1 {
      font-size: 2.35rem;
      margin: 0 0 10px;
      line-height: 1.1;
    }

    .hero p {
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.7;
      margin-bottom: 24px;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 14px;
      border-radius: 999px;
      background: rgba(255,255,255,0.75);
      border: 1px solid var(--border);
      font-size: 0.9rem;
      margin-bottom: 18px;
      color: var(--accent);
      font-weight: 700;
    }

    .form-wrap {
      background: rgba(255,255,255,0.78);
      border: 1px dashed rgba(236, 72, 153, 0.28);
      border-radius: 24px;
      padding: 20px;
    }

    .label {
      display: block;
      font-size: 0.95rem;
      margin-bottom: 10px;
      color: var(--muted);
      font-weight: 700;
    }

    input[type="file"] {
      width: 100%;
      padding: 14px;
      border-radius: 16px;
      border: 1px solid rgba(236, 72, 153, 0.22);
      background: #fff;
      color: var(--text);
      margin-bottom: 14px;
    }

    .btn {
      width: 100%;
      border: none;
      border-radius: 16px;
      padding: 14px 18px;
      cursor: pointer;
      color: white;
      font-weight: 700;
      font-size: 1rem;
      background: linear-gradient(135deg, var(--accent), var(--accent-3));
      box-shadow: 0 10px 25px rgba(236, 72, 153, 0.25);
    }

    .btn:hover {
      transform: translateY(-1px);
    }

    .small-note {
      margin-top: 12px;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.55;
    }

    .side-card h2,
    .result h2 {
      margin-top: 0;
      margin-bottom: 14px;
    }

    .flow {
      display: grid;
      gap: 12px;
      margin-top: 16px;
    }

    .step {
      border-radius: 18px;
      padding: 14px 16px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(192, 132, 252, 0.25);
    }

    .step strong {
      display: block;
      margin-bottom: 4px;
      color: var(--accent);
    }

    .muted {
      color: var(--muted);
    }

    .result {
      margin-top: 18px;
      padding: 18px;
      background: var(--good-bg);
      border-radius: 22px;
      border: 1px solid var(--good-border);
    }

    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }

    .metric {
      background: rgba(255,255,255,0.8);
      border-radius: 16px;
      padding: 14px;
      border: 1px solid rgba(236, 72, 153, 0.16);
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 0.88rem;
      margin-bottom: 5px;
    }

    .metric strong {
      font-size: 1.1rem;
    }

    .error {
      margin-top: 18px;
      padding: 16px;
      border-radius: 18px;
      border: 1px solid #f9a8d4;
      background: #fff1f2;
      color: #9d174d;
    }

    @media (max-width: 900px) {
      .shell { grid-template-columns: 1fr; }
      .hero h1 { font-size: 1.95rem; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="card hero">
      <div class="badge"> AI + Water Safety Demo</div>
      <h1>Water Detector:<br>Clean Water Detector AI</h1>
      <p>
        Upload a water image and the model estimates whether the water appears <strong>clean</strong>
        or <strong>dirty</strong> based on visual patterns like brightness, texture, edge clutter,
        and color distribution.
      </p>

      <div class="form-wrap">
        <form method="post" enctype="multipart/form-data">
          <label class="label" for="image">Choose a water image</label>
          <input id="image" type="file" name="image" accept="image/*" required>
          <button class="btn" type="submit">Analyze Water Image</button>
        </form>
        <div class="small-note">
          The backend saves the upload, extracts image features, runs the trained model,
          then returns clean vs dirty confidence scores.
        </div>
      </div>

      {% if error %}
        <div class="error">
          <strong>Upload issue:</strong> {{ error }}
        </div>
      {% endif %}

      {% if result %}
        <div class="result">
          <h2>Prediction: {{ result['prediction']|upper }}</h2>
          <p class="muted">Here is what water_ai 3 thinks about the uploaded image.</p>
          <div class="metric-grid">
            <div class="metric">
              <span>Clean probability</span>
              <strong>{{ result['clean_probability'] }}</strong>
            </div>
            <div class="metric">
              <span>Dirty probability</span>
              <strong>{{ result['dirty_probability'] }}</strong>
            </div>
            <div class="metric" style="grid-column: 1 / -1;">
              <span>Saved upload path</span>
              <strong style="font-size:0.96rem; word-break: break-word;">{{ result['image'] }}</strong>
            </div>
          </div>
        </div>
      {% endif %}
    </div>

    <div class="card side-card">
      <h2>What the app is doing</h2>
      <p class="muted">
        Flow
      </p>

      <div class="flow">
        <div class="step">
          <strong>1) User uploads an image</strong>
          The Flask app accepts a JPG, PNG, JPEG, or WEBP image and stores it in the uploads folder.
        </div>
        <div class="step">
          <strong>2) Features get extracted</strong>
          The backend converts the image into measurable values like color statistics, grayscale texture,
          edge strength, and tiny thumbnail pixel patterns.
        </div>
        <div class="step">
          <strong>3) Model makes a prediction</strong>
          The trained classifier reads those features and outputs a probability for the image being clean water.
        </div>
        <div class="step">
          <strong>4) Result is shown back</strong>
          The UI displays the final label plus clean/dirty confidence so the user sees more than just one word.
        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""
## until here

def allowed_file(filename: str) -> bool:
    """Check the file extension before saving anything to disk."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# I pulled the save logic into its own helper so both the HTML route and the API route
# behave the same way and I do not repeat the same few lines twice.
def save_uploaded_file(file_storage) -> Path:
    original_name = secure_filename(file_storage.filename)
    extension = original_name.rsplit('.', 1)[1].lower()

    # uuid keeps uploads from overwriting each other during repeated tests.
    save_path = UPLOAD_DIR / f"{uuid.uuid4()}.{extension}"
    file_storage.save(save_path)
    return save_path


@app.route('/', methods=['GET', 'POST'])
def home():
    # result holds the model output.
    # error is shown if the upload is missing or invalid.
    result = None
    error = None

    if request.method == 'POST':
        # Step 1: grab the uploaded file from the HTML form.
        file = request.files.get('image')

        if not file or not file.filename:
            error = 'Please upload an image first.'
        elif not allowed_file(file.filename):
            error = 'Unsupported file type. Please use png, jpg, jpeg, or webp.'
        else:
            try:
                # Step 2: store the image locally so the model can read it from disk.
                save_path = save_uploaded_file(file)

                # Step 3: hand the saved image path to predict.py.
                # That file loads the trained model and computes the clean/dirty probabilities.
                result = predict_image(save_path)
            except Exception as exc:
                # During demos this makes debugging easier than silently failing.
                error = f'Prediction failed: {exc}'

    # Step 4: render the page with either a result card or an error card.
    return render_template_string(HTML, result=result, error=error)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    # Same core logic as the HTML route, but returned as JSON so this could be reused
    # by a frontend, mobile app, or other service later.
    file = request.files.get('image')

    if not file or not file.filename:
        return jsonify({'error': 'No image uploaded'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    try:
        save_path = save_uploaded_file(file)
        return jsonify(predict_image(save_path))
    except Exception as exc:
        return jsonify({'error': f'Prediction failed: {exc}'}), 500


if __name__ == '__main__':
    # debug=True is fine here because this is a local prototype / hackathon demo build.
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))
