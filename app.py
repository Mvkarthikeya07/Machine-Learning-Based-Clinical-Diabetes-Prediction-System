# app.py
# Modified Flask app: loads model (tries model.joblib then model_pipeline.joblib),
# provides /, /health, /predict, /reload-model, /data-head and /data-tail endpoints.
# Builds a pandas DataFrame with correct feature names before prediction to avoid warnings.

import os
import traceback
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Try these filenames (prefers model.joblib if present)
MODEL_FILES = ["model.joblib", "model_pipeline.joblib", "best_model.joblib"]
MODEL_PATH = None
for fn in MODEL_FILES:
    if os.path.exists(fn):
        MODEL_PATH = fn
        break

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

app = Flask(__name__, template_folder="templates", static_folder="static")
model = None

def load_model(path=None):
    global model, MODEL_PATH
    if path:
        MODEL_PATH = path
    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        app.logger.error("Model file not found: %s", MODEL_PATH)
        model = None
        return False, f"model file not found: {MODEL_PATH}"
    try:
        model = joblib.load(MODEL_PATH)
        app.logger.info("Loaded model from %s", MODEL_PATH)
        return True, None
    except Exception as e:
        app.logger.error("Error loading model: %s", e)
        app.logger.error(traceback.format_exc())
        model = None
        return False, str(e)

# Attempt initial load
ok, err = load_model()
if not ok:
    app.logger.warning("Model not loaded on startup: %s", err)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"ok": True, "model_loaded": model is not None, "model_path": MODEL_PATH})

def build_input_dataframe(data):
    """
    Build a pandas DataFrame with correct column names so sklearn pipeline
    sees the expected feature names (avoids the 'invalid feature names' warning).
    Accepts dict-like data from request.form or JSON.
    """
    values = []
    for f in FEATURES:
        v = data.get(f, None)
        if v is None or v == "":
            values.append(np.nan)
            continue
        try:
            values.append(float(v))
        except Exception:
            # fallback: try removing commas then float
            try:
                values.append(float(str(v).replace(",", "")))
            except Exception:
                values.append(np.nan)
    df = pd.DataFrame([values], columns=FEATURES)
    return df

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run training and then /reload-model or ensure model file exists."}), 500
    try:
        payload = request.get_json() if request.is_json else request.form
        X_df = build_input_dataframe(payload)

        pred = int(model.predict(X_df)[0])
        prob = None
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X_df)[0][1])

        return jsonify({"prediction": pred, "probability": prob})
    except Exception as exc:
        app.logger.error("Prediction error: %s", exc)
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(exc)}), 500

@app.route("/reload-model", methods=["POST"])
def reload_model():
    ok, err = load_model()
    if ok:
        return jsonify({"reloaded": True, "model_path": MODEL_PATH})
    else:
        return jsonify({"reloaded": False, "error": err}), 500

@app.route("/data-head")
def data_head():
    try:
        df = pd.read_csv("diabetes_user.csv")
        # return first 5 rows as JSON records
        return jsonify({"head": df.head().to_dict(orient="records")})
    except Exception as e:
        app.logger.error("data-head error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/data-tail")
def data_tail():
    try:
        df = pd.read_csv("diabetes_user.csv")
        return jsonify({"tail": df.tail().to_dict(orient="records")})
    except Exception as e:
        app.logger.error("data-tail error: %s", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run on localhost only; debug True for development (shows tracebacks)
    app.run(debug=True, host="127.0.0.1", port=5000)
