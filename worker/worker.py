"""
Worker Module for URL Malware Classification
--------------------------------------------

This module runs inside a Celery worker and performs:

1. Load trained model assets (model, encoder, scaler, selected features)
2. Receive URL from backend task
3. Convert URL → model-ready feature vector (using the inference feature extractor)
4. Apply encoder + scaler (if used during training)
5. Return prediction + probability score

This version is fully synchronized with the training pipeline.
"""

from celery import Celery
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

# IMPORTANT:
# Use the unified inference pipeline from backend.feature_extractor
from backend.feature_extractor import prepare_features_for_model


# -------------------------------------------------------------
# 1. Asset paths
# -------------------------------------------------------------
ASSET_PATH = os.path.join(os.getcwd(), "assets")

MODEL = None
ENCODER = None
SCALER = None
SELECTED_FEATURES = None
ASSETS_LOADED = False


# -------------------------------------------------------------
# 2. Load model assets
# -------------------------------------------------------------
try:
    MODEL = joblib.load(os.path.join(ASSET_PATH, "best_model_xgb.pkl"))
    ENCODER = joblib.load(os.path.join(ASSET_PATH, "categorical_encoder.pkl"))
    SCALER = joblib.load(os.path.join(ASSET_PATH, "feature_scaler.pkl"))

    with open(os.path.join(ASSET_PATH, "selected_features.txt"), "r") as f:
        SELECTED_FEATURES = [line.strip() for line in f.readlines() if line.strip()]

    print("✅ Successfully loaded model assets from:", ASSET_PATH)
    ASSETS_LOADED = True

except Exception as e:
    print("❌ Failed to load assets:", e)
    ASSETS_LOADED = False


# -------------------------------------------------------------
# 3. Celery configuration
# -------------------------------------------------------------
app = Celery("tasks", broker="redis://redis:6379/0")


# -------------------------------------------------------------
# 4. Apply encoder and scaler (if exist)
# -------------------------------------------------------------
def apply_postprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply categorical encoding + numerical scaling exactly as done during training.
    """

    df = df.copy()

    # 1. Encoder
    if ENCODER is not None:
        try:
            categorical_cols = ENCODER.feature_names_in_
            df[categorical_cols] = ENCODER.transform(df[categorical_cols])
        except Exception as e:
            print("⚠ Encoder error:", e)

    # 2. Scaler
    if SCALER is not None:
        try:
            numeric_cols = SCALER.feature_names_in_
            df[numeric_cols] = SCALER.transform(df[numeric_cols])
        except Exception as e:
            print("⚠ Scaler error:", e)

    return df


# -------------------------------------------------------------
# 5. Main prediction pipeline
# -------------------------------------------------------------
def predict_url(url: str):
    """
    Complete inference pipeline for a single URL:
    - Extract raw training-compatible features
    - Align with training columns
    - Apply encoder + scaler
    - Predict using XGBoost model
    """

    if not ASSETS_LOADED:
        return 0, 0.0

    # 1. Extract model-ready features
    df = prepare_features_for_model(url)

    # 2. Apply encoder + scaler
    df = apply_postprocessing(df)

    # 3. Run model prediction
    try:
        y_pred = MODEL.predict(df)[0]
        y_score = float(MODEL.predict_proba(df)[:, 1][0])
    except Exception as e:
        print("❌ Model prediction error:", e)
        return 0, 0.0

    return int(y_pred), float(y_score)


# -------------------------------------------------------------
# 6. Celery task that backend will call
# -------------------------------------------------------------
@app.task(name="analyze_url_for_malware")
def analyze_url_for_malware(url: str):
    """
    Public Celery task used by the backend API.
    Returns:
      - URL
      - Prediction
      - Probability score
      - Status
      - Timestamp
    """

    if not ASSETS_LOADED:
        return {
            "url": url,
            "is_malicious": False,
            "score": 0.0,
            "error": "Model assets not loaded",
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }

    prediction, score = predict_url(url)

    return {
        "url": url,
        "is_malicious": bool(prediction),
        "score": score,
        "status": "completed",
        "timestamp": datetime.now().isoformat()
    }
