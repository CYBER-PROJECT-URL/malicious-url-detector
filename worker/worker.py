# worker/worker.py
import os
import pickle
import pandas as pd
from celery import Celery
import sys

# Ensure the feature_extractor is importable (it's mounted in the container)
# We add /app to the path to import shared logic from the backend directory mount
sys.path.append('/app')
from feature_extractor import extract_all_features

# Configuration for Celery
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
celery_app = Celery('worker', broker=f'redis://{REDIS_HOST}:6379/0', backend=f'redis://{REDIS_HOST}:6379/0')

# --- Model Loading ---
MODEL = None
MODEL_PATH = '/app/models/ml_model.pkl'

# NOTE: You MUST train your model (RF/XGBoost) and save it as ml_model.pkl
# in the backend/models/ directory before running the system.
try:
    with open(MODEL_PATH, 'rb') as f:
        MODEL = pickle.load(f)
    print(f"INFO: ML Model successfully loaded from {MODEL_PATH}")
except FileNotFoundError:
    print(f"WARNING: Model file not found at {MODEL_PATH}. Running in DUMMY mode.")
except Exception as e:
    print(f"ERROR: Failed to load model: {e}. Running in DUMMY mode.")


# --- Celery Task ---
@celery_app.task(name='worker.task_process_url', bind=True)
def task_process_url(self, url: str):
    """
    Celery task for parallel URL processing, feature extraction, and prediction.
    This demonstrates the handling of loads and parallel computing.
    """
    print(f"[{self.request.id}] Processing URL: {url}")

    self.update_state(state='FEATURE_EXTRACTION', meta={'url': url})

    # 1. Feature Extraction (The heavy, unique part of the work)
    try:
        # Use the shared logic
        features_df = extract_all_features(url)
    except Exception as e:
        # Fail the task if feature extraction fails
        print(f"[{self.request.id}] ERROR during Feature Extraction: {e}")
        raise self.retry(exc=e, countdown=5, max_retries=3)  # Retry on failure

    self.update_state(state='PREDICTION', meta={'url': url})

    # 2. Prediction using the lightweight model
    if MODEL is None:
        # Dummy result if model is not loaded
        prediction = 1 if 'malware' in url else 0
        confidence = 0.95
    else:
        # Perform prediction (example for XGBoost/RF)
        prediction = MODEL.predict(features_df)[0]
        # Get the confidence level
        confidence = MODEL.predict_proba(features_df)[0][prediction]

    # 3. Return the result
    is_malicious = bool(prediction)

    return {
        'url': url,
        'is_malicious': is_malicious,
        'confidence': f"{confidence:.4f}",
        'model_type': MODEL.__class__.__name__ if MODEL else 'Dummy'
    }