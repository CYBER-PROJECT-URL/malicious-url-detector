"""
Feature Extraction Module (Aligned with Training Pipeline)
----------------------------------------------------------

This module ensures that inference-time feature extraction is
IDENTICAL to the training pipeline.

It provides:
1. URL preprocessing
2. Full feature extraction using the SAME function from training
3. Column alignment to selected_features.txt
4. Optional scaler/encoder loading (if needed later)
5. A ready-to-use inference feature vector
"""

import pandas as pd
import os

# Import the canonical feature extractor from the training script
from train_model_finish import extract_all_features_url


# -------------------------------------------------------------
# 1. Basic URL Preprocessing
# -------------------------------------------------------------
def preprocess_input_url(url: str) -> str:
    """
    Normalize user-provided URLs so they can be processed correctly.
    """
    if not isinstance(url, str):
        return ""

    url = url.strip()

    # Add default scheme if missing
    if "://" not in url:
        url = "http://" + url

    return url.lower()


# -------------------------------------------------------------
# 2. Extract raw features (same as during training)
# -------------------------------------------------------------
def extract_raw_features(url: str) -> pd.DataFrame:
    """
    Extract raw features using the SAME function used during training.
    Returns a single-row DataFrame.
    """
    clean_url = preprocess_input_url(url)

    # Call the same training feature extractor (this is the key!)
    features_dict = extract_all_features_url(clean_url, label=None)

    # Convert dict → DataFrame (single row)
    df = pd.DataFrame([features_dict])

    return df


# -------------------------------------------------------------
# 3. Load selected features list
# -------------------------------------------------------------
def load_selected_features(path: str = "selected_features.txt") -> list:
    """
    Load the selected feature names used during training.
    This ensures the inference data vector is aligned to the same order.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing selected_features.txt. Expected at: {path}"
        )

    with open(path, "r") as f:
        features = [line.strip() for line in f.readlines() if line.strip()]

    return features


# -------------------------------------------------------------
# 4. Align DataFrame to match training feature order
# -------------------------------------------------------------
def align_features_to_training(df: pd.DataFrame,
                               selected_features: list) -> pd.DataFrame:
    """
    Ensure the DataFrame contains exactly the columns used during training.
    Missing columns → filled with 0
    Extra columns → removed
    """
    df = df.copy()

    # Add missing columns with value 0
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0

    # Keep only the training columns, in correct order
    df = df[selected_features]

    return df


# -------------------------------------------------------------
# 5. Full pipeline: URL → model-ready feature vector
# -------------------------------------------------------------
def prepare_features_for_model(url: str,
                               selected_features_path="selected_features.txt") -> pd.DataFrame:
    """
    Complete inference pipeline:
    1. Preprocess URL
    2. Extract raw features (training-identical)
    3. Align with selected_features.txt
    """
    # Load the selected features names
    selected_features = load_selected_features(selected_features_path)

    # Extract raw unaligned features
    df_raw = extract_raw_features(url)

    # Align to training column order
    df_aligned = align_features_to_training(df_raw, selected_features)

    return df_aligned


# -------------------------------------------------------------
# Manual test
# -------------------------------------------------------------
if __name__ == "__main__":
    test_url = "https://example.com/login"
    df = prepare_features_for_model(test_url)
    print("\nModel-ready features:")
    print(df.head())
    print("\nTotal features:", len(df.columns))
