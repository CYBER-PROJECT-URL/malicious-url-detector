import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pickle
import sys
import os

# Set the path to the feature extractor logic (assuming it's relative)
# This is necessary to use the same feature extraction logic for consistency
sys.path.append('./backend')
from feature_extractor import extract_all_features

# Define the location where the model will be saved
MODEL_DIR = './backend/models'
MODEL_FILENAME = 'ml_model.pkl'
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)


def load_and_preprocess_data(file_path):
    """
    Load the prepared dataset and simulate feature extraction if necessary.
    In a real project, this dataset should already contain the rich features
    (lexical, host-based, structural) that were engineered.
    """
    print("INFO: Loading dataset...")
    try:
        # Load your merged and engineered dataset (e.g., CSV)
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("ERROR: Dataset file not found. Creating dummy data.")
        # --- Create Dummy Data for illustration if file is missing ---
        data = {
            'url_length': np.random.randint(20, 150, 500),
            'num_dots': np.random.randint(1, 5, 500),
            'is_ip_address': np.random.randint(0, 2, 500),
            'has_https': np.random.randint(0, 2, 500),
            # Target variable (0: benign, 1: malicious)
            'label': np.random.randint(0, 2, 500)
        }
        df = pd.DataFrame(data)
        # -----------------------------------------------------------

    # Separate features (X) from the target (y)
    X = df.drop('label', axis=1)
    y = df['label']

    # Ensure the feature names match those expected by the feature_extractor.py
    # (Important for deployment consistency)

    return X, y


def train_and_save_model(X, y):
    """
    Trains the XGBoost model, evaluates performance, and saves the model
    as ml_model.pkl.
    """
    # 1. Split data (80% train, 20% test) [cite: 92]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"INFO: Training data size: {X_train.shape}")

    # 2. Initialize and Train the Model (XGBoost is chosen for high performance)
    # NOTE: Hyperparameters should be optimized using cross-validation (Grid Search) [cite: 93]
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    print("INFO: Starting model training...")
    model.fit(X_train, y_train)
    print("INFO: Training complete.")

    # 3. Evaluate the Model (Crucial for the Final Report) [cite: 149-152]
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Evaluate the key metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("-" * 30)
    print("Model Evaluation (Testing Set):")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("-" * 30)

    # 4. Save the Model as .pkl
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)

    print(f"SUCCESS: Model saved to {MODEL_PATH}")

    # The evaluation results (acc, f1, roc_auc) are what you use to compare
    # against Rathod et al. (92.5% accuracy) and justify your system's performance.


if __name__ == '__main__':
    # REPLACE 'your_prepared_dataset.csv' with the actual path to your merged/engineered dataset
    DATASET_FILE = 'your_prepared_dataset.csv'

    X, y = load_and_preprocess_data(DATASET_FILE)
    train_and_save_model(X, y)