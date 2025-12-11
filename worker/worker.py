# worker/worker.py

from celery import Celery
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

from backend.models.feature_extractor import extract_all_features_url

# ---------- 1. קבועים ונתיבי Assets ----------

# חשוב: ב-docker-compose למפות:
# - ./models:/app/assets
# - ./backend:/app/backend
ASSET_PATH = os.path.join(os.getcwd(), 'assets')
FAILURE_CODE = -1

CATEGORICAL_COLS = [
    'Server_Country_IPAPI',
    'Server_Country_Code',
    'WHOIS_Country',
    'WHOIS_Registrar',
    'Domain',
    'IP_Type'
]

# אם שמרת גם רשימה של numeric_feature_cols באימון – אפשר לטעון מכאן.
# כרגע נניח שכל מה שלא ב-CATEGORICAL_COLS הוא numeric.


# ---------- 2. טעינת נכסי המודל ----------

MODEL = None
ENCODER = None
SCALER = None
SELECTED_FEATURES = None
SELECTED_NO_LABEL = None
ASSETS_LOADED = False

try:
    MODEL = joblib.load(os.path.join(ASSET_PATH, "best_model_xgb.pkl"))
    ENCODER = joblib.load(os.path.join(ASSET_PATH, "categorical_encoder.pkl"))
    SCALER = joblib.load(os.path.join(ASSET_PATH, "feature_scaler.pkl"))

    with open(os.path.join(ASSET_PATH, "selected_features.txt"), 'r') as f:
        SELECTED_FEATURES = [line.strip() for line in f]

    SELECTED_NO_LABEL = [f for f in SELECTED_FEATURES if f != 'Label']

    print("✅ Assets נטענו בהצלחה מהתיקייה:", ASSET_PATH)
    print("   - מודל:", type(MODEL))
    print("   - מספר פיצ'רים נבחרים (ללא Label):", len(SELECTED_NO_LABEL))
    ASSETS_LOADED = True
except Exception as e:
    print(f"❌ שגיאה קריטית בטעינת Assets מתוך {ASSET_PATH}: {e}")
    ASSETS_LOADED = False


# ---------- 3. Celery app ----------

app = Celery('tasks', broker='redis://redis:6379/0')


# ---------- 4. עיבוד פיצ'רים + חיזוי ----------

def preprocess_and_predict(feats: dict):
    """
    מקבל dict מפונקציית extract_all_features_url,
    מבצע:
    - השלמת פיצ'רים חסרים
    - קידוד קטגוריאלי (OrdinalEncoder)
    - scaling לפיצ'רים נומריים
    - חיזוי בעזרת המודל
    """
    if not ASSETS_LOADED:
        return 0, 0.0

    # הופך לשורה אחת של DataFrame
    df = pd.DataFrame([feats])

    # עמודות שלא נדרשות למודל (Label ו-IP)
    df = df.drop(columns=['Resolved_IP', 'url_original', 'Label'], errors='ignore')

    # 1. לוודא שכל הפיצ'רים מהאימון קיימים
    #    פיצ'רים שנחסרו – יקבלו FAILURE_CODE (יוחלף אח"כ).
    for col in SELECTED_NO_LABEL:
        if col not in df.columns:
            df[col] = FAILURE_CODE

    # 2. זיהוי/הוספת עמודות קטגוריאליות חסרות
    for c in CATEGORICAL_COLS:
        if c not in df.columns:
            df[c] = "Unknown"

    # 3. קידוד קטגוריאלי (OrdinalEncoder עם handle_unknown='use_encoded_value', unknown_value=-1 באימון)
    cols_present = [c for c in CATEGORICAL_COLS if c in df.columns]
    if cols_present and ENCODER is not None:
        try:
            df[cols_present] = ENCODER.transform(df[cols_present])
        except Exception as e:
            print(f"⚠ שגיאה ב-ENCODER.transform: {e}")
            # במקרה כשל, ננסה להחליף בערכי unknown
            for c in cols_present:
                df[c] = -1

    # 4. החלפה ראשונית של FAILURE_CODE לערך 0 (הסקיילר אומן אחרי טיפול ב-Failure באימון)
    df.replace(FAILURE_CODE, 0, inplace=True)
    df.fillna(0, inplace=True)

    # 5. בחירת סדר הפיצ'רים לפי selected_features.txt
    #    המודל אומן על סדר זה, ולכן חייבים אותו כאן.
    X = df[SELECTED_NO_LABEL]

    # 6. זיהוי פיצ'רים נומריים – כאן נזהר שלא לשנות את העמודות הקטגוריאליות לאחר הקידוד
    numeric_feature_cols = [c for c in X.columns if c not in CATEGORICAL_COLS]

    # 7. Scaling לפיצ'רים נומריים בלבד
    if numeric_feature_cols and SCALER is not None:
        try:
            X[numeric_feature_cols] = SCALER.transform(X[numeric_feature_cols])
        except Exception as e:
            print(f"⚠ שגיאה ב-SCALER.transform: {e}")

    # 8. חיזוי
    try:
        prediction = MODEL.predict(X)[0]
        if hasattr(MODEL, "predict_proba"):
            score = MODEL.predict_proba(X)[:, 1][0]
        else:
            score = float(prediction)
    except Exception as e:
        print(f"❌ שגיאה בחיזוי המודל: {e}")
        return 0, 0.0

    return prediction, score


# ---------- 5. משימת Celery ----------

@app.task(name='analyze_url_for_malware')
def analyze_url_for_malware(url: str):
    """
    משימת Celery המקבלת URL ומחזירה ניבוי.
    1. חילוץ פיצ'רים (extract_all_features_url)
    2. עיבוד (encoder + scaler)
    3. חיזוי
    """
    if not ASSETS_LOADED:
        return {
            "url": url,
            "is_malicious": False,
            "score": 0.0,
            "status": "error",
            "error": "Model assets not loaded",
            "timestamp": datetime.now().isoformat()
        }

    # 1. חילוץ פיצ'רים – בדיוק כמו באימון
    feats = extract_all_features_url(url, label=None)

    # 2. עיבוד וחיזוי
    prediction, score = preprocess_and_predict(feats)

    return {
        "url": url,
        "is_malicious": bool(prediction),
        "score": float(score),
        "status": "completed",
        "timestamp": datetime.now().isoformat()
    }
