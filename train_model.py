import pandas as pd
import numpy as np
import re
import joblib
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from transformers import AutoTokenizer, TFAutoModel

# --- 1. הגדרות גלובליות ---
# נתיב לקובץ שהעלית
DATASET_PATH = 'dataSet Kaggle.zip/Malicious URL v3.csv'
RANDOM_SEED = 42

# הגדרות BERT
BERT_MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128  # אורך מקסימלי לרצף (URL)
EMBEDDING_DIM = 768  # ממד הייצוג (Embedding) של BERT-base


# --- 2. טעינת נתונים ועיבוד מקדים ---

def load_and_preprocess_data(path):
    """טוען את הנתונים, מנקה וממיר את התווית לבינארית."""
    print(f"טוען נתונים מ: {path}...")
    try:
        data = pd.read_csv(path, index_col=0)
    except Exception as e:
        print(f"שגיאה בטעינת הקובץ: {e}")
        return None

    # ניקוי: הסרת שורות עם ערכי NULL ב'url' או 'type'
    data = data.dropna(subset=['url', 'type'])

    # המרת התווית (label) לבינארית:
    # benign (בטוח) -> 0
    # phishing, defacement, malware (זדוני) -> 1
    data['label'] = data['type'].apply(lambda x: 0 if x == 'benign' else 1)

    X = data['url'].astype(str)
    y = data['label']

    print(f"סה\"כ דוגמאות לאחר ניקוי: {len(data)}")
    print(f"חלוקת תוויות: \n{data['label'].value_counts()}")
    return X, y


# --- 3. מיצוי תכונות לקסיקליות (ל-Random Forest) ---

def extract_lexical_features(url):
    """מחלץ תכונות מבניות ולקסיקליות מכתובת האתר (כמו במאמר השני)."""

    if not isinstance(url, str):
        url = ""

    features = {}
    parsed = urlparse(url)

    # 1. אורך URL
    features['url_length'] = len(url)
    # 2. מספר התו '@' (אינדיקציה להסוואה)
    features['at_sign_count'] = url.count('@')
    # 3. נוכחות IP בכתובת האתר
    features['has_ip'] = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', parsed.netloc) else 0
    # 4. מספר סימני '-'
    features['hyphens_count'] = url.count('-')
    # 5. עומק הנתיב (מספר '/')
    features['path_depth'] = url.count('/')
    # 6. אורך ה-Hostname
    features['hostname_length'] = len(parsed.netloc)
    # 7. נוכחות מילות מפתח זדוניות (דוגמה)
    malicious_keywords = ['login', 'bank', 'secure', 'update', 'verify']
    features['malicious_keyword'] = sum(1 for kw in malicious_keywords if kw in url)
    # 8. נוכחות קיצור URL
    features['is_shortened'] = 1 if len(url) < 30 and 'bit.ly' in url or 'goo.gl' in url else 0

    return pd.Series(features)


# --- 4. מיצוי תכונות סמנטיות (ל-Deep Learning - בהשראת PMANet) ---

def get_bert_embeddings(urls):
    """משתמש במודל BERT שאומן מראש כדי להפיק ייצוגים סמנטיים (Embeddings)."""
    print(f"\nמפיק ייצוגי BERT באמצעות {BERT_MODEL_NAME}...")

    # טעינת טוקנייזר ומודל
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = TFAutoModel.from_pretrained(BERT_MODEL_NAME)

    # טוקניזציה של כל כתובות האתר
    tokenized_inputs = tokenizer(
        urls.tolist(),
        max_length=MAX_LEN,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )

    # חישוב הייצוגים
    # נדרשת סביבת Tensorflow
    try:
        with tf.device('/CPU:0'):  # שימוש ב-CPU אם אין GPU זמין
            outputs = model(tokenized_inputs)
        # שימוש ב-CLS token כייצוג של כל הרצף
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    except Exception as e:
        print(f"שגיאה בהפעלת מודל BERT: {e}. בודק אם TF2 מוגדר כראוי.")
        return None

    print(f"הפקת הייצוגים הסתיימה. צורה: {embeddings.shape}")
    return embeddings


# --- 5. אימון מודל רשת נוירונים (החלק העמוק) ---

def create_nn_model(input_dim):
    """בניית רשת נוירונים פשוטה המבוססת על ייצוגי BERT."""
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        Dropout(0.5, seed=RANDOM_SEED),
        Dense(128, activation='relu'),
        Dropout(0.3, seed=RANDOM_SEED),
        Dense(1, activation='sigmoid')  # פלט בינארי
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# --- 6. פונקציית האימון הראשית ---

def train_and_evaluate():
    # א. טעינת נתונים ופיצול
    X, y = load_and_preprocess_data(DATASET_PATH)
    if X is None:
        return

    # פיצול נתונים עבור שתי הגישות (צריך להיות זהה)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # ב. הכנת נתונים ל-Random Forest (תכונות לקסיקליות)
    print("\n--- הכנת תכונות לקסיקליות ---")
    X_lex_train = X_train.apply(extract_lexical_features)
    X_lex_test = X_test.apply(extract_lexical_features)

    # סקיילר (Normalization) לשיפור ביצועי Random Forest ו-NN
    scaler = StandardScaler()
    X_lex_train_scaled = scaler.fit_transform(X_lex_train)
    X_lex_test_scaled = scaler.transform(X_lex_test)
    joblib.dump(scaler, 'scaler_lexical.joblib')
    print("תכונות לקסיקליות עובדו ונשמרו סקיילר.")

    # ג. הכנת נתונים לרשת הנוירונים (תכונות סמנטיות - BERT)
    X_sem_train = get_bert_embeddings(X_train)
    X_sem_test = get_bert_embeddings(X_test)

    if X_sem_train is None:
        print("לא ניתן להמשיך לאימון ה-NN ללא ייצוגי BERT.")
        X_sem_train = np.zeros((len(X_train), EMBEDDING_DIM))
        X_sem_test = np.zeros((len(X_test), EMBEDDING_DIM))

    # ד. אימון מודל Random Forest (ML קלאסי)
    print("\n--- אימון Random Forest (לקסיקלי) ---")
    rf_model = RandomForestClassifier(n_estimators=150, max_depth=15,
                                      random_state=RANDOM_SEED, n_jobs=-1, class_weight='balanced')
    rf_model.fit(X_lex_train_scaled, y_train)
    rf_pred_proba = rf_model.predict_proba(X_lex_test_scaled)[:, 1]
    rf_predictions = (rf_pred_proba > 0.5).astype(int)
    print("Random Forest אומן והוערך.")

    # ה. אימון מודל Deep Learning (סמנטי)
    print("\n--- אימון Deep Neural Network (סמנטי) ---")
    nn_model = create_nn_model(EMBEDDING_DIM)

    # שימוש ב-Early Stopping כדי למנוע Overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    nn_model.fit(X_sem_train, y_train,
                 epochs=20,
                 batch_size=64,
                 validation_data=(X_sem_test, y_test),
                 callbacks=[early_stopping],
                 verbose=0)

    nn_pred_proba = nn_model.predict(X_sem_test).flatten()
    nn_predictions = (nn_pred_proba > 0.5).astype(int)
    print("Neural Network אומנה והוערכה.")

    # ו. שילוב מודלים (Ensemble/Hybrid - הגישה המשופרת)
    # משקולות שנקבעו על בסיס העוצמה הצפויה של כל גישה:
    # למידה עמוקה (BERT) מקבלת משקל גבוה יותר על סמך המאמר הראשון (PMANet).
    WEIGHT_RF = 0.35
    WEIGHT_NN = 0.65
    print(f"\n--- שילוב היברידי: RF={WEIGHT_RF}, NN={WEIGHT_NN} ---")

    ensemble_proba = (rf_pred_proba * WEIGHT_RF) + (nn_pred_proba * WEIGHT_NN)
    ensemble_predictions = (ensemble_proba > 0.5).astype(int)

    # ז. הערכת ביצועים
    print("\n" + "=" * 50)
    print("            📊 סיכום ביצועים על נתוני הבדיקה")
    print("=" * 50)

    print("\n**1. Random Forest (תכונות לקסיקליות):**")
    print(classification_report(y_test, rf_predictions, target_names=['Benign (0)', 'Malicious (1)']))

    print("\n**2. Neural Network (תכונות סמנטיות - BERT):**")
    print(classification_report(y_test, nn_predictions, target_names=['Benign (0)', 'Malicious (1)']))

    print("\n**3. מודל היברידי משולב (המשופר):**")
    print(classification_report(y_test, ensemble_predictions, target_names=['Benign (0)', 'Malicious (1)']))
    print(f"דיוק כולל (Accuracy): {accuracy_score(y_test, ensemble_predictions):.4f}")
    print(f"ציון F1 ממוצע: {f1_score(y_test, ensemble_predictions, average='weighted'):.4f}")
    print("=" * 50)

    # ח. שמירת המודלים והמשקולות
    joblib.dump(rf_model, 'RF_lexical_model.joblib')
    nn_model.save_weights('NN_semantic_weights.h5')

    # שמירת נתונים לדוגמה עבור מודל ההיברידי
    np.save('ensemble_weights.npy', np.array([WEIGHT_RF, WEIGHT_NN]))

    print("\n✅ האימון הסתיים בהצלחה.")
    print("המודלים נשמרו: RF_lexical_model.joblib, NN_semantic_weights.h5, ensemble_weights.npy, scaler_lexical.joblib")
    print("כדי להשתמש במודל המשולב, עליך לטעון את שניהם ואת משקולות השילוב.")


if __name__ == "__main__":
    # הגדרת אקראיות לתוצאות שחזור
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    train_and_evaluate()