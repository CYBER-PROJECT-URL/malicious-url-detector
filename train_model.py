import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from urllib.parse import urlparse
import re

# --- 1. הגדרות וטעינת נתונים (יש להתאים את קובץ הנתונים) ---
# נדרש קובץ CSV עם עמודות 'url' (כתובת האתר) ו-'label' (0 - בטוח, 1 - זדוני)
DATASET_PATH = 'malicious_urls_dataset.csv'  # שנה לנתיב האמיתי של הקובץ שלך
RANDOM_SEED = 42

try:
    data = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"שגיאה: הקובץ {DATASET_PATH} לא נמצא. אנא ודא שהנתיב נכון.")
    exit()

# הסרת שורות עם ערכי NULL
data = data.dropna(subset=['url', 'label'])
print(f"סה\"כ דוגמאות לניתוח: {len(data)}")


# --- 2. מיצוי תכונות לקסיקליות/מבניות (כמו במאמר השני) ---

def extract_lexical_features(url):
    """מחלץ תכונות מבניות ולקסיקליות מכתובת האתר."""
    if not isinstance(url, str):
        return [0] * 9

    features = {}

    # 1. אורך URL כולל
    features['url_length'] = len(url)

    # 2. נוכחות IP בכתובת האתר
    features['has_ip'] = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0

    # 3. מספר סימני '@'
    features['at_sign'] = url.count('@')

    # 4. מספר סימני '-'
    features['hyphens'] = url.count('-')

    # 5. מספר סימני '/'
    features['slash'] = url.count('/')

    # 6. אורך ה-Hostname
    parsed = urlparse(url)
    features['hostname_length'] = len(parsed.netloc)

    # 7. עומק הנתיב (מספר הסאב-דומיינים)
    features['path_depth'] = url.count('//')

    # 8. נוכחות קיצור (למשל, 'bit.ly')
    features['is_shortened'] = 1 if ('bit.ly' in url or 'goo.gl' in url) else 0

    # 9. נוכחות HTTPS (אינדיקציה לא בהכרח בטוחה, אבל שימושית)
    features['has_https'] = 1 if url.startswith('https') else 0

    return list(features.values())


# החלת מיצוי התכונות
lexical_features = data['url'].apply(lambda x: pd.Series(extract_lexical_features(x)))
lexical_features.columns = [
    'url_length', 'has_ip', 'at_sign', 'hyphens', 'slash',
    'hostname_length', 'path_depth', 'is_shortened', 'has_https'
]
X_lexical = lexical_features.values
y = data['label'].values

print("תכונות לקסיקליות חולצו.")
#

# --- 3. הדמיית ייצוג סמנטי (Embedding - בהשראת PMANet) ---
# הערה: יצירת ייצוגי BERT בפועל דורשת התקנת ספריות כבדות (transformers, PyTorch/TensorFlow)
# וזמן חישוב משמעותי. לצורך הדגמה, אנו יוצרים ערכים רנדומליים המדמים ייצוגים אלו.
# בשימוש אמיתי, יש להחליף את הקוד הזה בייצוג אמיתי.

EMBEDDING_DIM = 768  # ממד הייצוג של BERT
NUM_SAMPLES = len(data)

# הדמיה של הייצוג הסמנטי (יש להחליף ב-BERT Embeddings אמיתי!)
X_semantic = np.random.rand(NUM_SAMPLES, EMBEDDING_DIM)
print(f"תכונות סמנטיות (הדמיה): {X_semantic.shape}")

# --- 4. פיצול נתונים ---
X_lex_train, X_lex_test, y_train, y_test = train_test_split(
    X_lexical, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)
X_sem_train, X_sem_test, _, _ = train_test_split(
    X_semantic, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

# --- 5. אימון מודל קלאסי (Random Forest - כמו במאמר השני) ---
print("\n--- אימון Random Forest ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
rf_model.fit(X_lex_train, y_train)

# --- 6. אימון מודל רשת נוירונים (Deep Learning - בהשראת PMANet) ---
# מודל רשת נוירונים פשוט (במקום רשת הקשב המורכבת של PMANet)
print("\n--- אימון רשת נוירונים (Deep Learning) ---")

nn_model = Sequential([
    Dense(512, activation='relu', input_shape=(EMBEDDING_DIM,)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # יציאה בינארית
])

nn_model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

# אימון הרשת הנוירונית על התכונות הסמנטיות
nn_model.fit(X_sem_train, y_train, epochs=5, batch_size=32, verbose=0)
print("אימון רשת נוירונים הסתיים.")

# --- 7. שילוב מודלים (Ensemble - שיפור משולב) ---
print("\n--- שילוב מודלים (Voting Classifier) ---")

# יצירת תחזיות הסתברות
rf_proba = rf_model.predict_proba(X_lex_test)[:, 1]
nn_proba = nn_model.predict(X_sem_test).flatten()

# שילוב התחזיות באמצעות ממוצע משוקלל (ניתן להתאים משקלים)
WEIGHT_RF = 0.4
WEIGHT_NN = 0.6
ensemble_proba = (rf_proba * WEIGHT_RF) + (nn_proba * WEIGHT_NN)
ensemble_predictions = (ensemble_proba > 0.5).astype(int)


# --- 8. הערכת ביצועים ---

## 📊 הערכת ביצועים
def evaluate_model(y_true, y_pred, model_name):
    """מדפיס מדדי ביצועים."""
    print(f"\n### {model_name} ###")
    print(f"דיוק (Accuracy): {accuracy_score(y_true, y_pred):.4f}")
    print(f"דיוק חיובי (Precision): {precision_score(y_true, y_pred):.4f}")
    print(f"כיסוי (Recall): {recall_score(y_true, y_pred):.4f}")
    print(f"ציון F1: {f1_score(y_true, y_pred):.4f}")


# הערכת Random Forest (תכונות לקסיקליות)
rf_predictions = rf_model.predict(X_lex_test)
evaluate_model(y_test, rf_predictions, "Random Forest (לקסיקלי)")

# הערכת רשת נוירונים (תכונות סמנטיות)
nn_predictions = (nn_model.predict(X_sem_test).flatten() > 0.5).astype(int)
evaluate_model(y_test, nn_predictions, "Neural Network (סמנטי)")

# הערכת המודל המשולב (היברידי) - הגישה המשופרת
evaluate_model(y_test, ensemble_predictions, "מודל משולב (היברידי)")

# --- 9. שמירת המודלים (Random Forest ושמירת משקולות הרשת הנוירונית) ---
import joblib

# שמירת מודל Random Forest
JOB_LIB_PATH = 'random_forest_model.joblib'
joblib.dump(rf_model, JOB_LIB_PATH)
print(f"\nמודל Random Forest נשמר ב: {JOB_LIB_PATH}")

# שמירת משקולות הרשת הנוירונית
H5_PATH = 'neural_network_weights.h5'
nn_model.save_weights(H5_PATH)
print(f"משקולות הרשת הנוירונית נשמרו ב: {H5_PATH}")

print("\n✅ סיום האימון. ניתן להשתמש במודלים שנשמרו לקבלת תחזיות.")

# --- סוף הקוד ---