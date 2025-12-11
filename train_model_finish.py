import pandas as pd
import numpy as np
import whois
from urllib.parse import urlparse
from datetime import datetime
import time
import re
import os
import socket
import ipapi
import tldextract
import requests
import dns.resolver
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import json

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import joblib

# ---------- קבועים ----------

RAW_DATA_PATH = 'DataSet MALICIOS URLs/DataSet MALICIOS URLs/CSV WITH MAL URLS/dataSet Kaggle.zip'
ENRICHED_CSV_FILE = 'dataset_full_latex_pipeline.csv'
BEST_MODEL_PATH = 'best_model_xgb.pkl'
ENCODER_PATH = 'categorical_encoder.pkl'
SCALER_PATH = 'feature_scaler.pkl'
FEATURE_LIST_PATH = 'selected_features.txt'
MODEL_META_PATH = 'model_metadata.json'

API_DELAY_SECONDS = 0.0
FAILURE_CODE = -1

URL_SHORTENERS = {
    'bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'ow.ly', 'buff.ly',
    'is.gd', 'cutt.ly', 'shorturl.at', 'rebrand.ly'
}

CATEGORICAL_COLS = [
    'Server_Country_IPAPI',
    'Server_Country_Code',
    'WHOIS_Country',
    'WHOIS_Registrar',
    'Domain',
    'IP_Type'
]

MANDATORY_FEATURES = [
    "Domain_Age_Days",
    "DNS_A_Record",
    "DNS_NS_Record",
    "DNS_MX_Record",
    "First_Seen_SSL_Days",
    "Server_Safety_Score",
    "entropy",
    "External_Failure_Count"
]

socket.setdefaulttimeout(2)


# ---------- Retry decorator עם backoff בסיסי ----------

def retry(times=3, delay=1, backoff=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except requests.HTTPError as e:
                    # Rate limit / 5xx handling בסיסי
                    status = getattr(e.response, "status_code", None)
                    if status in (429, 500, 502, 503, 504) and attempt < times - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
                        continue
                    return None
                except Exception:
                    if attempt < times - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        return None
        return wrapper
    return decorator


# ---------- Helpers ----------

def get_domain_from_url(url):
    if not isinstance(url, str):
        return ""
    try:
        url = url.strip()
        if "://" not in url:
            url = "http://" + url
        extracted = tldextract.extract(url)
        if extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}".lower()
        return ""
    except:
        return ""


def shannon_entropy(data):
    if not data:
        return 0
    counts = pd.Series(list(data)).value_counts()
    probabilities = counts / len(data)
    entropy = - (probabilities * np.log2(probabilities)).sum()
    return entropy


def get_ip_type(ip):
    try:
        socket.inet_pton(socket.AF_INET, ip)
        return "IPv4"
    except OSError:
        try:
            socket.inet_pton(socket.AF_INET6, ip)
            return "IPv6"
        except OSError:
            return "Unknown"


# ---------- Cached external calls + timeouts ----------

@lru_cache(maxsize=50000)
@retry(times=3, delay=1.5)
def get_whois_features(domain):
    if not domain or domain.count('.') < 1:
        return 0, 0, "Unknown", "Unknown"
    try:
        w = whois.whois(domain, timeout=4)
    except Exception:
        return FAILURE_CODE, 0, "Unknown", "Unknown"

    dns_available = 1 if getattr(w, "domain_name", None) else 0

    age_days = 0
    try:
        cdate = getattr(w, "creation_date", None)
        if isinstance(cdate, list) and cdate:
            cdate = cdate[0]
        if isinstance(cdate, datetime):
            age_days = (datetime.now() - cdate).days
        else:
            age_days = FAILURE_CODE
    except Exception:
        age_days = FAILURE_CODE

    country = "Unknown"
    try:
        c = getattr(w, "country", None)
        if c:
            if isinstance(c, list) and c:
                country = str(c[0])
            else:
                country = str(c)
    except Exception:
        country = "Unknown"

    registrar = "Unknown"
    try:
        r = getattr(w, "registrar", None)
        if r:
            registrar = str(r)
        else:
            org = getattr(w, "org", None)
            if org:
                registrar = str(org)
    except Exception:
        registrar = "Unknown"

    return age_days, dns_available, country, registrar


@lru_cache(maxsize=50000)
@retry(times=3, delay=1)
def get_dns_validation(domain):
    if not domain or domain.count('.') < 1:
        return 0, 0, 0
    a_record, ns_record, mx_record = 0, 0, 0
    try:
        if dns.resolver.resolve(domain, 'A', lifetime=2):
            a_record = 1
    except Exception:
        pass
    try:
        if dns.resolver.resolve(domain, 'NS', lifetime=2):
            ns_record = 1
    except Exception:
        pass
    try:
        if dns.resolver.resolve(domain, 'MX', lifetime=2):
            mx_record = 1
    except Exception:
        pass
    return a_record, ns_record, mx_record


@lru_cache(maxsize=50000)
@retry(times=3, delay=1)
def get_ip_from_domain(domain):
    try:
        return socket.gethostbyname(domain)
    except Exception:
        return None


@retry(times=3, delay=1)
def get_geolocation_features(ip):
    if not ip or ip in ['127.0.0.1', '0.0.0.0']:
        return "Not Found", 0, "Unknown"
    try:
        data = ipapi.location(ip, timeout=3)
        server_country = data.get('country_name', 'Not Found')
        server_country_code = data.get('country', 'Unknown')
        is_safe = 1 if server_country in ['United States', 'Canada', 'United Kingdom', 'Germany', 'France', 'Israel'] else 0
        return server_country, is_safe, server_country_code
    except Exception:
        return "API Error", FAILURE_CODE, "Unknown"


@lru_cache(maxsize=50000)
@retry(times=3, delay=1)
def get_history_features(domain):
    first_seen_ssl = 0
    first_seen_archive = 0

    try:
        archive_url = f"https://archive.org/wayback/available?url={domain}"
        r = requests.get(archive_url, timeout=3)
        if 'application/json' in r.headers.get('Content-Type', ''):
            response = r.json()
        else:
            response = {}
        if response.get('archived_snapshots') and response['archived_snapshots'].get('closest'):
            timestamp = response['archived_snapshots']['closest']['timestamp']
            first_seen_archive = (datetime.now() - datetime.strptime(timestamp[:8], "%Y%m%d")).days
        else:
            first_seen_archive = FAILURE_CODE
    except Exception:
        first_seen_archive = FAILURE_CODE

    try:
        crtsh_url = f"https://crt.sh/?q={domain}&output=json"
        r = requests.get(crtsh_url, timeout=3)
        if 'application/json' in r.headers.get('Content-Type', ''):
            data = r.json()
        else:
            data = []
        if data and isinstance(data, list):
            earliest_date = min([
                datetime.strptime(item.get('not_before', '2099-01-01T00:00:00')[:10], "%Y-%m-%d")
                for item in data if item.get('not_before')
            ])
            if earliest_date:
                first_seen_ssl = (datetime.now() - earliest_date).days
        else:
            first_seen_ssl = FAILURE_CODE
    except Exception:
        first_seen_ssl = FAILURE_CODE

    return first_seen_ssl, first_seen_archive


@retry(times=3, delay=1)
def get_url_availability(url):
    try:
        response = requests.head(url, timeout=3, allow_redirects=True)
        is_available = 1 if 200 <= response.status_code < 400 else 0
        return is_available, response.status_code
    except Exception:
        return 0, 999


# ---------- Lexical for URL ----------

def extract_lexical_from_url(url):
    parsed = urlparse(url)
    ext = tldextract.extract(url)

    features = {}
    features['url_original'] = url

    features['url_length'] = len(url)
    features['domain_length'] = len(ext.domain)
    features['path_length'] = len(parsed.path)
    features['entropy'] = shannon_entropy(url)
    features['domain_to_path_ratio'] = features['domain_length'] / (features['path_length'] + 1e-6)

    features['num_dots'] = url.count('.')
    features['num_at_sign'] = url.count('@')
    features['num_hyphens'] = url.count('-')
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['has_https'] = 1 if parsed.scheme == 'https' else 0

    features['suspicious_keywords'] = 1 if re.search(
        r'login|verify|secure|account|update|confirm|pay|bank|webscr|token',
        url, re.IGNORECASE
    ) else 0

    subdomain = ext.subdomain or ""
    features['subdomain_depth'] = subdomain.count('.') + (1 if subdomain else 0)

    special_chars = sum(not c.isalnum() for c in url)
    features['special_char_ratio'] = special_chars / (len(url) + 1e-6)

    path_tokens = [p for p in parsed.path.split('/') if p]
    features['path_token_count'] = len(path_tokens)

    netloc = parsed.netloc.lower()
    features['is_shortened'] = 1 if netloc in URL_SHORTENERS else 0

    features['Domain'] = get_domain_from_url(url)

    return features


# ---------- Unified pipeline for single URL ----------

def extract_all_features_url(url, label=None):
    url = url.strip()
    feats = extract_lexical_from_url(url)
    domain = feats['Domain']

    age_dns_country_reg = get_whois_features(domain)
    if age_dns_country_reg is not None:
        age, dns_avail, whois_country, whois_registrar = age_dns_country_reg
    else:
        age, dns_avail, whois_country, whois_registrar = FAILURE_CODE, 0, "Unknown", "Unknown"

    feats["Domain_Age_Days"] = age
    feats["DNS_Record_Available"] = dns_avail
    feats["WHOIS_Country"] = whois_country
    feats["WHOIS_Registrar"] = whois_registrar

    dns_recs = get_dns_validation(domain)
    a_record, ns_record, mx_record = (
        dns_recs if dns_recs is not None else (FAILURE_CODE, FAILURE_CODE, FAILURE_CODE)
    )
    feats["DNS_A_Record"] = a_record
    feats["DNS_NS_Record"] = ns_record
    feats["DNS_MX_Record"] = mx_record

    history_recs = get_history_features(domain)
    ssl_days, archive_days = (
        history_recs if history_recs is not None else (FAILURE_CODE, FAILURE_CODE)
    )
    feats["First_Seen_SSL_Days"] = ssl_days
    feats["First_Seen_Archive_Days"] = archive_days

    avail_status = get_url_availability(url)
    is_available, status_code = (
        avail_status if avail_status is not None else (0, 999)
    )
    feats["URL_Available"] = is_available
    feats["HTTP_Status_Code"] = status_code

    ip = get_ip_from_domain(domain)
    ip = ip if ip is not None else ""
    ip_type = get_ip_type(ip) if ip else "Unknown"

    server_country, safe_score, server_country_code = (
        get_geolocation_features(ip) if ip else ("Not Found", 0, "Unknown")
    )
    feats["Resolved_IP"] = ip if ip else "None"
    feats["IP_Type"] = ip_type
    feats["Server_Country_IPAPI"] = server_country
    feats["Server_Country_Code"] = server_country_code
    feats["Server_Safety_Score"] = safe_score

    failure_signals = [
        ('DNS_A_Record', feats['DNS_A_Record']),
        ('DNS_NS_Record', feats['DNS_NS_Record']),
        ('DNS_MX_Record', feats['DNS_MX_Record']),
        ('DNS_Record_Available', feats['DNS_Record_Available']),
        ('URL_Available', feats['URL_Available']),
        ('HTTP_Status_Code', feats['HTTP_Status_Code']),
        ('First_Seen_SSL_Days', feats['First_Seen_SSL_Days']),
        ('First_Seen_Archive_Days', feats['First_Seen_Archive_Days']),
        ('Server_Safety_Score', feats['Server_Safety_Score'])
    ]

    external_failure_count = 0
    for name, val in failure_signals:
        if name == "HTTP_Status_Code":
            if val == 0 or val == FAILURE_CODE or val >= 400:
                external_failure_count += 1
        elif val == 0 or val == FAILURE_CODE:
            external_failure_count += 1

    feats["External_Failure_Count"] = external_failure_count

    if label is not None:
        feats['Label'] = label

    return feats


# ---------- Parallel dataset build ----------

def process_row_for_parallel(row):
    url = row['url_original']
    label = row['Label']
    feats = extract_all_features_url(url, label=label)

    clean_feats = {}
    for k, v in feats.items():
        if isinstance(v, (int, float, np.number)):
            try:
                if pd.isna(v) or np.isinf(v):
                    clean_feats[k] = FAILURE_CODE
                else:
                    clean_feats[k] = v
            except TypeError:
                clean_feats[k] = v
        else:
            clean_feats[k] = v
    return clean_feats


def build_enriched_dataset_parallel(raw_data_path, max_workers=None):
    try:
        df_raw = pd.read_csv(raw_data_path, compression='zip')
        malicious_types = ['phishing', 'malware', 'defacement']
        df_raw['Label'] = df_raw['type'].apply(lambda x: 1.0 if x in malicious_types else 0.0)
        df_raw = df_raw.rename(columns={'url': 'url_original'})
    except Exception as e:
        print(f"שגיאה בטעינת הדאטה: {e}")
        return None

    if max_workers is None:
        max_workers = min(16, (multiprocessing.cpu_count() or 4) * 2)

    rows = []
    print(f"🚀 מתחיל pipeline מקבילי על כל ה-URLs... (max_workers={max_workers})")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_row_for_parallel, row): idx for idx, row in df_raw.iterrows()}
        for i, fut in enumerate(as_completed(futures)):
            try:
                f = fut.result()
                rows.append(f)
            except Exception:
                continue
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}")

    df_enriched = pd.DataFrame(rows)
    df_enriched.to_csv(ENRICHED_CSV_FILE, index=False)
    print(f"\n✅ הסתיים pipeline. נשמר: {ENRICHED_CSV_FILE}")
    print(f"Shape: {df_enriched.shape}")
    return df_enriched


# ---------- Normalization for FAILURE_CODE ----------

def replace_failure_codes(df):
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        if col == 'Label':
            continue
        col_series = df_clean[col]
        mask_fail = (col_series == FAILURE_CODE)
        if not mask_fail.any():
            continue

        non_fail = col_series[~mask_fail]
        if non_fail.empty:
            df_clean.loc[mask_fail, col] = 0
        else:
            median_val = non_fail.median()
            df_clean.loc[mask_fail, col] = median_val

    return df_clean


# ---------- Feature selection + encoding + scaling ----------

def feature_selection_encode_scale(df, fit=True, encoder=None, scaler=None):
    df_fs = df.copy()
    df_fs = df_fs.drop(columns=['Resolved_IP', 'url_original'], errors='ignore')

    df_fs = replace_failure_codes(df_fs)

    cols_present = [c for c in CATEGORICAL_COLS if c in df_fs.columns]

    if fit:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        if cols_present:
            df_fs[cols_present] = enc.fit_transform(df_fs[cols_present])
    else:
        enc = encoder
        if cols_present:
            df_fs[cols_present] = enc.transform(df_fs[cols_present])

    y = df_fs['Label']
    X = df_fs.drop(columns=['Label'])

    corr = X.corrwith(y).abs()
    selected_corr = corr[corr > 0.05].index.tolist()

    mi = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi, index=X.columns)
    selected_mi = mi_series[mi_series > 0.0].index.tolist()

    selected_features = sorted(list(set(selected_corr) | set(selected_mi) | set(MANDATORY_FEATURES)))
    selected_features = [f for f in selected_features if f in X.columns]
    selected_features.append('Label')

    df_fs = df_fs[selected_features]

    numeric_cols = [c for c in df_fs.columns if c != 'Label' and c not in cols_present]

    if fit:
        sc = MinMaxScaler()
        if numeric_cols:
            df_fs[numeric_cols] = sc.fit_transform(df_fs[numeric_cols])
    else:
        sc = scaler
        if numeric_cols:
            df_fs[numeric_cols] = sc.transform(df_fs[numeric_cols])

    return df_fs, enc, sc, selected_features


# ---------- Train + compare models, save best + metadata ----------

def train_and_compare_models(df_final):
    start_time = time.time()

    df_fs, enc, sc, selected_features = feature_selection_encode_scale(df_final, fit=True)

    with open(FEATURE_LIST_PATH, 'w') as f:
        for feat in selected_features:
            f.write(feat + "\n")
    joblib.dump(enc, ENCODER_PATH)
    joblib.dump(sc, SCALER_PATH)

    X = df_fs.drop(columns=['Label'])
    y = df_fs['Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    models = {
        'DecisionTree': (DecisionTreeClassifier(random_state=42),
                         {'max_depth': [3, 5, 7, None]}),

        'RandomForest': (RandomForestClassifier(random_state=42, n_jobs=-1),
                         {'n_estimators': [100, 200],
                          'max_depth': [None, 10]}),

        'SVM': (SVC(probability=True, random_state=42),
                {'C': [0.5, 1, 2],
                 'kernel': ['rbf']}),

        'XGBoost': (XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        ),
            {'n_estimators': [200, 350],
             'max_depth': [5, 8],
             'learning_rate': [0.05, 0.1]})
    }

    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_models = {}

    for name, (model, param_grid) in models.items():
        print(f"\n=== {name} + GridSearchCV ===")
        grid = GridSearchCV(
            model,
            param_grid,
            cv=skf,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        best_models[name] = best_model

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

        print("Best params:", grid.best_params_)
        print("Classification report:")
        print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Malicious (1)']))
        if roc_auc is not None:
            print(f"ROC-AUC: {roc_auc:.4f}")

        results.append({
            'model': name,
            'best_params': grid.best_params_,
            'accuracy': acc,
            'f1': f1,
            'roc_auc': roc_auc
        })

    results_df = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(results_df)

    best_by_f1 = results_df.sort_values('f1', ascending=False).iloc[0]['model']
    print(f"\n🔵 Best model by F1: {best_by_f1}")

    best_xgb = best_models.get('XGBoost')
    if best_xgb is not None:
        joblib.dump(best_xgb, BEST_MODEL_PATH)
        print(f"נשמר מודל XGBoost ל: {BEST_MODEL_PATH}")

    # explainability בסיסי: feature importances של XGBoost
    fi = None
    if best_xgb is not None and hasattr(best_xgb, "feature_importances_"):
        fi = dict(zip(X.columns.tolist(), best_xgb.feature_importances_.tolist()))

    train_time = time.time() - start_time

    meta = {
        "train_time_seconds": train_time,
        "n_samples": int(df_final.shape[0]),
        "n_features_raw": int(df_final.shape[1]),
        "n_features_selected": int(len(selected_features) - 1),
        "results": results,
        "best_model_by_f1": best_by_f1,
        "feature_importances_xgb": fi
    }
    with open(MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return best_models, results_df


# ---------- Prediction pipeline for single URL ----------

def load_selected_features():
    with open(FEATURE_LIST_PATH, 'r') as f:
        feats = [line.strip() for line in f.readlines()]
    return feats


def predict_url(url, model_path=BEST_MODEL_PATH, encoder_path=ENCODER_PATH,
                scaler_path=SCALER_PATH, feature_list_path=FEATURE_LIST_PATH):
    model = joblib.load(model_path)
    enc = joblib.load(encoder_path)
    sc = joblib.load(scaler_path)
    selected_features = load_selected_features()

    feats = extract_all_features_url(url, label=None)
    df = pd.DataFrame([feats])

    df = df.drop(columns=['Resolved_IP', 'url_original'], errors='ignore')

    for c in CATEGORICAL_COLS:
        if c not in df.columns:
            df[c] = "Unknown"

    cols_present = [c for c in CATEGORICAL_COLS if c in df.columns]
    if cols_present:
        df[cols_present] = enc.transform(df[cols_present])

    selected_no_label = [f for f in selected_features if f != 'Label']
    for col in selected_no_label:
        if col not in df.columns:
            df[col] = 0

    df = df[selected_no_label]

    num_cols = [c for c in df.columns if c not in cols_present]
    if num_cols:
        df[num_cols] = sc.transform(df[num_cols])

    y_pred = model.predict(df)
    y_proba = model.predict_proba(df)[:, 1] if hasattr(model, "predict_proba") else None

    return {
        "url": url,
        "malicious": bool(y_pred[0]),
        "score": float(y_proba[0]) if y_proba is not None else None
    }


# ---------- Main ----------

if __name__ == "__main__":
    # איפוס cache אם רצת על דאטהסט קודם
    get_whois_features.cache_clear()
    get_dns_validation.cache_clear()
    get_ip_from_domain.cache_clear()
    get_history_features.cache_clear()

    if os.path.exists(ENRICHED_CSV_FILE):
        df_final = pd.read_csv(ENRICHED_CSV_FILE)
        print(f"נטען קובץ קיים: {ENRICHED_CSV_FILE}, shape={df_final.shape}")
    else:
        df_final = build_enriched_dataset_parallel(RAW_DATA_PATH, max_workers=None)

    if df_final is not None and not df_final.empty:
        best_models, results_df = train_and_compare_models(df_final)
        test_url = "https://example.com/login"
        res = predict_url(test_url)
        print("\nPrediction example:", res)
    else:
        print("שגיאה: לא נוצר CSV מועשר, לא ניתן להתחיל אימון.")
