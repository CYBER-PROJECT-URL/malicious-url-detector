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
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score
from functools import wraps

# --- 0. הגדרות וקבועים ---
RAW_DATA_PATH = 'DataSet MALICIOS URLs/DataSet MALICIOS URLs/CSV WITH MAL URLS/dataSet Kaggle.zip'
ENRICHED_CSV_FILE = 'dataset_final.csv'
REJECT_LOG_FILE = 'reject_log.csv'
API_DELAY_SECONDS = 2
FAILURE_CODE = -1  # קוד כשל טכני (שגיאת רשת/API)


# --- 1. דקורטור עמידות (Retry) ---

def retry(times=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt < times - 1:
                        time.sleep(delay)
                    else:
                        return None
        return wrapper
    return decorator


# --- 2. פונקציות חילוץ ---

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


@retry(times=3, delay=2)
def get_whois_features(domain):
    if not domain or domain.count('.') < 1:
        return 0, 0
    w = whois.whois(domain, timeout=4)
    dns_available = 1 if w.domain_name else 0
    age_days = 0
    if w.creation_date:
        date = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
        if isinstance(date, datetime):
            age_days = (datetime.now() - date).days
    return age_days, dns_available


@retry(times=3, delay=1)
def get_dns_validation(domain):
    if not domain or domain.count('.') < 1:
        return 0, 0, 0
    a_record, ns_record, mx_record = 0, 0, 0
    try:
        if dns.resolver.resolve(domain, 'A', lifetime=2):
            a_record = 1
        if dns.resolver.resolve(domain, 'NS', lifetime=2):
            ns_record = 1
        if dns.resolver.resolve(domain, 'MX', lifetime=2):
            mx_record = 1
    except:
        pass
    return a_record, ns_record, mx_record


@retry(times=3, delay=1)
def get_ip_from_domain(domain):
    return socket.gethostbyname(domain)


def get_geolocation_features(ip):
    if not ip or ip in ['127.0.0.1', '0.0.0.0']:
        return "Not Found", 0
    try:
        data = ipapi.location(ip)
        country = data.get('country_name', 'Not Found')
        is_safe = 1 if country in ['United States', 'Canada', 'United Kingdom', 'Germany', 'France', 'Israel'] else 0
        return country, is_safe
    except:
        return "API Error", FAILURE_CODE


@retry(times=3, delay=1)
def get_history_features(domain):
    first_seen_ssl = 0
    first_seen_archive = 0

    archive_url = f"https://archive.org/wayback/available?url={domain}"
    response = requests.get(archive_url, timeout=4).json()
    if response.get('archived_snapshots') and response['archived_snapshots'].get('closest'):
        timestamp = response['archived_snapshots']['closest']['timestamp']
        first_seen_archive = (datetime.now() - datetime.strptime(timestamp[:8], "%Y%m%d")).days

    crtsh_url = f"https://crt.sh/?q={domain}&output=json"
    response = requests.get(crtsh_url, timeout=4).json()
    if response and isinstance(response, list) and len(response) > 0:
        earliest_date = min([
            datetime.strptime(item.get('not_before', '2099-01-01T00:00:00')[:10], "%Y-%m-%d")
            for item in response if item.get('not_before')
        ])
        if earliest_date:
            first_seen_ssl = (datetime.now() - earliest_date).days

    return first_seen_ssl, first_seen_archive


@retry(times=3, delay=1)
def get_url_availability(url):
    response = requests.head(url, timeout=5, allow_redirects=True)
    is_available = 1 if 200 <= response.status_code < 400 else 0
    return is_available, response.status_code


# --- 3. לוגיקת ניקוי ---

def is_valid_domain(row):
    genuine_failure_score = 0
    log_reasons = []

    if row['DNS_A_Record'] == 0:
        genuine_failure_score += 1
        log_reasons.append("No_A_Record")

    if row['DNS_NS_Record'] == 0:
        genuine_failure_score += 1
        log_reasons.append("No_NS_Record")

    if row['DNS_Record_Available'] == 0:
        genuine_failure_score += 1
        log_reasons.append("No_WHOIS_DNS_Entry")

    if row['HTTP_Status_Code'] >= 400 or row['HTTP_Status_Code'] == 0:
        if row['HTTP_Status_Code'] != FAILURE_CODE and row['HTTP_Status_Code'] != 999:
            genuine_failure_score += 1
            log_reasons.append(f"Bad_HTTP_Status_{row['HTTP_Status_Code']}")
        elif row['HTTP_Status_Code'] == 999:
            log_reasons.append("Critical_Connection_Failure")
            genuine_failure_score += 1

    if row['URL_Available'] == 0:
        genuine_failure_score += 1
        log_reasons.append("URL_Not_Available")

    return genuine_failure_score < 3, log_reasons


# --- 4. מאפיינים לקסיקליים ---

def extract_all_features_lexical(row: pd.Series) -> pd.Series:
    url = row['url_original'].strip()
    parsed = urlparse(url)
    ext = tldextract.extract(url)

    features = {}

    features['Label'] = row['Label']
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
    features['Domain'] = get_domain_from_url(url)

    return pd.Series(features)


# --- 5. יצירת/המשך CSV מועשר ---

def create_or_continue_enriched_dataset(raw_data_path):
    try:
        df_raw = pd.read_csv(raw_data_path, compression='zip')
        malicious_types = ['phishing', 'malware', 'defacement']
        df_raw['Label'] = df_raw['type'].apply(lambda x: 1.0 if x in malicious_types else 0.0)
        df_raw = df_raw.rename(columns={'url': 'url_original'})
    except Exception as e:
        print(f"שגיאה בטעינת הדאטה: {e}")
        return None

    print("מתחיל חילוץ מאפיינים לקסיקליים...")
    df_lexical = df_raw.apply(extract_all_features_lexical, axis=1)

    if os.path.exists(ENRICHED_CSV_FILE):
        df_cache = pd.read_csv(ENRICHED_CSV_FILE)
        processed_domains = df_cache['Domain'].unique()
        df_to_process = df_lexical[~df_lexical['Domain'].isin(processed_domains)].copy()
        write_header = False
        print(
            f"נמצא קובץ '{ENRICHED_CSV_FILE}' עם {len(df_cache)} רשומות. ממשיך עם {len(df_to_process)} דומיינים חדשים...")
    else:
        df_to_process = df_lexical.copy()
        write_header = True
        print("מתחיל עיבוד מלא...")

    if not os.path.exists(REJECT_LOG_FILE):
        with open(REJECT_LOG_FILE, 'w') as f:
            f.write("url_original,Label,Rejection_Reason\n")

    valid_domains_count = len(df_cache) if os.path.exists(ENRICHED_CSV_FILE) else 0

    if not df_to_process.empty:
        for i, row in df_to_process.iterrows():
            domain = row['Domain']
            url = row['url_original']

            age_dns_avail = get_whois_features(domain)
            age, dns_avail = (age_dns_avail if age_dns_avail is not None else (FAILURE_CODE, 0))

            dns_recs = get_dns_validation(domain)
            a_record, ns_record, mx_record = (
                dns_recs if dns_recs is not None else (FAILURE_CODE, FAILURE_CODE, FAILURE_CODE)
            )

            history_recs = get_history_features(domain)
            ssl_days, archive_days = (
                history_recs if history_recs is not None else (FAILURE_CODE, FAILURE_CODE)
            )

            avail_status = get_url_availability(url)
            is_available, status_code = (
                avail_status if avail_status is not None else (0, 999)
            )

            ip = get_ip_from_domain(domain)
            ip = ip if ip is not None else ""

            geo_recs = get_geolocation_features(ip)
            country, safe = (
                geo_recs if geo_recs is not None else ("API Error", FAILURE_CODE)
            )

            row['Domain_Age_Days'] = age
            row['DNS_Record_Available'] = dns_avail
            row['DNS_A_Record'] = a_record
            row['DNS_NS_Record'] = ns_record
            row['DNS_MX_Record'] = mx_record
            row['First_Seen_SSL_Days'] = ssl_days
            row['First_Seen_Archive_Days'] = archive_days
            row['URL_Available'] = is_available
            row['HTTP_Status_Code'] = status_code
            row['Resolved_IP'] = ip if ip else "None"
            row['Server_Country'] = country
            row['Server_Safety_Score'] = safe

            # External_Failure_Count – ספירת כשלים חיצוניים (0 ו־FAILURE_CODE)
            failure_signals = [
                ('DNS_A_Record', row['DNS_A_Record']),
                ('DNS_NS_Record', row['DNS_NS_Record']),
                ('DNS_MX_Record', row['DNS_MX_Record']),
                ('DNS_Record_Available', row['DNS_Record_Available']),
                ('URL_Available', row['URL_Available']),
                ('HTTP_Status_Code', row['HTTP_Status_Code']),
                ('First_Seen_SSL_Days', row['First_Seen_SSL_Days']),
                ('First_Seen_Archive_Days', row['First_Seen_Archive_Days']),
                ('Server_Safety_Score', row['Server_Safety_Score'])
            ]

            external_failure_count = 0
            for name, val in failure_signals:
                if name == 'HTTP_Status_Code':
                    if val == 0 or val == FAILURE_CODE or val >= 400:
                        external_failure_count += 1
                elif name == 'URL_Available':
                    if val == 0 or val == FAILURE_CODE:
                        external_failure_count += 1
                else:
                    if val == 0 or val == FAILURE_CODE:
                        external_failure_count += 1

            row['External_Failure_Count'] = external_failure_count

            is_valid, reasons = is_valid_domain(row)
            if not is_valid:
                with open(REJECT_LOG_FILE, 'a') as f:
                    f.write(f"\"{url}\",{row['Label']},{';'.join(reasons)}\n")
                continue

            valid_domains_count += 1

            row_to_save = row.drop(labels=['url_original']).copy()

            numeric_cols = [col for col, val in row_to_save.items() if isinstance(val, (int, float, np.number))]
            for col in numeric_cols:
                val = row_to_save[col]
                try:
                    if pd.isna(val) or np.isinf(val):
                        row_to_save[col] = FAILURE_CODE
                except TypeError:
                    continue

            row_to_save.to_frame().T.to_csv(
                ENRICHED_CSV_FILE, mode='a', header=write_header, index=False
            )
            write_header = False

            if (i + 1) % 50 == 0:
                print(f"עובדו {i + 1} דוגמאות. נשמרו {valid_domains_count} שורות. ממתין...")
            time.sleep(API_DELAY_SECONDS)

        print("\n✅ סיום מלא של חילוץ הנתונים. ה-CSV המועשר נוצר.")
        print(f"סה\"כ דומיינים תקינים שנשמרו: {valid_domains_count}")

    if os.path.exists(ENRICHED_CSV_FILE):
        return pd.read_csv(ENRICHED_CSV_FILE)
    else:
        return None


# --- 6. אימון המודל ---

def train_model_from_enriched_data(df_final):
    df_final = df_final.drop(columns=['Resolved_IP'], errors='ignore')
    df_final['Server_Country'] = df_final['Server_Country'].replace(['Not Found', 'API Error'], 'Service_Issue')

    df_final = pd.get_dummies(df_final, columns=['Server_Country', 'Domain'],
                              prefix=['Country', 'Domain'], drop_first=True)

    df_final.replace(FAILURE_CODE, 0, inplace=True)
    df_final.fillna(0, inplace=True)

    X = df_final.drop(columns=['Label'])
    y = df_final['Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=350,
        learning_rate=0.05,
        max_depth=8,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )

    print("\nמאמן את המודל...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n--- דוח סופי ---")
    print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Malicious (1)']))

    f1 = f1_score(y_test, y_pred, pos_label=1)
    print(f"F1-Score סופי (Malicious): {f1:.4f}")

    return model, X.columns.tolist()


# --- 7. הרצה ראשית ---

df_final = create_or_continue_enriched_dataset(RAW_DATA_PATH)

if df_final is not None and not df_final.empty:
    print(f"\nCSV מועשר נוצר בהצלחה. סה\"כ {df_final.shape[1]} מאפיינים.")
    final_model, feature_names = train_model_from_enriched_data(df_final)
else:
    print("שגיאה: לא נוצר CSV מועשר, לא ניתן להתחיל אימון.")
