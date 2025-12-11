# backend/models/feature_extractor.py

import pandas as pd
import numpy as np
import whois
from urllib.parse import urlparse
from datetime import datetime
import time
import re
import socket
import ipapi
import tldextract
import requests
import dns.resolver
from functools import wraps, lru_cache

# ---------- קבועים (חייבים להיות זהים ל-train_model) ----------

FAILURE_CODE = -1

URL_SHORTENERS = {
    'bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'ow.ly', 'buff.ly',
    'is.gd', 'cutt.ly', 'shorturl.at', 'rebrand.ly'
}

SAFE_COUNTRIES = [
    'United States', 'Canada', 'United Kingdom',
    'Germany', 'France', 'Israel'
]

socket.setdefaulttimeout(2)


# ---------- Retry decorator ----------

def retry(times=3, delay=1, backoff=2):
    """דקורטור לניסיונות חוזרים עם backoff אקספוננציאלי."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except requests.HTTPError as e:
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

def shannon_entropy(data: str) -> float:
    """אנטרופיית שאנון של המחרוזת (פיצ'ר חשוב ל-URL אקראיים). [web:191][web:192]"""
    if not data:
        return 0.0
    s = pd.Series(list(data))
    probs = s.value_counts(normalize=True)
    return float(-(probs * np.log2(probs)).sum())


def get_domain_from_url(url: str) -> str:
    """מחלץ דומיין ראשי בעזרת tldextract. [web:26]"""
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
    except Exception:
        return ""


def get_ip_type(ip: str) -> str:
    """מזהה סוג IP (IPv4/IPv6/Unknown)."""
    try:
        socket.inet_pton(socket.AF_INET, ip)
        return "IPv4"
    except OSError:
        try:
            socket.inet_pton(socket.AF_INET6, ip)
            return "IPv6"
        except OSError:
            return "Unknown"


# ---------- קריאות חיצוניות עם cache ----------

@lru_cache(maxsize=50000)
@retry(times=3, delay=1.5)
def get_whois_features(domain: str):
    """מחזיר (age_days, dns_available, country, registrar) או None במקרה כשל."""
    if not domain or domain.count('.') < 1:
        return 0, 0, "Unknown", "Unknown"
    try:
        w = whois.whois(domain, timeout=4)
    except Exception:
        return None

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
def get_dns_validation(domain: str):
    """A / NS / MX רשומות DNS בסיסיות."""
    if not domain or domain.count('.') < 1:
        return 0, 0, 0
    a_record = ns_record = mx_record = 0
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
def get_ip_from_domain(domain: str):
    """פענוח IP ע"י DNS."""
    try:
        return socket.gethostbyname(domain)
    except Exception:
        return None


@retry(times=3, delay=1)
def get_geolocation_features(ip: str):
    """IP → מדינה, safety score (0/1), country_code."""
    if not ip or ip in ['127.0.0.1', '0.0.0.0']:
        return "Not Found", 0, "Unknown"
    try:
        data = ipapi.location(ip, timeout=3)
        server_country = data.get('country_name', 'Not Found')
        server_country_code = data.get('country', 'Unknown')
        is_safe = 1 if server_country in SAFE_COUNTRIES else 0
        return server_country, is_safe, server_country_code
    except Exception:
        return "API Error", FAILURE_CODE, "Unknown"


@lru_cache(maxsize=50000)
@retry(times=3, delay=1)
def get_history_features(domain: str):
    """Wayback Machine + crt.sh → (first_seen_ssl_days, first_seen_archive_days)."""
    first_seen_ssl = FAILURE_CODE
    first_seen_archive = FAILURE_CODE

    try:
        archive_url = f"https://archive.org/wayback/available?url={domain}"
        r = requests.get(archive_url, timeout=3)
        response = r.json()
        snap = response.get('archived_snapshots', {}).get('closest')
        if snap and snap.get('timestamp'):
            ts = snap['timestamp'][:8]
            dt = datetime.strptime(ts, "%Y%m%d")
            first_seen_archive = (datetime.now() - dt).days
    except Exception:
        pass

    try:
        crtsh_url = f"https://crt.sh/?q={domain}&output=json"
        r = requests.get(crtsh_url, timeout=3)
        data = r.json()
        if data and isinstance(data, list):
            dates = [
                datetime.strptime(item.get('not_before', '2099-01-01')[:10], "%Y-%m-%d")
                for item in data if item.get('not_before')
            ]
            if dates:
                earliest = min(dates)
                first_seen_ssl = (datetime.now() - earliest).days
    except Exception:
        pass

    return first_seen_ssl, first_seen_archive


@retry(times=3, delay=1)
def get_url_availability(url: str):
    """בקשת HEAD → זמינות + קוד סטטוס."""
    try:
        if not url.startswith("http"):
            url = "http://" + url
        response = requests.head(url, timeout=3, allow_redirects=True)
        is_available = 1 if 200 <= response.status_code < 400 else 0
        return is_available, response.status_code
    except Exception:
        return 0, 999


# ---------- פיצ'רים לקסיקליים ל-URL ----------

def extract_lexical_features(url: str):
    """
    מחזיר dict עם כל הפיצ'רים הלקסיקליים שהמודל משתמש בהם.
    חייב להיות עקבי עם train_model. [web:76][web:194]
    """
    feats = {}
    parsed = urlparse(url)
    ext = tldextract.extract(url)

    # אורך ורכיבים
    feats['url_length'] = len(url)
    feats['domain_length'] = len(ext.domain) if ext.domain else 0
    feats['path_length'] = len(parsed.path)
    feats['tld_length'] = len(ext.suffix) if ext.suffix else 0

    # אנטרופיה ויחסים
    feats['entropy'] = shannon_entropy(url)
    feats['domain_to_path_ratio'] = (
        feats['domain_length'] / (feats['path_length'] + 1e-6)
    )

    # תווים וסימנים
    feats['num_dots'] = url.count('.')
    feats['num_at_sign'] = url.count('@')
    feats['num_hyphens'] = url.count('-')
    feats['num_digits'] = sum(c.isdigit() for c in url)
    feats['num_special_chars'] = sum(not c.isalnum() for c in url)

    feats['digit_ratio'] = feats['num_digits'] / (len(url) + 1e-6)
    feats['special_char_ratio'] = feats['num_special_chars'] / (len(url) + 1e-6)

    # HTTPS / IP / subdomain
    feats['has_https'] = 1 if parsed.scheme == 'https' else 0
    feats['has_ip'] = 1 if re.match(
        r'^(\d{1,3}\.){3}\d{1,3}$', parsed.netloc.split(':')[0]
    ) else 0

    subdomain = ext.subdomain or ""
    feats['subdomain_depth'] = subdomain.count('.') + (1 if subdomain else 0)
    feats['subdomain_length'] = len(subdomain)

    # path tokens
    path_tokens = [p for p in parsed.path.split('/') if p]
    feats['path_token_count'] = len(path_tokens)

    # מילות מפתח חשודות
    suspicious_keywords = [
        'login', 'verify', 'secure', 'account', 'update',
        'confirm', 'pay', 'bank', 'webscr', 'token', 'signin'
    ]
    feats['suspicious_keywords_count'] = sum(
        1 for kw in suspicious_keywords if re.search(kw, url, re.IGNORECASE)
    )

    # קיצור כתובת
    netloc = parsed.netloc.lower()
    feats['is_shortened'] = 1 if netloc in URL_SHORTENERS else 0

    # דומיין כ-קטגוריה
    feats['Domain'] = get_domain_from_url(url)

    return feats


# ---------- Unified pipeline for single URL ----------

def extract_all_features_url(url: str, label=None):
    """
    מחלץ את *כל* הפיצ'רים שהמודל צריך עבור URL יחיד.
    מחזיר dict עם פיצ'רים + Label (אם סופק).
    """
    url = (url or "").strip()

    # 1. פיצ'רים לקסיקליים
    feats = extract_lexical_features(url)
    domain = feats['Domain']

    # 2. WHOIS
    age_dns_country_reg = get_whois_features(domain)
    if age_dns_country_reg is not None:
        age_days, dns_avail, whois_country, whois_registrar = age_dns_country_reg
    else:
        age_days, dns_avail, whois_country, whois_registrar = FAILURE_CODE, 0, "Unknown", "Unknown"

    feats["Domain_Age_Days"] = age_days
    feats["DNS_Record_Available"] = dns_avail
    feats["WHOIS_Country"] = whois_country
    feats["WHOIS_Registrar"] = whois_registrar

    # 3. DNS
    dns_recs = get_dns_validation(domain)
    a_record, ns_record, mx_record = (
        dns_recs if dns_recs is not None else (FAILURE_CODE, FAILURE_CODE, FAILURE_CODE)
    )
    feats["DNS_A_Record"] = a_record
    feats["DNS_NS_Record"] = ns_record
    feats["DNS_MX_Record"] = mx_record

    # 4. היסטוריה (SSL + Archive)
    history_recs = get_history_features(domain)
    ssl_days, archive_days = (
        history_recs if history_recs is not None else (FAILURE_CODE, FAILURE_CODE)
    )
    feats["First_Seen_SSL_Days"] = ssl_days
    feats["First_Seen_Archive_Days"] = archive_days

    # 5. זמינות HTTP
    avail_status = get_url_availability(url)
    is_available, status_code = (
        avail_status if avail_status is not None else (0, 999)
    )
    feats["URL_Available"] = is_available
    feats["HTTP_Status_Code"] = status_code

    # 6. IP + GEO
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

    # 7. External_Failure_Count
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
            if val == 0 or val == FAILURE_CODE or (isinstance(val, int) and val >= 400):
                external_failure_count += 1
        else:
            if val == 0 or val == FAILURE_CODE:
                external_failure_count += 1

    feats["External_Failure_Count"] = external_failure_count

    if label is not None:
        feats['Label'] = label

    return feats
