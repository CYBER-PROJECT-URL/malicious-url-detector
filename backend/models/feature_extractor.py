# backend/feature_extractor.py
import pandas as pd
import tldextract
import ipaddress
import re
from urllib.parse import urlparse


# --- Helper Functions (PLACEHOLDER IMPLEMENTATIONS) ---

def get_lexical_features(url):
    """Extracts lexical features from the URL string."""
    parsed_url = urlparse(url)
    ext = tldextract.extract(url)

    features = {
        'url_length': len(url),
        'domain_length': len(ext.domain),
        'path_length': len(parsed_url.path),
        'num_dots': url.count('.'),
        'num_at_sign': url.count('@'),
        'num_hyphens': url.count('-'),
        'suspicious_keywords': 1 if re.search(r'login|verify|secure|account', url, re.IGNORECASE) else 0,
        # Add more lexical features as proposed...
    }
    return features


def get_host_based_features(url):
    """Extracts host-based features (requires external libs/services like WHOIS)."""
    # In a real system, this would involve slow external lookups (WHOIS, DNS)
    # For a simplified environment, we focus on easily derivable features:
    parsed_url = urlparse(url)
    host = parsed_url.netloc

    is_ip = 0
    try:
        ipaddress.ip_address(host)
        is_ip = 1
    except ValueError:
        pass

    features = {
        'is_ip_address': is_ip,
        # Placeholder for WHOIS data: In the final project, you must integrate WHOIS or use a dataset with pre-extracted host features.
        'domain_age_days': 0,  # Placeholder
        'registrar_info_available': 0  # Placeholder
    }
    return features


def get_structural_indicators(url):
    """Extracts structural indicators."""
    parsed_url = urlparse(url)

    features = {
        'has_https': 1 if parsed_url.scheme == 'https' else 0,
        'has_shortening': 1 if len(parsed_url.netloc) < 15 and re.match(r't\.co|bit\.ly', parsed_url.netloc) else 0,
        # Add more structural indicators...
    }
    return features


def extract_all_features(url: str) -> pd.DataFrame:
    """
    Main function to aggregate all feature types into a DataFrame for the ML model.
    The order of features MUST match the order used during model training!
    """
    url = url.strip()

    # 1. Gather all features
    lexical = get_lexical_features(url)
    host = get_host_based_features(url)
    structural = get_structural_indicators(url)

    # 2. Combine all features
    all_features = {**lexical, **host, **structural}

    # 3. Convert to DataFrame (essential for consistent feature order/shape)
    feature_vector = pd.DataFrame([all_features])

    # NOTE: In a production environment, you would also need to ensure
    # the DataFrame columns are in the correct order and scale the features
    # using the same scaler/encoder used during training.

    return feature_vector


# Example usage (for testing):
if __name__ == '__main__':
    test_url = "http://www.google.com/search?q=test"
    features = extract_all_features(test_url)
    print(features)