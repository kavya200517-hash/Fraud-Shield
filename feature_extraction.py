"""
feature_extraction.py
Role A – AI/NLP Fraud Detection
Extracts hand-crafted features: urgency markers, suspicious URL structures,
fake bank names, and statistical text features.
"""

import re
import math
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature lexicons
# ---------------------------------------------------------------------------

URGENCY_MARKERS = [
    "urgent", "immediately", "act now", "last chance", "expires today",
    "within 24 hours", "limited time", "don't delay", "right now",
    "asap", "alert", "warning", "account blocked", "suspended", "verify now",
    "final notice", "claim your prize", "congratulations you won",
]

FAKE_BANK_PATTERNS = [
    r"sbi\s*(?:bank|india|ltd)?",
    r"hdfc\s*(?:bank)?",
    r"icici\s*(?:bank)?",
    r"axis\s*(?:bank)?",
    r"paytm\s*(?:bank|payments)?",
    r"phonepe",
    r"google\s*pay",
    r"bhim\s*upi",
    r"npci",
    r"uidai",
    r"nsdl",
    r"reserve\s*bank",
    r"rbi",
]

SUSPICIOUS_URL_KEYWORDS = [
    "login", "verify", "secure", "update", "account", "confirm",
    "banking", "kyc", "otp", "reward", "prize", "winner", "free",
]

SENSITIVE_REQUEST_PATTERNS = [
    r"\botp\b", r"\bpin\b", r"\bcvv\b", r"\bpassword\b",
    r"\baadhar\b", r"\bpan\s*(?:card|number)?\b",
    r"\baccount\s*number\b", r"\bifsc\b",
    r"share\s*(?:your)?\s*(?:otp|pin|password)",
    r"do\s*not\s*share",
]


# ---------------------------------------------------------------------------
# Individual feature extractors
# ---------------------------------------------------------------------------

def count_urgency_markers(text: str) -> int:
    text_lower = text.lower()
    return sum(1 for m in URGENCY_MARKERS if m in text_lower)


def has_fake_bank_name(text: str) -> int:
    text_lower = text.lower()
    for pattern in FAKE_BANK_PATTERNS:
        if re.search(pattern, text_lower):
            return 1
    return 0


def count_sensitive_requests(text: str) -> int:
    text_lower = text.lower()
    return sum(1 for p in SENSITIVE_REQUEST_PATTERNS if re.search(p, text_lower))


def count_urls(text: str) -> int:
    url_pattern = re.compile(r"https?://[^\s]+|www\.[^\s]+|<URL>")
    return len(url_pattern.findall(text))


def has_suspicious_url(text: str) -> int:
    urls = re.findall(r"https?://[^\s]+|www\.[^\s]+", text, flags=re.IGNORECASE)
    for url in urls:
        parsed = urlparse(url if url.startswith("http") else "http://" + url)
        hostname = parsed.hostname or ""
        path = parsed.path.lower()
        # Check for IP-address-based URLs
        if re.match(r"\d{1,3}(\.\d{1,3}){3}", hostname):
            return 1
        # Homoglyph / lookalike detection (numbers replacing letters)
        if re.search(r"[0-9][a-z]|[a-z][0-9]", hostname):
            return 1
        # Suspicious keywords in URL
        for keyword in SUSPICIOUS_URL_KEYWORDS:
            if keyword in hostname or keyword in path:
                return 1
        # Excessive subdomain depth
        if hostname.count(".") >= 3:
            return 1
    return 0


def url_entropy(text: str) -> float:
    """Shannon entropy of URL characters — high entropy suggests obfuscation."""
    urls = re.findall(r"https?://[^\s]+|www\.[^\s]+", text, flags=re.IGNORECASE)
    if not urls:
        return 0.0
    url = urls[0]
    freq = {c: url.count(c) / len(url) for c in set(url)}
    return -sum(p * math.log2(p) for p in freq.values() if p > 0)


def message_length(text: str) -> int:
    return len(text)


def exclamation_count(text: str) -> int:
    return text.count("!")


def question_count(text: str) -> int:
    return text.count("?")


def capitalisation_ratio(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


def digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.isdigit()) / len(text)


def has_phone_number(text: str) -> int:
    return int(bool(re.search(r"\+?[\d\s\-\(\)]{7,15}", text)))


def has_masked_tokens(text: str) -> int:
    """Check if upstream preprocessing masked any URLs/phones."""
    return int("<URL>" in text or "<PHONE>" in text)


# ---------------------------------------------------------------------------
# Composite feature vector
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "urgency_count",
    "has_fake_bank",
    "sensitive_request_count",
    "url_count",
    "has_suspicious_url",
    "url_entropy",
    "message_length",
    "exclamation_count",
    "question_count",
    "capitalisation_ratio",
    "digit_ratio",
    "has_phone",
    "has_masked_tokens",
]


def extract_features(text: str) -> Dict[str, float]:
    """Return a dict of all hand-crafted features for a single message."""
    return {
        "urgency_count": count_urgency_markers(text),
        "has_fake_bank": has_fake_bank_name(text),
        "sensitive_request_count": count_sensitive_requests(text),
        "url_count": count_urls(text),
        "has_suspicious_url": has_suspicious_url(text),
        "url_entropy": url_entropy(text),
        "message_length": message_length(text),
        "exclamation_count": exclamation_count(text),
        "question_count": question_count(text),
        "capitalisation_ratio": capitalisation_ratio(text),
        "digit_ratio": digit_ratio(text),
        "has_phone": has_phone_number(text),
        "has_masked_tokens": has_masked_tokens(text),
    }


def extract_feature_matrix(texts: List[str]) -> np.ndarray:
    """
    Extract feature vectors for a list of texts.

    Returns:
        np.ndarray of shape (n_samples, n_features)
    """
    rows = [list(extract_features(t).values()) for t in texts]
    return np.array(rows, dtype=np.float32)


def build_feature_dataframe(df: pd.DataFrame, text_col: str = "cleaned_text") -> pd.DataFrame:
    """
    Append feature columns to an existing DataFrame.
    """
    features = df[text_col].apply(extract_features).apply(pd.Series)
    return pd.concat([df, features], axis=1)


# ---------------------------------------------------------------------------
# Save feature vectors
# ---------------------------------------------------------------------------

def save_feature_vectors(
    df: pd.DataFrame,
    output_path: str = "data/processed/feature_vectors.npy",
    text_col: str = "cleaned_text",
) -> np.ndarray:
    matrix = extract_feature_matrix(df[text_col].tolist())
    np.save(output_path, matrix)
    logger.info(f"Saved feature matrix {matrix.shape} → {output_path}")
    return matrix


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Extract hand-crafted features.")
    parser.add_argument("--input", required=True, help="Cleaned messages CSV.")
    parser.add_argument("--output-npy", default="data/processed/feature_vectors.npy")
    parser.add_argument("--text-col", default="cleaned_text")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    save_feature_vectors(df, output_path=args.output_npy, text_col=args.text_col)
    print("Feature extraction complete.")
