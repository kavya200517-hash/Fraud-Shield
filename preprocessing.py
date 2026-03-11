"""
preprocessing.py
Role A – AI/NLP Fraud Detection
Cleans and normalizes multi-domain fraud messages for downstream feature extraction and model training.
"""

import re
import string
import unicodedata
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HINDI_TRANSLITERATION_MAP = {
    "rupaye": "rupees", "paisa": "money", "khata": "account",
    "OTP": "OTP", "UPI": "UPI", "nahi": "not", "agar": "if",
}

SCAM_DOMAIN_PATTERNS = [
    r"job[_\-]?scam", r"courier[_\-]?fraud", r"upi[_\-]?fraud",
    r"kyc[_\-]?update", r"prize[_\-]?winner",
]

URL_REGEX = re.compile(
    r"(https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9\-]+\.[a-z]{2,}(?:/[^\s]*)?)"
)

PHONE_REGEX = re.compile(r"\+?[\d\s\-\(\)]{7,15}")


# ---------------------------------------------------------------------------
# Core cleaning helpers
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to ASCII where possible."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def transliterate_hinglish(text: str) -> str:
    """Replace common Hinglish/Hindi-Roman tokens with English equivalents."""
    for hindi_token, english_token in HINDI_TRANSLITERATION_MAP.items():
        text = re.sub(rf"\b{hindi_token}\b", english_token, text, flags=re.IGNORECASE)
    return text


def mask_urls(text: str, placeholder: str = "<URL>") -> str:
    """Replace URLs with a placeholder token."""
    return URL_REGEX.sub(placeholder, text)


def mask_phone_numbers(text: str, placeholder: str = "<PHONE>") -> str:
    """Replace phone numbers with a placeholder token."""
    return PHONE_REGEX.sub(placeholder, text)


def remove_extra_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def lowercase(text: str) -> str:
    return text.lower()


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


# ---------------------------------------------------------------------------
# Domain-specific tag injection
# ---------------------------------------------------------------------------

def tag_scam_domains(text: str) -> str:
    """Append domain tags detected from text for downstream feature use."""
    tags = []
    for pattern in SCAM_DOMAIN_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            tag = pattern.replace(r"[_\-]?", "_").strip("^$")
            tags.append(f"<DOMAIN:{tag.upper()}>")
    if tags:
        text = text + " " + " ".join(tags)
    return text


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def clean_message(
    text: str,
    mask_urls_flag: bool = True,
    mask_phones_flag: bool = True,
    transliterate: bool = True,
    tag_domains: bool = True,
    to_lower: bool = True,
) -> str:
    """
    Full preprocessing pipeline for a single message string.

    Steps:
        1. Unicode normalization
        2. Hinglish transliteration (optional)
        3. URL masking (optional)
        4. Phone masking (optional)
        5. Domain tagging (optional)
        6. Lowercase
        7. Whitespace normalization
    """
    text = normalize_unicode(text)
    if transliterate:
        text = transliterate_hinglish(text)
    if mask_urls_flag:
        text = mask_urls(text)
    if mask_phones_flag:
        text = mask_phone_numbers(text)
    if tag_domains:
        text = tag_scam_domains(text)
    if to_lower:
        text = lowercase(text)
    text = remove_extra_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# DataFrame-level pipeline
# ---------------------------------------------------------------------------

def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "message",
    label_col: Optional[str] = "label",
    **clean_kwargs,
) -> pd.DataFrame:
    """
    Apply clean_message to an entire DataFrame.

    Args:
        df: Input DataFrame.
        text_col: Column containing raw message text.
        label_col: Column containing labels (0=Legit, 1=Fraud). None if absent.
        **clean_kwargs: Forwarded to clean_message.

    Returns:
        Cleaned DataFrame with a 'cleaned_text' column added.
    """
    logger.info(f"Preprocessing {len(df)} messages …")
    df = df.copy()
    df["cleaned_text"] = df[text_col].astype(str).apply(
        lambda x: clean_message(x, **clean_kwargs)
    )
    # Drop rows that became empty after cleaning
    before = len(df)
    df = df[df["cleaned_text"].str.strip().astype(bool)].reset_index(drop=True)
    logger.info(f"Dropped {before - len(df)} empty rows after cleaning.")

    if label_col and label_col in df.columns:
        df[label_col] = df[label_col].astype(int)

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess fraud message datasets.")
    parser.add_argument("--input", required=True, help="Path to raw CSV file.")
    parser.add_argument("--output", required=True, help="Path to save cleaned CSV.")
    parser.add_argument("--text-col", default="message")
    parser.add_argument("--label-col", default="label")
    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    clean_df = preprocess_dataframe(raw_df, text_col=args.text_col, label_col=args.label_col)
    clean_df.to_csv(args.output, index=False)
    print(f"Saved cleaned dataset to {args.output}  ({len(clean_df)} rows)")
