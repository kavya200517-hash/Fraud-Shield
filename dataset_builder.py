"""
dataset_builder.py
Role A – AI/NLP Fraud Detection
Consolidates SMS Spam Collection, PhishTank URLs, and Indian Bank Scam Corpus
into a unified, stratified train/val/test split.
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_dataframe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset source configs
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "sms_spam": {
        "filename": "sms_spam_collection.csv",
        "text_col": "message",
        "label_col": "label",          # 'ham' / 'spam'
        "label_map": {"ham": 0, "spam": 1},
        "domain": "sms_spam",
    },
    "phishtank": {
        "filename": "phishtank_urls.csv",
        "text_col": "url",
        "label_col": "verified",       # True / False (phishing or not)
        "label_map": {True: 1, False: 0, "yes": 1, "no": 0},
        "domain": "phishing_url",
    },
    "indian_bank_scam": {
        "filename": "indian_bank_scam_corpus.csv",
        "text_col": "message",
        "label_col": "label",          # 0 / 1
        "label_map": {0: 0, 1: 1, "legit": 0, "fraud": 1},
        "domain": "bank_scam",
    },
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_single_dataset(raw_dir: str, cfg: Dict) -> pd.DataFrame:
    """Load one raw CSV, normalise labels, and add a domain tag."""
    filepath = os.path.join(raw_dir, cfg["filename"])
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath} — skipping.")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    text_col = cfg["text_col"]
    label_col = cfg["label_col"]

    # Rename to canonical column names
    df = df[[text_col, label_col]].rename(
        columns={text_col: "message", label_col: "label"}
    )

    # Normalise labels
    df["label"] = df["label"].map(cfg["label_map"]).fillna(df["label"])
    df["label"] = df["label"].astype(int)

    # Add domain metadata
    df["domain"] = cfg["domain"]

    logger.info(f"Loaded {cfg['filename']}: {len(df)} rows  "
                f"(fraud={df['label'].sum()}, legit={(df['label']==0).sum()})")
    return df


def load_all_datasets(raw_dir: str) -> pd.DataFrame:
    """Consolidate all configured datasets into one DataFrame."""
    frames = []
    for name, cfg in DATASET_CONFIGS.items():
        df = load_single_dataset(raw_dir, cfg)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No datasets found in {raw_dir}. "
                                "Please add CSV files as per DATASET_CONFIGS.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["message", "label"])
    combined = combined.drop_duplicates(subset=["message"])
    logger.info(f"Combined dataset: {len(combined)} rows  "
                f"(fraud={combined['label'].sum()}, "
                f"legit={(combined['label']==0).sum()})")
    return combined


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def create_splits(
    df: pd.DataFrame,
    val_size: float = 0.10,
    test_size: float = 0.10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / val / test split."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=random_state
    )
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=train_val["label"],
        random_state=random_state
    )
    logger.info(f"Split sizes — train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    processed_dir: str,
) -> None:
    os.makedirs(processed_dir, exist_ok=True)
    full = pd.concat([train, val, test], ignore_index=True)
    full.to_csv(os.path.join(processed_dir, "cleaned_messages.csv"), index=False)

    split_meta = {
        "train_indices": list(train.index),
        "val_indices": list(val.index),
        "test_indices": list(test.index),
    }
    with open(os.path.join(processed_dir, "tokenized_dataset.pkl"), "wb") as f:
        pickle.dump({"data": full, "splits": split_meta}, f)

    logger.info(f"Saved cleaned_messages.csv and tokenized_dataset.pkl to {processed_dir}")


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_dataset(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    val_size: float = 0.10,
    test_size: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full dataset build pipeline."""
    combined = load_all_datasets(raw_dir)
    cleaned = preprocess_dataframe(combined, text_col="message", label_col="label")
    train, val, test = create_splits(cleaned, val_size=val_size, test_size=test_size)
    save_splits(train, val, test, processed_dir)
    return train, val, test


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Build unified fraud detection dataset.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--test-size", type=float, default=0.10)
    args = parser.parse_args()

    build_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        val_size=args.val_size,
        test_size=args.test_size,
    )
