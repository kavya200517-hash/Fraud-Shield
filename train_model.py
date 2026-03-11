"""
train_model.py
Role A – AI/NLP Fraud Detection
Fine-tunes a DistilBERT model for binary fraud classification (Legit=0, Fraud=1).
"""

import os
import logging
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_fscore_support
)
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_CFG = {
    "model_name": "distilbert-base-uncased",
    "max_length": 128,
    "batch_size": 32,
    "num_epochs": 5,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "dropout": 0.3,
    "seed": 42,
    "output_model_path": "models/fraud_classifier.pt",
    "processed_dir": "data/processed",
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FraudDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class FraudClassifierTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(cfg["model_name"])
        self.model = DistilBertForSequenceClassification.from_pretrained(
            cfg["model_name"],
            num_labels=2,
            hidden_dropout_prob=cfg.get("dropout", 0.1),
        ).to(self.device)

    # ------------------------------------------------------------------ data
    def _load_splits(self):
        import pickle
        pkl_path = os.path.join(self.cfg["processed_dir"], "tokenized_dataset.pkl")
        with open(pkl_path, "rb") as f:
            bundle = pickle.load(f)

        data: pd.DataFrame = bundle["data"]
        splits = bundle["splits"]

        train = data.loc[splits["train_indices"]]
        val   = data.loc[splits["val_indices"]]
        test  = data.loc[splits["test_indices"]]
        return train, val, test

    def _make_loader(self, df: pd.DataFrame, shuffle: bool = False) -> DataLoader:
        texts = df["cleaned_text"].tolist()
        labels = df["label"].tolist()
        dataset = FraudDataset(texts, labels, self.tokenizer, self.cfg["max_length"])
        return DataLoader(dataset, batch_size=self.cfg["batch_size"], shuffle=shuffle)

    # ----------------------------------------------------------------- train
    def train(self):
        train_df, val_df, test_df = self._load_splits()
        train_loader = self._make_loader(train_df, shuffle=True)
        val_loader   = self._make_loader(val_df)
        test_loader  = self._make_loader(test_df)

        # Class weights to handle imbalance
        fraud_ratio = train_df["label"].mean()
        weights = torch.tensor(
            [fraud_ratio, 1 - fraud_ratio], dtype=torch.float
        ).to(self.device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg["learning_rate"],
            weight_decay=self.cfg["weight_decay"],
        )
        total_steps = len(train_loader) * self.cfg["num_epochs"]
        warmup_steps = int(total_steps * self.cfg["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        best_val_f1 = 0.0

        for epoch in range(1, self.cfg["num_epochs"] + 1):
            # -- Training
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                logits = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
                loss = loss_fn(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # -- Validation
            val_f1, val_report = self._evaluate(val_loader)
            logger.info(
                f"Epoch {epoch}/{self.cfg['num_epochs']} | "
                f"Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self._save_model()
                logger.info(f"  ✓ New best model saved (F1={best_val_f1:.4f})")

        # -- Test evaluation on best checkpoint
        self._load_model()
        test_f1, test_report = self._evaluate(test_loader)
        logger.info(f"\n=== TEST RESULTS ===\n{test_report}")
        return test_report

    # --------------------------------------------------------------- helpers
    def _evaluate(self, loader: DataLoader):
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].cpu().numpy()

                logits = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                preds = logits.argmax(dim=-1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)
                all_probs.extend(probs)

        report = classification_report(all_labels, all_preds,
                                       target_names=["Legit", "Fraud"])
        _, _, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="binary", pos_label=1
        )
        return f1, report

    def _save_model(self):
        os.makedirs(os.path.dirname(self.cfg["output_model_path"]), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.cfg,
            },
            self.cfg["output_model_path"],
        )

    def _load_model(self):
        checkpoint = torch.load(self.cfg["output_model_path"], map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DistilBERT fraud classifier.")
    parser.add_argument("--processed-dir", default=DEFAULT_CFG["processed_dir"])
    parser.add_argument("--output-model", default=DEFAULT_CFG["output_model_path"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CFG["num_epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CFG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CFG["learning_rate"])
    args = parser.parse_args()

    cfg = {**DEFAULT_CFG}
    cfg.update({
        "processed_dir": args.processed_dir,
        "output_model_path": args.output_model,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
    })

    trainer = FraudClassifierTrainer(cfg)
    trainer.train()
