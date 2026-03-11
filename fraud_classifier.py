"""
fraud_classifier.py
Role A – AI/NLP Fraud Detection
Wrapper around the trained DistilBERT model, exposing a clean predict interface
with confidence scores and class labels.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prediction result container
# ---------------------------------------------------------------------------

@dataclass
class FraudPrediction:
    text: str
    label: int                          # 0 = Legit, 1 = Fraud
    label_str: str                      # "Legit" | "Fraud"
    confidence: float                   # probability of predicted class
    fraud_probability: float            # P(Fraud)
    legit_probability: float            # P(Legit)
    flagged_for_llm: bool = False       # True when confidence < threshold → LLM fallback
    explanation: Optional[str] = None   # populated by explainability engine


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class FraudClassifier:
    """
    Loads a fine-tuned DistilBERT model and provides batch inference.

    Args:
        model_path: Path to saved checkpoint (.pt).
        model_name: HuggingFace model string (must match training).
        max_length: Tokeniser max sequence length.
        confidence_threshold: Predictions below this threshold are flagged
                              for LLM fallback reasoning.
        device: 'cuda' | 'cpu' | 'auto'.
    """

    LABEL_MAP = {0: "Legit", 1: "Fraud"}

    def __init__(
        self,
        model_path: str = "models/fraud_classifier.pt",
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        confidence_threshold: float = 0.75,
        device: str = "auto",
    ):
        self.max_length = max_length
        self.confidence_threshold = confidence_threshold

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Loading FraudClassifier on {self.device} from {model_path}")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self._load_weights(model_path)
        self.model.to(self.device)
        self.model.eval()

    # ---------------------------------------------------------------- public

    def predict(self, text: str) -> FraudPrediction:
        """Predict a single message."""
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: List[str]) -> List[FraudPrediction]:
        """
        Predict a list of messages.

        Returns:
            List of FraudPrediction objects in the same order as input.
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()   # (B, 2)

        predictions = []
        for i, text in enumerate(texts):
            legit_p, fraud_p = float(probs[i, 0]), float(probs[i, 1])
            label = int(np.argmax(probs[i]))
            confidence = max(legit_p, fraud_p)

            predictions.append(
                FraudPrediction(
                    text=text,
                    label=label,
                    label_str=self.LABEL_MAP[label],
                    confidence=confidence,
                    fraud_probability=fraud_p,
                    legit_probability=legit_p,
                    flagged_for_llm=confidence < self.confidence_threshold,
                )
            )
        return predictions

    def is_fraud(self, text: str) -> bool:
        """Quick boolean check."""
        return self.predict(text).label == 1

    # --------------------------------------------------------------- private

    def _load_weights(self, model_path: str) -> None:
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
            logger.info("Model weights loaded successfully.")
        except FileNotFoundError:
            logger.warning(
                f"Model file not found at {model_path}. "
                "Using untrained weights — run train_model.py first."
            )
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            raise
