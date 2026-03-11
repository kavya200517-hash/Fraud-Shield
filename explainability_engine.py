"""
explainability_engine.py
Role A – AI/NLP Fraud Detection
Highlights red-flag tokens in message text using gradient-based attribution
(Integrated Gradients) and a keyword fallback, providing human-readable
explanations for each fraud verdict.
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Red-flag lexicon (used as fast keyword fallback)
# ---------------------------------------------------------------------------

RED_FLAG_LEXICON = {
    "urgency": [
        "urgent", "immediately", "act now", "expires", "last chance",
        "within 24 hours", "right now", "asap",
    ],
    "credential_theft": [
        "otp", "pin", "cvv", "password", "aadhar", "pan", "account number",
        "ifsc", "share your", "do not share",
    ],
    "impersonation": [
        "sbi", "hdfc", "icici", "axis bank", "paytm", "rbi", "uidai",
        "reserve bank", "nsdl", "npci",
    ],
    "lure": [
        "prize", "winner", "congratulations", "reward", "cashback",
        "free", "lottery", "selected", "lucky",
    ],
    "threat": [
        "blocked", "suspended", "closed", "deactivated", "action required",
        "kyc update", "verify now", "final notice",
    ],
    "url_suspicious": [
        "<url>", "http", "www.", "click here", "link below",
    ],
}

# Flatten for fast lookup
_ALL_RED_FLAGS: List[Tuple[str, str]] = [
    (token, category)
    for category, tokens in RED_FLAG_LEXICON.items()
    for token in tokens
]


# ---------------------------------------------------------------------------
# Span dataclass
# ---------------------------------------------------------------------------

@dataclass
class HighlightedSpan:
    text: str
    start: int
    end: int
    category: str
    importance_score: float      # 0–1; higher = more suspicious


# ---------------------------------------------------------------------------
# Explainability engine
# ---------------------------------------------------------------------------

class ExplainabilityEngine:
    """
    Produces token-level importance scores and keyword highlights.

    Strategy:
        1. Try gradient × input attribution if model weights are available.
        2. Fall back to fast lexicon-based keyword highlighting.
    """

    def __init__(
        self,
        model: Optional[DistilBertForSequenceClassification] = None,
        tokenizer: Optional[DistilBertTokenizerFast] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)

    # ---------------------------------------------------------------- public

    def highlight(
        self,
        text: str,
        predicted_label: int,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Return a list of highlighted span dicts for a given text.

        Args:
            text: Cleaned message string.
            predicted_label: 0 (Legit) or 1 (Fraud) — used for gradient direction.
            top_k: Max number of gradient-attributed tokens to return.

        Returns:
            List of dicts with keys: text, start, end, category, importance_score.
        """
        if self.model and self.tokenizer:
            spans = self._gradient_attribution(text, predicted_label, top_k)
        else:
            spans = self._lexicon_highlight(text)

        return [
            {
                "text": s.text,
                "start": s.start,
                "end": s.end,
                "category": s.category,
                "importance_score": round(s.importance_score, 4),
            }
            for s in spans
        ]

    def explain_text(self, text: str, predicted_label: int) -> str:
        """
        Return a human-readable explanation string.
        """
        spans = self.highlight(text, predicted_label)
        if not spans:
            return "No specific red flags detected."

        lines = ["Red flags identified:"]
        for span in sorted(spans, key=lambda x: -x["importance_score"])[:5]:
            lines.append(
                f"  • \"{span['text']}\" ({span['category']}, "
                f"score={span['importance_score']:.2f})"
            )
        return "\n".join(lines)

    # --------------------------------------------------------------- private

    def _lexicon_highlight(self, text: str) -> List[HighlightedSpan]:
        """Fast keyword-based fallback."""
        text_lower = text.lower()
        spans: List[HighlightedSpan] = []
        seen_ranges = set()

        for token, category in _ALL_RED_FLAGS:
            for match in re.finditer(re.escape(token), text_lower):
                rng = (match.start(), match.end())
                if rng in seen_ranges:
                    continue
                seen_ranges.add(rng)
                importance = self._category_weight(category)
                spans.append(
                    HighlightedSpan(
                        text=text[match.start():match.end()],
                        start=match.start(),
                        end=match.end(),
                        category=category,
                        importance_score=importance,
                    )
                )
        return spans

    def _gradient_attribution(
        self, text: str, label: int, top_k: int
    ) -> List[HighlightedSpan]:
        """Integrated Gradients (simplified) for token-level attribution."""
        try:
            self.model.eval()
            self.model.to(self.device)

            encoding = self.tokenizer(
                text, return_tensors="pt", truncation=True,
                padding="max_length", max_length=128,
            )
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            # Get token embeddings
            embeddings = self.model.distilbert.embeddings(input_ids)
            embeddings = embeddings.detach().requires_grad_(True)

            output = self.model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
            )
            logits = output.logits
            score = logits[0, label]
            score.backward()

            # Grad × input
            gradients = embeddings.grad[0].detach().cpu().numpy()  # (seq_len, hidden)
            token_scores = np.abs(gradients).mean(axis=-1)         # (seq_len,)

            # Map tokens back to subword strings
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            word_ids = encoding.word_ids(batch_index=0)

            top_indices = np.argsort(token_scores)[::-1][:top_k]
            spans = []
            for idx in top_indices:
                if tokens[idx] in ("[PAD]", "[CLS]", "[SEP]"):
                    continue
                token_text = tokens[idx].replace("##", "")
                importance = float(token_scores[idx]) / (float(token_scores.max()) + 1e-8)
                spans.append(
                    HighlightedSpan(
                        text=token_text,
                        start=idx,
                        end=idx + len(token_text),
                        category="model_attribution",
                        importance_score=importance,
                    )
                )
            return spans

        except Exception as e:
            logger.warning(f"Gradient attribution failed ({e}), falling back to lexicon.")
            return self._lexicon_highlight(text)

    @staticmethod
    def _category_weight(category: str) -> float:
        weights = {
            "credential_theft": 0.95,
            "threat": 0.85,
            "urgency": 0.75,
            "impersonation": 0.70,
            "url_suspicious": 0.65,
            "lure": 0.60,
        }
        return weights.get(category, 0.50)
