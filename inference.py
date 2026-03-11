"""
inference.py
Role A – AI/NLP Fraud Detection
End-to-end inference pipeline: preprocessing → DistilBERT classifier →
optional LLM fallback → explainability → returns a structured result dict.
"""

import logging
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

from preprocessing import clean_message
from fraud_classifier import FraudClassifier, FraudPrediction
from llm_reasoning import LLMReasoner
from explainability_engine import ExplainabilityEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    raw_text: str
    cleaned_text: str
    label: int
    label_str: str
    confidence: float
    fraud_probability: float
    legit_probability: float
    llm_used: bool
    llm_reasoning: Optional[str]
    highlighted_spans: List[Dict[str, Any]]   # from explainability engine
    latency_ms: float


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------

class FraudInferencePipeline:
    """
    Orchestrates the full inference flow for a single or batch of messages.

    Args:
        classifier: Loaded FraudClassifier instance.
        reasoner: LLMReasoner instance for fallback (can be None to disable).
        explainer: ExplainabilityEngine instance (can be None to disable).
        use_llm_fallback: If False, LLM fallback is never called.
    """

    def __init__(
        self,
        classifier: Optional[FraudClassifier] = None,
        reasoner: Optional[LLMReasoner] = None,
        explainer: Optional[ExplainabilityEngine] = None,
        use_llm_fallback: bool = True,
    ):
        self.classifier = classifier or FraudClassifier()
        self.reasoner = reasoner
        self.explainer = explainer
        self.use_llm_fallback = use_llm_fallback

        if use_llm_fallback and reasoner is None:
            logger.info("LLM fallback enabled — initialising LLMReasoner.")
            self.reasoner = LLMReasoner()

        if explainer is None:
            self.explainer = ExplainabilityEngine()

    # ---------------------------------------------------------------- public

    def run(self, text: str) -> InferenceResult:
        """Run the full pipeline on a single message."""
        t0 = time.perf_counter()

        cleaned = clean_message(text)
        prediction: FraudPrediction = self.classifier.predict(cleaned)

        llm_used = False
        llm_reasoning = None

        # LLM fallback for low-confidence or complex social-engineering cases
        if self.use_llm_fallback and prediction.flagged_for_llm and self.reasoner:
            llm_result = self.reasoner.reason(cleaned)
            llm_used = True
            llm_reasoning = llm_result.get("reasoning")
            # Override label with LLM verdict if available
            llm_label = llm_result.get("label")
            if llm_label is not None:
                prediction.label = llm_label
                prediction.label_str = "Fraud" if llm_label == 1 else "Legit"

        # Explainability
        highlighted_spans = []
        if self.explainer:
            highlighted_spans = self.explainer.highlight(
                cleaned, prediction.label
            )

        latency_ms = (time.perf_counter() - t0) * 1000

        return InferenceResult(
            raw_text=text,
            cleaned_text=cleaned,
            label=prediction.label,
            label_str=prediction.label_str,
            confidence=prediction.confidence,
            fraud_probability=prediction.fraud_probability,
            legit_probability=prediction.legit_probability,
            llm_used=llm_used,
            llm_reasoning=llm_reasoning,
            highlighted_spans=highlighted_spans,
            latency_ms=round(latency_ms, 2),
        )

    def run_batch(self, texts: List[str]) -> List[InferenceResult]:
        """Run pipeline on a list of messages."""
        return [self.run(t) for t in texts]

    def to_dict(self, result: InferenceResult) -> Dict[str, Any]:
        return asdict(result)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_pipeline(
    model_path: str = "models/fraud_classifier.pt",
    confidence_threshold: float = 0.75,
    use_llm_fallback: bool = True,
) -> FraudInferencePipeline:
    classifier = FraudClassifier(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
    )
    return FraudInferencePipeline(
        classifier=classifier,
        use_llm_fallback=use_llm_fallback,
    )


# ---------------------------------------------------------------------------
# CLI / quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Run fraud inference on a message.")
    parser.add_argument("--message", required=True, help="Message text to classify.")
    parser.add_argument("--model-path", default="models/fraud_classifier.pt")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    pipeline = build_pipeline(
        model_path=args.model_path,
        use_llm_fallback=not args.no_llm,
    )
    result = pipeline.run(args.message)
    print(json.dumps(pipeline.to_dict(result), indent=2))
