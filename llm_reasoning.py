"""
llm_reasoning.py
Role A – AI/NLP Fraud Detection
Integrates Claude (Anthropic) as a zero-shot LLM fallback for complex social
engineering messages where the DistilBERT classifier is uncertain.
"""

import os
import json
import logging
import re
import anthropic
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert in financial fraud detection, specialising in
SMS phishing, UPI scams, bank impersonation, job scams, and social engineering.

Analyse the provided message and return ONLY a valid JSON object (no markdown,
no explanation outside the JSON) with the following keys:

{
  "label": 0 or 1,           // 0 = Legitimate, 1 = Fraud/Scam
  "confidence": 0.0 to 1.0,  // how confident you are
  "reasoning": "string",     // concise explanation of your verdict
  "red_flags": ["list", "of", "specific", "red", "flags"],
  "scam_type": "string or null"  // e.g. "UPI fraud", "phishing", "job scam", "bank impersonation"
}

Be especially alert to:
- Urgency cues designed to bypass critical thinking
- Requests for OTP, PIN, CVV, Aadhaar, PAN
- Impersonation of Indian banks (SBI, HDFC, ICICI, etc.) or government bodies
- Fake prize/lottery announcements
- Courier / parcel delivery scams
- Fake KYC update requests
"""


# ---------------------------------------------------------------------------
# Reasoner
# ---------------------------------------------------------------------------

class LLMReasoner:
    """
    Uses Claude as a zero-shot fraud reasoning engine.

    The output of reason() is a dict with keys:
        label (int), confidence (float), reasoning (str),
        red_flags (list), scam_type (str | None)
    """

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        max_tokens: int = 512,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)

    # ---------------------------------------------------------------- public

    def reason(self, message: str) -> Dict[str, Any]:
        """
        Send message to Claude for zero-shot fraud classification.

        Returns:
            Parsed result dict, or a default UNCERTAIN dict on failure.
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": f"Message to analyse:\n\n{message}"}
                ],
            )
            raw_text = response.content[0].text.strip()
            return self._parse_response(raw_text)

        except anthropic.APIConnectionError as e:
            logger.error(f"LLM API connection error: {e}")
        except anthropic.RateLimitError as e:
            logger.warning(f"LLM rate limit hit: {e}")
        except Exception as e:
            logger.error(f"Unexpected LLM error: {e}")

        return self._uncertain_result(message)

    def reason_batch(self, messages: list) -> list:
        return [self.reason(m) for m in messages]

    # --------------------------------------------------------------- private

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        # Strip any accidental markdown fences
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        try:
            result = json.loads(raw)
            # Validate required keys
            result.setdefault("label", 1)
            result.setdefault("confidence", 0.5)
            result.setdefault("reasoning", "No reasoning provided.")
            result.setdefault("red_flags", [])
            result.setdefault("scam_type", None)
            result["label"] = int(result["label"])
            result["confidence"] = float(result["confidence"])
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"LLM returned non-JSON: {raw[:200]}… Error: {e}")
            return self._uncertain_result(raw)

    @staticmethod
    def _uncertain_result(text: str) -> Dict[str, Any]:
        return {
            "label": 1,          # default to Fraud when uncertain
            "confidence": 0.5,
            "reasoning": "LLM reasoning unavailable — defaulting to Fraud.",
            "red_flags": [],
            "scam_type": None,
        }
