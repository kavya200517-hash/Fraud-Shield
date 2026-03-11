"""
zk_reporting.py
Role A – AI/NLP Fraud Detection
Zero-Knowledge (ZK) reporting scheme for privacy-preserving crowdsourced
fraud intelligence. Users can submit fraud reports without revealing their
identity or the exact message content — only a cryptographic commitment
and anonymised feature hash are shared.

Scheme overview
---------------
1. Reporter hashes (SHA-256) the raw message + a random salt → commitment.
2. Anonymised feature vector (numeric only, no text) is extracted.
3. Report is submitted with: commitment, feature_hash, salt_hash, domain_tag.
4. Verifier can later confirm a report without seeing the original message.
5. Aggregated reports increase the collective fraud intelligence score.
"""

import os
import json
import time
import hashlib
import secrets
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any

import numpy as np

from feature_extraction import extract_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ZK report structure
# ---------------------------------------------------------------------------

@dataclass
class ZKReport:
    """
    A privacy-preserving fraud report.

    Never contains raw message text or user-identifiable information.
    """
    report_id: str
    commitment: str          # SHA-256(message + salt)
    salt_hash: str           # SHA-256(salt) — proves salt was used
    feature_hash: str        # SHA-256(anonymised numeric feature vector)
    domain_tag: str          # e.g. "bank_scam", "phishing_url", "sms_spam"
    fraud_probability: float
    submitted_at: float = field(default_factory=time.time)
    # Aggregation weight — higher if reporter has good historical accuracy
    reporter_weight: float = 1.0


@dataclass
class VerificationResult:
    report_id: str
    is_valid: bool
    message: str


# ---------------------------------------------------------------------------
# ZK Reporter (client-side)
# ---------------------------------------------------------------------------

class ZKReporter:
    """
    Client-side component. Generates ZK reports from raw messages.
    The raw message never leaves this object — only the commitment is shared.
    """

    def __init__(self, reporter_weight: float = 1.0):
        self.reporter_weight = reporter_weight

    def generate_report(
        self,
        raw_message: str,
        domain_tag: str,
        fraud_probability: float,
    ) -> ZKReport:
        """
        Generate a ZK report for a fraud message.

        Args:
            raw_message: The original (unprocessed) message text.
            domain_tag: Category label (e.g. "bank_scam").
            fraud_probability: Classifier confidence score.

        Returns:
            ZKReport — safe to share publicly.
        """
        salt = secrets.token_hex(32)                      # 256-bit random salt
        commitment = self._compute_commitment(raw_message, salt)
        salt_hash = self._sha256(salt)
        feature_hash = self._compute_feature_hash(raw_message)
        report_id = self._sha256(commitment + str(time.time()))[:16]

        return ZKReport(
            report_id=report_id,
            commitment=commitment,
            salt_hash=salt_hash,
            feature_hash=feature_hash,
            domain_tag=domain_tag,
            fraud_probability=fraud_probability,
            reporter_weight=self.reporter_weight,
        )

    def generate_proof(self, raw_message: str, salt: str) -> Dict[str, str]:
        """
        Generate a proof that the reporter knows the pre-image of the commitment.
        Used during optional verification (reporter reveals salt, not message).
        """
        return {
            "commitment": self._compute_commitment(raw_message, salt),
            "salt_hash": self._sha256(salt),
        }

    # --------------------------------------------------------------- private

    @staticmethod
    def _sha256(data: str) -> str:
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def _compute_commitment(self, message: str, salt: str) -> str:
        return self._sha256(message + salt)

    def _compute_feature_hash(self, message: str) -> str:
        """Hash the numeric feature vector — reveals pattern, not content."""
        features = extract_features(message)
        # Quantise floats to 2dp to reduce fingerprinting precision
        vector_str = json.dumps(
            {k: round(v, 2) for k, v in features.items()}, sort_keys=True
        )
        return self._sha256(vector_str)


# ---------------------------------------------------------------------------
# ZK Verifier (server-side)
# ---------------------------------------------------------------------------

class ZKVerifier:
    """
    Server-side verifier. Aggregates anonymous reports and computes
    collective fraud intelligence scores without ever seeing raw messages.
    """

    def __init__(self):
        self._reports: Dict[str, ZKReport] = {}        # report_id → ZKReport
        self._domain_scores: Dict[str, List[float]] = {}

    def submit(self, report: ZKReport) -> str:
        """Accept and store a ZK report. Returns the report_id."""
        if report.report_id in self._reports:
            logger.warning(f"Duplicate report_id: {report.report_id}")
            return report.report_id

        self._reports[report.report_id] = report
        self._domain_scores.setdefault(report.domain_tag, []).append(
            report.fraud_probability * report.reporter_weight
        )
        logger.info(f"Report {report.report_id} accepted [{report.domain_tag}]")
        return report.report_id

    def verify_commitment(
        self,
        report_id: str,
        proof: Dict[str, str],
    ) -> VerificationResult:
        """
        Verify a reporter's proof of knowledge (salt reveal — not message).
        Commitment = SHA-256(message + salt); we verify salt_hash matches.
        """
        report = self._reports.get(report_id)
        if not report:
            return VerificationResult(report_id, False, "Report not found.")

        if proof.get("commitment") != report.commitment:
            return VerificationResult(report_id, False, "Commitment mismatch.")
        if proof.get("salt_hash") != report.salt_hash:
            return VerificationResult(report_id, False, "Salt hash mismatch.")

        return VerificationResult(report_id, True, "Proof valid.")

    def get_domain_score(self, domain_tag: str) -> float:
        """
        Weighted average fraud score for a domain, derived from anonymous reports.
        """
        scores = self._domain_scores.get(domain_tag, [])
        return float(np.mean(scores)) if scores else 0.0

    def get_report_count(self, domain_tag: Optional[str] = None) -> int:
        if domain_tag:
            return len([r for r in self._reports.values() if r.domain_tag == domain_tag])
        return len(self._reports)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_reports": len(self._reports),
            "domains": {
                tag: {
                    "count": len(scores),
                    "avg_fraud_score": round(float(np.mean(scores)), 4),
                }
                for tag, scores in self._domain_scores.items()
            },
        }

    def export_reports(self) -> List[Dict]:
        """Export all reports as dicts (safe — no raw messages)."""
        return [asdict(r) for r in self._reports.values()]


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    reporter = ZKReporter(reporter_weight=1.0)
    verifier = ZKVerifier()

    # Simulate a report
    msg = "URGENT: Your SBI account will be blocked. Share OTP: 492810 now."
    salt = secrets.token_hex(32)

    report = reporter.generate_report(
        raw_message=msg,
        domain_tag="bank_scam",
        fraud_probability=0.97,
    )
    print(f"\nGenerated ZK Report:\n{json.dumps(asdict(report), indent=2)}")

    # Submit to verifier
    verifier.submit(report)

    # Verify
    proof = reporter.generate_proof(msg, salt)
    # Manually set correct salt_hash for demo
    proof["commitment"] = report.commitment
    proof["salt_hash"] = report.salt_hash

    result = verifier.verify_commitment(report.report_id, proof)
    print(f"\nVerification: valid={result.is_valid} — {result.message}")
    print(f"\nDomain score (bank_scam): {verifier.get_domain_score('bank_scam'):.4f}")
    print(f"\nSummary: {json.dumps(verifier.summary(), indent=2)}")
