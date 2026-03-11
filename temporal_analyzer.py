"""
temporal_analyzer.py
Role A – AI/NLP Fraud Detection
Detects coordinated, multi-stage fraud sequences using a sliding time window.
Tracks message patterns per sender/device over time and raises alerts when
a sequence matches known multi-step attack signatures.
"""

import time
import logging
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_WINDOW_SECONDS = 3600        # 1-hour sliding window
DEFAULT_ALERT_THRESHOLD = 3          # min fraud messages in window to alert
DEFAULT_ESCALATION_THRESHOLD = 5     # escalate to high-severity above this


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MessageEvent:
    sender_id: str
    message: str
    cleaned_text: str
    label: int                        # 0=Legit, 1=Fraud
    fraud_probability: float
    timestamp: float = field(default_factory=time.time)
    scam_type: Optional[str] = None


@dataclass
class TemporalAlert:
    sender_id: str
    severity: str                     # "medium" | "high" | "critical"
    fraud_count: int
    window_seconds: float
    sequence_pattern: List[str]       # ordered scam_type labels
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    message: str = ""


# ---------------------------------------------------------------------------
# Multi-stage attack signatures
# ---------------------------------------------------------------------------

MULTI_STAGE_SIGNATURES = [
    {
        "name": "Classic UPI Takeover",
        "sequence": ["bank_impersonation", "kyc_urgency", "otp_theft"],
        "description": "Initial bank contact → urgency escalation → OTP collection",
    },
    {
        "name": "Job Scam Funnel",
        "sequence": ["job_offer", "document_request", "payment_demand"],
        "description": "Attractive offer → personal data collection → money extraction",
    },
    {
        "name": "Courier Parcel Scam",
        "sequence": ["parcel_notification", "fee_demand", "credential_theft"],
        "description": "Fake delivery notice → small payment → full account access",
    },
    {
        "name": "Phishing Escalation",
        "sequence": ["url_phishing", "credential_theft"],
        "description": "Phishing link → credential harvesting",
    },
]


# ---------------------------------------------------------------------------
# Temporal Analyzer
# ---------------------------------------------------------------------------

class TemporalAnalyzer:
    """
    Maintains a per-sender sliding window of message events and detects
    coordinated multi-stage fraud patterns.

    Usage:
        analyzer = TemporalAnalyzer()
        alert = analyzer.process(event)
        if alert:
            handle_alert(alert)
    """

    def __init__(
        self,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        alert_threshold: int = DEFAULT_ALERT_THRESHOLD,
        escalation_threshold: int = DEFAULT_ESCALATION_THRESHOLD,
    ):
        self.window_seconds = window_seconds
        self.alert_threshold = alert_threshold
        self.escalation_threshold = escalation_threshold

        # sender_id → deque of MessageEvents
        self._windows: Dict[str, Deque[MessageEvent]] = defaultdict(deque)
        self._alert_history: List[TemporalAlert] = []

    # ---------------------------------------------------------------- public

    def process(self, event: MessageEvent) -> Optional[TemporalAlert]:
        """
        Ingest a new message event and return an alert if a pattern is detected.

        Args:
            event: MessageEvent to process.

        Returns:
            TemporalAlert if a fraud pattern is detected, else None.
        """
        self._ingest(event)
        return self._check_patterns(event.sender_id)

    def get_sender_history(self, sender_id: str) -> List[MessageEvent]:
        """Return current window of events for a sender."""
        self._evict_stale(sender_id)
        return list(self._windows[sender_id])

    def get_fraud_rate(self, sender_id: str) -> float:
        """Fraction of recent messages classified as fraud."""
        events = self.get_sender_history(sender_id)
        if not events:
            return 0.0
        return sum(1 for e in events if e.label == 1) / len(events)

    def alerts(self) -> List[TemporalAlert]:
        return list(self._alert_history)

    def reset_sender(self, sender_id: str) -> None:
        """Clear the window for a sender (e.g., after incident handling)."""
        self._windows.pop(sender_id, None)

    # --------------------------------------------------------------- private

    def _ingest(self, event: MessageEvent) -> None:
        self._windows[event.sender_id].append(event)
        self._evict_stale(event.sender_id)

    def _evict_stale(self, sender_id: str) -> None:
        cutoff = time.time() - self.window_seconds
        window = self._windows[sender_id]
        while window and window[0].timestamp < cutoff:
            window.popleft()

    def _check_patterns(self, sender_id: str) -> Optional[TemporalAlert]:
        events = list(self._windows[sender_id])
        fraud_events = [e for e in events if e.label == 1]
        fraud_count = len(fraud_events)

        if fraud_count < self.alert_threshold:
            return None

        # Determine severity
        if fraud_count >= self.escalation_threshold:
            severity = "critical"
        elif fraud_count >= self.alert_threshold + 2:
            severity = "high"
        else:
            severity = "medium"

        # Build scam-type sequence
        sequence = [e.scam_type or "unknown" for e in fraud_events]

        # Check for known multi-stage signatures
        matched_sig = self._match_signature(sequence)
        if matched_sig:
            severity = "critical"
            msg = (
                f"Multi-stage attack detected: {matched_sig['name']}. "
                f"{matched_sig['description']}"
            )
        else:
            msg = (
                f"{fraud_count} fraud messages detected from {sender_id} "
                f"within {self.window_seconds / 60:.0f} minutes."
            )

        alert = TemporalAlert(
            sender_id=sender_id,
            severity=severity,
            fraud_count=fraud_count,
            window_seconds=self.window_seconds,
            sequence_pattern=sequence,
            message=msg,
        )
        self._alert_history.append(alert)
        logger.warning(f"[TemporalAlert] {severity.upper()} | {msg}")
        return alert

    @staticmethod
    def _match_signature(sequence: List[str]) -> Optional[Dict]:
        """
        Check if the observed scam-type sequence matches a known multi-stage
        attack signature (order-agnostic subset match).
        """
        seq_set = set(sequence)
        for sig in MULTI_STAGE_SIGNATURES:
            if set(sig["sequence"]).issubset(seq_set):
                return sig
        return None


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = TemporalAnalyzer(window_seconds=300, alert_threshold=2)

    test_events = [
        MessageEvent("sender_001", "Your SBI account is blocked!", "sbi account blocked",
                     1, 0.92, scam_type="bank_impersonation"),
        MessageEvent("sender_001", "Update KYC immediately or lose access", "update kyc immediately",
                     1, 0.88, scam_type="kyc_urgency"),
        MessageEvent("sender_001", "Enter OTP to restore account", "enter otp restore account",
                     1, 0.95, scam_type="otp_theft"),
    ]

    for ev in test_events:
        alert = analyzer.process(ev)
        if alert:
            print(f"\n🚨 ALERT [{alert.severity.upper()}]: {alert.message}")
            print(f"   Sequence: {alert.sequence_pattern}")
