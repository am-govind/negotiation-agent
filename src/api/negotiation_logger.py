"""
Negotiation Session Logger: Persist completed negotiations to JSON files.

Tracks session_id, category, timestamps, rounds, pricing, outcome, and messages
for the admin analytics dashboard.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import NEGOTIATION_LOG_DIR

logger = logging.getLogger(__name__)


class NegotiationLogger:
    """Persist negotiation sessions to JSON for analytics."""

    def __init__(self):
        self._log_dir = NEGOTIATION_LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def log_session(
        self,
        session_id: str,
        state: dict[str, Any],
        messages: list[dict],
        outcome: str = "unknown",
    ) -> Path:
        """Write a completed negotiation session to disk.

        Args:
            session_id: Unique session identifier.
            state: Final NegotiationState dict.
            messages: List of {role, content} message dicts.
            outcome: "closed", "abandoned", or "active".

        Returns:
            Path to the saved JSON file.
        """
        record = {
            "session_id": session_id,
            "product_category": state.get("product_category", ""),
            "timestamp": datetime.utcnow().isoformat(),
            "negotiation_round": state.get("negotiation_round", 0),
            "target_price": state.get("target_price", 0),
            "floor_price": state.get("floor_price", 0),
            "opening_price": state.get("opening_price", 0),
            "final_price": state.get("current_offer", 0),
            "outcome": outcome,
            "deal_closed": state.get("deal_closed", False),
            "deal_abandoned": state.get("deal_abandoned", False),
            "intent": state.get("intent", ""),
            "value_adds_offered": state.get("value_adds_offered", []),
            "messages": messages,
        }

        filepath = self._log_dir / f"{session_id}.json"
        with open(filepath, "w") as f:
            json.dump(record, f, indent=2, default=str)

        logger.info(f"Session logged to {filepath.name}")
        return filepath

    def load_all_sessions(self) -> list[dict]:
        """Load all logged sessions for analytics."""
        sessions = []
        for fp in sorted(self._log_dir.glob("*.json"), reverse=True):
            try:
                with open(fp) as f:
                    sessions.append(json.load(f))
            except Exception as e:
                logger.warning(f"Could not load {fp.name}: {e}")
        return sessions

    def get_analytics(self) -> dict:
        """Compute aggregate analytics from all logged sessions."""
        sessions = self.load_all_sessions()
        if not sessions:
            return {
                "total_negotiations": 0,
                "deals_closed": 0,
                "deals_abandoned": 0,
                "win_rate_pct": 0.0,
                "avg_margin_retained_pct": 0.0,
                "avg_rounds": 0.0,
                "avg_discount_pct": 0.0,
                "total_revenue": 0.0,
                "sessions": [],
            }

        closed = [s for s in sessions if s.get("deal_closed")]
        abandoned = [s for s in sessions if s.get("deal_abandoned")]
        total = len(sessions)

        margins = []
        discounts = []
        revenue = 0.0
        for s in closed:
            target = s.get("target_price", 0)
            floor = s.get("floor_price", 0)
            final = s.get("final_price", 0)
            margin_range = target - floor
            if margin_range > 0:
                margins.append(((final - floor) / margin_range) * 100)
            if target > 0:
                discounts.append(((target - final) / target) * 100)
            revenue += final

        return {
            "total_negotiations": total,
            "deals_closed": len(closed),
            "deals_abandoned": len(abandoned),
            "win_rate_pct": round(len(closed) / total * 100, 1) if total > 0 else 0,
            "avg_margin_retained_pct": round(sum(margins) / len(margins), 1) if margins else 0,
            "avg_rounds": round(sum(s.get("negotiation_round", 0) for s in sessions) / total, 1),
            "avg_discount_pct": round(sum(discounts) / len(discounts), 1) if discounts else 0,
            "total_revenue": round(revenue, 2),
            "sessions": [
                {
                    "session_id": s["session_id"],
                    "category": s.get("product_category", ""),
                    "outcome": s.get("outcome", ""),
                    "rounds": s.get("negotiation_round", 0),
                    "final_price": s.get("final_price", 0),
                    "timestamp": s.get("timestamp", ""),
                }
                for s in sessions[:50]  # last 50 sessions
            ],
        }


# ── Module singleton ──────────────────────────────────────────
_logger: NegotiationLogger | None = None


def get_negotiation_logger() -> NegotiationLogger:
    global _logger
    if _logger is None:
        _logger = NegotiationLogger()
    return _logger
