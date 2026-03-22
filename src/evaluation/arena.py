"""
Multi-Agent Arena: Automated negotiation simulation harness (Improvisation 3).

Orchestrates N conversations between the seller agent and buyer personas,
collecting metrics for the analytics dashboard.

Usage:
    python -m src.evaluation.arena --runs 10 --personas aggressive value urgent
"""
import logging
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from src.utils.llm import get_buyer_llm
from langchain_core.messages import SystemMessage, HumanMessage

from src.agent.state import create_initial_state
from src.agent.graph import run_negotiation_turn
from src.ml.price_calculator import get_calculator
from src.evaluation.buyer_personas import (
    ALL_PERSONAS, get_persona, BuyerPersona,
)
from src.config import ARENA_DIR
from src.utils.gemini import extract_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class NegotiationArena:
    """Automated multi-agent simulation arena."""

    def __init__(self):
        self.buyer_llm = get_buyer_llm()
        self.calculator = get_calculator()
        self.results: list[dict] = []

    def _generate_buyer_response(
        self,
        persona: BuyerPersona,
        conversation_history: list[dict],
        seller_message: str,
    ) -> str:
        """Generate a buyer response using the persona's LLM."""
        messages = [SystemMessage(content=persona.system_prompt)]

        for msg in conversation_history:
            if msg["role"] == "buyer":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(HumanMessage(content=f"[Seller]: {msg['content']}"))

        messages.append(HumanMessage(content=f"[Seller]: {seller_message}\n\nRespond as the buyer:"))

        response = self.buyer_llm.invoke(messages)
        return extract_text(response.content).strip()

    def _detect_deal_outcome(self, buyer_response: str) -> str:
        """Detect if the buyer accepted, rejected, or is continuing."""
        lower = buyer_response.lower()

        accept_signals = [
            "i'll take it", "deal", "i'll buy", "sounds good", "let's do it",
            "i accept", "sold", "great package", "let's get this shipped",
            "fine, i'll take", "agreed",
        ]
        reject_signals = [
            "no deal", "i'm leaving", "forget it", "i'm going to",
            "i'll think about it", "not interested", "too slow",
            "i'll find it elsewhere", "pass",
        ]

        for signal in accept_signals:
            if signal in lower:
                return "accepted"

        for signal in reject_signals:
            if signal in lower:
                return "rejected"

        return "continuing"

    def run_single_negotiation(
        self,
        persona: BuyerPersona,
        product_category: str = "computers_accessories",
        customer_state: str = "SP",
        run_id: int = 0,
    ) -> dict:
        """Run a single buyer-seller negotiation and return metrics."""
        logger.info(f"\n{'='*60}")
        logger.info(f"  Run {run_id} | Persona: {persona.name} | Product: {product_category}")
        logger.info(f"{'='*60}")

        # Get ML pricing
        pricing = self.calculator.get_optimal_price(
            product_category=product_category,
            customer_state=customer_state,
        )

        # Initialize seller state
        state = create_initial_state(
            target_price=pricing["target_price"],
            floor_price=pricing["floor_price"],
            optimal_price=pricing["optimal_price"],
            product_category=product_category,
            customer_state=customer_state,
            price_simulations=pricing["price_simulations"],
            conversion_probability=pricing["optimal_conversion_prob"],
        )

        # First turn: buyer initiates
        opening = f"Hi, I'm looking at the {product_category.replace('_', ' ')}. What's the price?"
        state, seller_response = run_negotiation_turn(state, opening)

        conversation_log = [
            {"role": "buyer", "content": opening},
            {"role": "seller", "content": seller_response},
        ]
        logger.info(f"  Buyer: {opening}")
        logger.info(f"  Seller: {seller_response[:100]}...")

        # Negotiation loop
        outcome = "timeout"
        rounds = 1

        for round_num in range(2, persona.max_rounds + 1):
            # Generate buyer response
            buyer_msg = self._generate_buyer_response(
                persona, conversation_log, seller_response,
            )
            logger.info(f"  [Round {round_num}] Buyer: {buyer_msg}")

            # Check for deal outcome
            outcome = self._detect_deal_outcome(buyer_msg)
            conversation_log.append({"role": "buyer", "content": buyer_msg})

            if outcome != "continuing":
                rounds = round_num
                break

            # Seller responds
            state, seller_response = run_negotiation_turn(state, buyer_msg)
            conversation_log.append({"role": "seller", "content": seller_response})
            logger.info(f"  [Round {round_num}] Seller: {seller_response[:100]}...")
            rounds = round_num

        # Compute metrics
        final_offer = state["current_offer"]
        target = pricing["target_price"]
        floor = pricing["floor_price"]
        margin_retained = max(0, ((final_offer - floor) / (target - floor)) * 100) if target != floor else 100

        result = {
            "run_id": run_id,
            "persona": persona.name,
            "persona_style": persona.style,
            "product_category": product_category,
            "target_price": target,
            "floor_price": floor,
            "optimal_price": pricing["optimal_price"],
            "final_offer": final_offer,
            "outcome": outcome,
            "rounds": rounds,
            "margin_retained_pct": round(margin_retained, 1),
            "discount_given_pct": round((1 - final_offer / target) * 100, 1) if target > 0 else 0,
            "value_adds_offered": state.get("value_adds_offered", []),
            "num_value_adds": len(state.get("value_adds_offered", [])),
            "timestamp": datetime.now().isoformat(),
        }

        status_emoji = "✅" if outcome == "accepted" else "❌" if outcome == "rejected" else "⏰"
        logger.info(f"  {status_emoji} Outcome: {outcome} | Final: ${final_offer:.2f} | Margin: {margin_retained:.1f}%")

        self.results.append(result)
        return result

    def run_arena(
        self,
        runs_per_persona: int = 10,
        personas: list[str] | None = None,
        product_categories: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Run the full arena simulation.
        
        Args:
            runs_per_persona: Number of negotiations per persona
            personas: List of persona styles to include (default: all)
            product_categories: Categories to test (default: computers_accessories)
        """
        if personas is None:
            selected_personas = ALL_PERSONAS
        else:
            selected_personas = [get_persona(p) for p in personas]

        if product_categories is None:
            product_categories = ["computers_accessories"]

        total_runs = len(selected_personas) * len(product_categories) * runs_per_persona
        logger.info(f"\n{'#'*60}")
        logger.info(f"  STARTING ARENA: {total_runs} total negotiations")
        logger.info(f"  Personas: {[p.name for p in selected_personas]}")
        logger.info(f"  Categories: {product_categories}")
        logger.info(f"{'#'*60}\n")

        run_id = 0
        for persona in selected_personas:
            for category in product_categories:
                for i in range(runs_per_persona):
                    try:
                        self.run_single_negotiation(
                            persona=persona,
                            product_category=category,
                            run_id=run_id,
                        )
                    except Exception as e:
                        logger.error(f"  Run {run_id} FAILED: {e}")
                        self.results.append({
                            "run_id": run_id,
                            "persona": persona.name,
                            "persona_style": persona.style,
                            "product_category": category,
                            "outcome": "error",
                            "error": str(e),
                            "timestamp": datetime.now().isoformat(),
                        })
                    run_id += 1
                    time.sleep(1)  # Rate limiting

        # Save results
        df = pd.DataFrame(self.results)
        output_path = ARENA_DIR / f"arena_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"\n  Results saved to {output_path}")

        # Print summary
        self._print_summary(df)

        return df

    def _print_summary(self, df: pd.DataFrame):
        """Print arena results summary."""
        logger.info(f"\n{'='*60}")
        logger.info("  ARENA RESULTS SUMMARY")
        logger.info(f"{'='*60}")

        for persona in df["persona"].unique():
            pdf = df[df["persona"] == persona]
            win_rate = (pdf["outcome"] == "accepted").mean() * 100
            avg_margin = pdf["margin_retained_pct"].mean()
            avg_rounds = pdf["rounds"].mean()
            avg_discount = pdf.get("discount_given_pct", pd.Series([0])).mean()

            logger.info(f"\n  📊 {persona}:")
            logger.info(f"     Win Rate:       {win_rate:.0f}%")
            logger.info(f"     Avg Margin:     {avg_margin:.1f}%")
            logger.info(f"     Avg Rounds:     {avg_rounds:.1f}")
            logger.info(f"     Avg Discount:   {avg_discount:.1f}%")


# ── CLI Entry Point ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run the Multi-Agent Negotiation Arena")
    parser.add_argument("--runs", type=int, default=10, help="Runs per persona")
    parser.add_argument(
        "--personas", nargs="+", default=None,
        choices=["aggressive", "value", "urgent"],
        help="Persona styles to include",
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Product categories to test",
    )
    args = parser.parse_args()

    arena = NegotiationArena()
    arena.run_arena(
        runs_per_persona=args.runs,
        personas=args.personas,
        product_categories=args.categories,
    )


if __name__ == "__main__":
    main()
