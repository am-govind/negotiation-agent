"""
Negotiation State Management for LangGraph.

Defines the typed state dictionary that tracks all conversation
and pricing context across the agent's lifecycle.
"""
from typing import TypedDict, Literal, Annotated
from langgraph.graph.message import add_messages


# ── Intent Types ──────────────────────────────────────────────
IntentType = Literal[
    "price_objection",
    "competitor_mention",
    "shipping_query",
    "general_question",
    "acceptance",
    "rejection",
    "greeting",
    "walkaway_threat",
]


# ── Negotiation State ────────────────────────────────────────

class NegotiationState(TypedDict):
    """
    Complete state for a single negotiation session.
    Updated by each node in the LangGraph pipeline.
    """
    # ── Message History ───────────────────────
    messages: Annotated[list, add_messages]

    # ── ML-Derived Boundaries ─────────────────
    target_price: float
    floor_price: float
    optimal_price: float
    opening_price: float

    # ── Live Negotiation Tracking ─────────────
    current_offer: float       # Agent's last offered price
    user_counter_offer: float  # User's last counter-offer (0 if none)
    negotiation_round: int
    max_rounds: int

    # ── Intent & Routing ──────────────────────
    intent: IntentType
    intent_confidence: float

    # ── Product Context ───────────────────────
    product_category: str
    customer_state: str
    seller_state: str

    # ── Outcome Tracking ─────────────────────
    deal_closed: bool
    deal_abandoned: bool
    value_adds_offered: list[str]    # Track which value-adds have been used

    # ── Tool Call Results ─────────────────────
    last_tool_result: str       # Feedback from submit_official_offer

    # ── Price Elasticity Context ──────────────
    conversion_probability: float
    price_simulations: list[dict]


def create_initial_state(
    target_price: float,
    floor_price: float,
    optimal_price: float,
    product_category: str,
    customer_state: str = "SP",
    seller_state: str = "SP",
    price_simulations: list[dict] | None = None,
    conversion_probability: float = 0.0,
) -> NegotiationState:
    """Create a fresh negotiation state with ML-derived pricing boundaries."""
    opening_price = round(optimal_price * 1.10, 2)  # Start 10% above optimal

    return NegotiationState(
        messages=[],
        target_price=target_price,
        floor_price=floor_price,
        optimal_price=optimal_price,
        opening_price=opening_price,
        current_offer=opening_price,
        user_counter_offer=0.0,
        negotiation_round=0,
        max_rounds=10,
        intent="greeting",
        intent_confidence=1.0,
        product_category=product_category,
        customer_state=customer_state,
        seller_state=seller_state,
        deal_closed=False,
        deal_abandoned=False,
        value_adds_offered=[],
        last_tool_result="",
        conversion_probability=conversion_probability,
        price_simulations=price_simulations or [],
    )
