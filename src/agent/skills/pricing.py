"""
Pricing Skills: Deterministic price computation and elasticity analysis.

These skills handle ALL math — the LLM never calculates prices itself.
"""
from src.agent.skills.registry import Skill, SkillRegistry


def _calculate_counteroffer(
    user_counter: float,
    current_offer: float,
    floor_price: float,
    target_price: float,
    negotiation_round: int,
    **kwargs,
) -> dict:
    """
    Calculate the optimal counteroffer using a concession curve.
    
    Uses a diminishing concession strategy:
    - Round 1-2: concede 30-40% of the gap
    - Round 3-4: concede 15-20%
    - Round 5+: concede 5-10% (hold firm)
    """
    gap = current_offer - user_counter

    if user_counter >= current_offer:
        return {
            "action": "accept",
            "recommended_price": current_offer,
            "message": "Customer's offer meets or exceeds current price. Accept the deal.",
        }

    if user_counter < floor_price:
        # Customer is below floor — calculate how far
        floor_gap_pct = ((floor_price - user_counter) / floor_price) * 100
        midpoint = (floor_price + current_offer) / 2

        return {
            "action": "counter_with_value_add",
            "recommended_price": round(midpoint, 2),
            "customer_below_floor_by_pct": round(floor_gap_pct, 1),
            "message": (
                f"Customer's offer of ${user_counter:.2f} is {floor_gap_pct:.0f}% below "
                f"the floor. Recommend offering ${midpoint:.2f} with a value-add."
            ),
            "suggest_value_add": True,
        }

    # Concession curve based on round
    if negotiation_round <= 2:
        concession_pct = 0.35
    elif negotiation_round <= 4:
        concession_pct = 0.17
    else:
        concession_pct = 0.07

    concession = gap * concession_pct
    new_offer = max(floor_price, round(current_offer - concession, 2))

    margin_retained = ((new_offer - floor_price) / (target_price - floor_price)) * 100 if target_price != floor_price else 100

    return {
        "action": "counter",
        "recommended_price": new_offer,
        "concession_amount": round(concession, 2),
        "concession_pct": round(concession_pct * 100, 1),
        "margin_retained_pct": round(margin_retained, 1),
        "message": (
            f"Recommend countering at ${new_offer:.2f} "
            f"(${concession:.2f} concession, {margin_retained:.0f}% margin retained)."
        ),
    }


def _analyze_price_gap(
    user_counter: float,
    current_offer: float,
    floor_price: float,
    target_price: float,
    **kwargs,
) -> dict:
    """Analyze the gap between user's offer and seller's positions."""
    total_range = target_price - floor_price
    user_gap_from_floor = user_counter - floor_price
    user_gap_from_offer = current_offer - user_counter

    if total_range == 0:
        position_pct = 100
    else:
        position_pct = (user_gap_from_floor / total_range) * 100

    return {
        "user_counter": user_counter,
        "current_offer": current_offer,
        "floor_price": floor_price,
        "target_price": target_price,
        "gap_from_offer": round(user_gap_from_offer, 2),
        "gap_from_floor": round(user_gap_from_floor, 2),
        "user_position_in_range_pct": round(max(0, min(100, position_pct)), 1),
        "is_above_floor": user_counter >= floor_price,
        "negotiation_room": round(current_offer - floor_price, 2),
        "assessment": (
            "favorable" if position_pct >= 60 else
            "moderate" if position_pct >= 30 else
            "challenging" if position_pct >= 0 else
            "below_floor"
        ),
    }


def register_pricing_skills(registry: SkillRegistry):
    """Register all pricing-related skills."""

    registry.register(Skill(
        name="calculate_counteroffer",
        description=(
            "Calculate the mathematically optimal counteroffer based on the "
            "customer's counter-offer, current negotiation round, and pricing "
            "boundaries. Uses a diminishing concession curve strategy."
        ),
        category="pricing",
        execute=_calculate_counteroffer,
        requires_state=True,
        tool_schema={
            "type": "function",
            "function": {
                "name": "calculate_counteroffer",
                "description": (
                    "Calculate the optimal counteroffer price. Call this when the "
                    "customer proposes a specific price and you need to determine "
                    "your best counter."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_counter": {
                            "type": "number",
                            "description": "The price the customer proposed",
                        },
                    },
                    "required": ["user_counter"],
                },
            },
        },
    ))

    registry.register(Skill(
        name="analyze_price_gap",
        description=(
            "Analyze the gap between the customer's offer and the seller's "
            "positions (floor, target, current). Returns a structured assessment."
        ),
        category="pricing",
        execute=_analyze_price_gap,
        requires_state=True,
        tool_schema={
            "type": "function",
            "function": {
                "name": "analyze_price_gap",
                "description": (
                    "Analyze where the customer's counter-offer sits relative to "
                    "the floor and target prices. Use this to understand how much "
                    "room you have before making a decision."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_counter": {
                            "type": "number",
                            "description": "The customer's proposed price",
                        },
                    },
                    "required": ["user_counter"],
                },
            },
        },
    ))
