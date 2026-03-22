"""
Negotiation Tactics Skill: Strategic decision engine.

Determines the optimal negotiation tactic based on the current state,
round number, intent, and pricing position. Pure logic — no LLM needed.
"""
from src.agent.skills.registry import Skill, SkillRegistry


def _select_tactic(
    intent: str,
    negotiation_round: int,
    current_offer: float,
    floor_price: float,
    target_price: float,
    value_adds_offered: list[str] | None = None,
    **kwargs,
) -> dict:
    """
    Select the optimal negotiation tactic based on current state.
    
    Tactic hierarchy:
    1. Anchor high → 2. Value-add first → 3. Small concession → 4. Bundle → 5. Final offer
    """
    value_adds_offered = value_adds_offered or []
    margin_pct = ((current_offer - floor_price) / (target_price - floor_price) * 100) if target_price != floor_price else 100
    all_value_adds = ["free_shipping", "extended_warranty", "priority_support", "bundle_discount"]
    remaining_value_adds = [v for v in all_value_adds if v not in value_adds_offered]

    # Decision tree
    if negotiation_round <= 1:
        tactic = {
            "name": "anchor_high",
            "action": "Present the opening price confidently with key benefits",
            "price_action": "hold",
            "recommend_value_add": False,
            "confidence_level": "high",
            "tone": "enthusiastic_professional",
        }

    elif intent in ("competitor_mention", "walkaway_threat") and margin_pct > 40:
        # Competitor pressure — offer value-add first, small price drop second
        if remaining_value_adds:
            tactic = {
                "name": "differentiate_and_sweeten",
                "action": f"Highlight competitive advantages, then offer {remaining_value_adds[0]}",
                "price_action": "hold_or_small_drop",
                "recommend_value_add": True,
                "suggested_value_add": remaining_value_adds[0],
                "max_price_drop_pct": 5,
                "confidence_level": "medium",
                "tone": "confident_empathetic",
            }
        else:
            # All value-adds exhausted — small price concession
            tactic = {
                "name": "final_concession",
                "action": "Make a small final price concession",
                "price_action": "small_drop",
                "recommend_value_add": False,
                "max_price_drop_pct": 8,
                "confidence_level": "medium",
                "tone": "firm_but_fair",
            }

    elif intent == "price_objection" and negotiation_round <= 3:
        if remaining_value_adds:
            tactic = {
                "name": "value_add_first",
                "action": f"Offer {remaining_value_adds[0]} before dropping price",
                "price_action": "hold",
                "recommend_value_add": True,
                "suggested_value_add": remaining_value_adds[0],
                "confidence_level": "high",
                "tone": "helpful_professional",
            }
        else:
            tactic = {
                "name": "controlled_concession",
                "action": "Make a calculated price concession",
                "price_action": "small_drop",
                "recommend_value_add": False,
                "max_price_drop_pct": 7,
                "confidence_level": "medium",
                "tone": "understanding",
            }

    elif intent == "price_objection" and negotiation_round > 3:
        if margin_pct > 20:
            tactic = {
                "name": "split_the_difference",
                "action": "Propose splitting the difference between current offers",
                "price_action": "moderate_drop",
                "recommend_value_add": False,
                "max_price_drop_pct": 10,
                "confidence_level": "low",
                "tone": "collaborative",
            }
        else:
            tactic = {
                "name": "hold_firm",
                "action": "Hold the current price as the best and final offer",
                "price_action": "hold",
                "recommend_value_add": False,
                "confidence_level": "high",
                "tone": "firm_respectful",
            }

    elif intent == "acceptance":
        tactic = {
            "name": "close_deal",
            "action": "Confirm the deal and express appreciation",
            "price_action": "hold",
            "recommend_value_add": False,
            "confidence_level": "high",
            "tone": "warm_grateful",
        }

    elif intent == "rejection":
        tactic = {
            "name": "graceful_retention",
            "action": "Thank them, leave the door open, make one final offer if margin allows",
            "price_action": "last_chance_drop" if margin_pct > 30 else "hold",
            "recommend_value_add": bool(remaining_value_adds),
            "suggested_value_add": remaining_value_adds[0] if remaining_value_adds else None,
            "max_price_drop_pct": 12 if margin_pct > 30 else 0,
            "confidence_level": "low",
            "tone": "gracious_professional",
        }

    else:
        tactic = {
            "name": "engage_and_inform",
            "action": "Answer the question helpfully and steer toward closing",
            "price_action": "hold",
            "recommend_value_add": False,
            "confidence_level": "medium",
            "tone": "helpful_professional",
        }

    # Add context
    tactic["current_margin_pct"] = round(margin_pct, 1)
    tactic["remaining_value_adds"] = remaining_value_adds
    tactic["negotiation_round"] = negotiation_round
    tactic["intent"] = intent

    return tactic


def register_negotiation_skills(registry: SkillRegistry):
    """Register negotiation tactic skills."""

    registry.register(Skill(
        name="select_tactic",
        description=(
            "Select the optimal negotiation tactic based on current round, "
            "intent, pricing position, and value-adds already offered. "
            "Returns the recommended action, tone, and price movement."
        ),
        category="negotiation",
        requires_state=True,
        execute=_select_tactic,
        tool_schema={
            "type": "function",
            "function": {
                "name": "select_tactic",
                "description": (
                    "Get the recommended negotiation tactic for the current "
                    "situation. Call this before deciding what to say or offer."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string",
                            "description": "The classified customer intent",
                        },
                    },
                    "required": ["intent"],
                },
            },
        },
    ))
