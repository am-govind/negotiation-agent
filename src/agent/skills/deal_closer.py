"""
Deal Closer Skill: Final deal packaging and confirmation logic.

Handles the mechanics of closing a deal: finalizing price,
packaging value-adds, and generating the deal summary.
"""
from src.agent.skills.registry import Skill, SkillRegistry


def _finalize_deal(
    final_price: float,
    floor_price: float,
    target_price: float,
    opening_price: float,
    value_adds_offered: list[str] | None = None,
    negotiation_round: int = 1,
    product_category: str = "",
    **kwargs,
) -> dict:
    """
    Finalize and package the deal for confirmation.
    Calculates all final metrics and generates a deal summary.
    """
    value_adds_offered = value_adds_offered or []
    margin_retained = ((final_price - floor_price) / (target_price - floor_price) * 100) if target_price != floor_price else 100
    discount_from_opening = ((opening_price - final_price) / opening_price * 100) if opening_price > 0 else 0

    value_add_details = {
        "free_shipping": {"name": "Free Express Shipping", "value": 25.00},
        "extended_warranty": {"name": "2-Year Extended Warranty", "value": 50.00},
        "priority_support": {"name": "Priority Customer Support", "value": 30.00},
        "bundle_discount": {"name": "Accessories Bundle", "value": 40.00},
    }

    included_adds = [
        value_add_details[va] for va in value_adds_offered
        if va in value_add_details
    ]
    total_value_add = sum(a["value"] for a in included_adds)

    return {
        "deal_confirmed": True,
        "final_price": final_price,
        "product_category": product_category.replace("_", " ").title(),
        "discount_from_opening_pct": round(discount_from_opening, 1),
        "margin_retained_pct": round(max(0, min(100, margin_retained)), 1),
        "rounds_to_close": negotiation_round,
        "included_value_adds": included_adds,
        "total_value_add_value": total_value_add,
        "total_package_value": round(final_price + total_value_add, 2),
        "deal_summary": (
            f"✅ Deal Closed at ${final_price:.2f} for "
            f"{product_category.replace('_', ' ').title()}"
            + (f" with {len(included_adds)} bonus(es) worth ${total_value_add:.0f}"
               if included_adds else "")
            + f" — closed in {negotiation_round} rounds."
        ),
        "rating": (
            "excellent" if margin_retained > 80 else
            "good" if margin_retained > 50 else
            "acceptable" if margin_retained > 20 else
            "thin_margin"
        ),
    }


def _generate_walkaway_response(
    current_offer: float,
    floor_price: float,
    value_adds_offered: list[str] | None = None,
    **kwargs,
) -> dict:
    """Generate a graceful walkaway response when the deal can't close."""
    value_adds_offered = value_adds_offered or []
    remaining = [v for v in ["free_shipping", "extended_warranty", "priority_support", "bundle_discount"]
                 if v not in value_adds_offered]

    can_make_final = len(remaining) > 0 or current_offer > floor_price * 1.02

    return {
        "should_attempt_save": can_make_final,
        "final_offer_possible": current_offer > floor_price * 1.02,
        "remaining_value_adds": remaining,
        "message_strategy": (
            "Make one final offer combining lowest price with remaining value-adds"
            if can_make_final else
            "Thank them gracefully and leave the door open for future business"
        ),
        "tone": "gracious_not_desperate",
    }


def register_deal_skills(registry: SkillRegistry):
    """Register deal closing skills."""

    registry.register(Skill(
        name="finalize_deal",
        description=(
            "Finalize the deal package with price, value-adds, and metrics. "
            "Call this when the customer accepts an offer."
        ),
        category="closing",
        requires_state=True,
        execute=_finalize_deal,
        tool_schema={
            "type": "function",
            "function": {
                "name": "finalize_deal",
                "description": "Package the final deal for customer confirmation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "final_price": {
                            "type": "number",
                            "description": "The agreed-upon final price",
                        },
                    },
                    "required": ["final_price"],
                },
            },
        },
    ))

    registry.register(Skill(
        name="generate_walkaway_response",
        description="Generate strategy for when the customer is about to leave.",
        category="closing",
        requires_state=True,
        execute=_generate_walkaway_response,
        tool_schema={
            "type": "function",
            "function": {
                "name": "generate_walkaway_response",
                "description": "Get the strategy for handling a customer walkaway.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ))
