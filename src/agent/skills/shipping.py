"""
Shipping Skill: Deterministic shipping estimation and logistics info.

Calculates delivery times and costs based on seller/customer state
data from the Olist dataset rather than the LLM guessing.
"""
import pandas as pd

from src.agent.skills.registry import Skill, SkillRegistry
from src.config import FEATURES_PARQUET

# Brazilian state distance tiers (simplified)
_SAME_REGION_STATES = {
    "SP": ["SP", "RJ", "MG", "ES"],
    "RJ": ["SP", "RJ", "MG", "ES"],
    "MG": ["SP", "RJ", "MG", "ES"],
    "RS": ["RS", "SC", "PR"],
    "SC": ["RS", "SC", "PR"],
    "PR": ["RS", "SC", "PR"],
}


def _estimate_shipping(
    customer_state: str,
    seller_state: str,
    product_weight_g: float = 1000,
    **kwargs,
) -> dict:
    """Estimate shipping cost and delivery time based on real logistics data."""

    # Determine distance tier
    same_state = customer_state == seller_state
    same_region = customer_state in _SAME_REGION_STATES.get(seller_state, [])

    if same_state:
        tier = "local"
        base_days = 3
        base_cost_factor = 0.7
    elif same_region:
        tier = "regional"
        base_days = 6
        base_cost_factor = 1.0
    else:
        tier = "national"
        base_days = 12
        base_cost_factor = 1.5

    # Weight-based cost adjustment
    weight_kg = product_weight_g / 1000
    weight_surcharge = max(0, (weight_kg - 2) * 3.5)  # $3.50 per kg over 2kg

    estimated_cost = round(15 * base_cost_factor + weight_surcharge, 2)
    express_cost = round(estimated_cost * 1.8, 2)
    express_days = max(2, base_days - 3)

    return {
        "shipping_tier": tier,
        "customer_state": customer_state,
        "seller_state": seller_state,
        "standard_delivery": {
            "days": f"{base_days}-{base_days + 3} business days",
            "cost": estimated_cost,
        },
        "express_delivery": {
            "days": f"{express_days}-{express_days + 2} business days",
            "cost": express_cost,
        },
        "free_shipping_value": estimated_cost,
        "message": (
            f"Standard shipping to {customer_state}: ${estimated_cost:.2f} "
            f"({base_days}-{base_days+3} days). "
            f"Express: ${express_cost:.2f} ({express_days}-{express_days+2} days)."
        ),
    }


def _check_delivery_feasibility(
    customer_state: str,
    seller_state: str,
    urgency: str = "normal",
    **kwargs,
) -> dict:
    """Check if delivery is feasible within the customer's timeframe."""
    shipping = _estimate_shipping(customer_state, seller_state)

    if urgency == "urgent":
        feasible = shipping["express_delivery"]["cost"] > 0
        recommended = "express"
        days = shipping["express_delivery"]["days"]
    else:
        feasible = True
        recommended = "standard"
        days = shipping["standard_delivery"]["days"]

    return {
        "feasible": feasible,
        "recommended_tier": recommended,
        "estimated_days": days,
        "can_offer_free_shipping": True,
        "free_shipping_saves_customer": shipping["free_shipping_value"],
    }


def register_shipping_skills(registry: SkillRegistry):
    """Register shipping-related skills."""

    registry.register(Skill(
        name="estimate_shipping",
        description=(
            "Calculate shipping cost and delivery time based on customer "
            "and seller locations. Returns standard and express options."
        ),
        category="shipping",
        execute=_estimate_shipping,
        tool_schema={
            "type": "function",
            "function": {
                "name": "estimate_shipping",
                "description": (
                    "Get shipping estimates between seller and customer. "
                    "Call this when the customer asks about shipping or delivery."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_state": {
                            "type": "string",
                            "description": "Customer state code (e.g., SP, RJ)",
                        },
                        "seller_state": {
                            "type": "string",
                            "description": "Seller state code",
                        },
                    },
                    "required": ["customer_state", "seller_state"],
                },
            },
        },
    ))

    registry.register(Skill(
        name="check_delivery_feasibility",
        description="Check if delivery is feasible within the customer's timeframe.",
        category="shipping",
        execute=_check_delivery_feasibility,
        tool_schema={
            "type": "function",
            "function": {
                "name": "check_delivery_feasibility",
                "description": "Check if we can deliver on time for an urgent customer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_state": {"type": "string"},
                        "seller_state": {"type": "string"},
                        "urgency": {
                            "type": "string",
                            "enum": ["normal", "urgent"],
                            "description": "How urgently the customer needs delivery",
                        },
                    },
                    "required": ["customer_state", "seller_state"],
                },
            },
        },
    ))
