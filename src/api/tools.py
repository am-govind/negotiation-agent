"""
Tool Calling Module (Improvisation 1: Strict Tool Calling).

Defines the `submit_official_offer` tool that the LLM must call
instead of outputting prices in plain text. The Python function
validates against the floor price before approving.
"""
import logging
from typing import Annotated
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class OfferResult(BaseModel):
    """Result of an offer submission."""
    approved: bool
    message: str
    offered_price: float
    floor_price: float


class OfferInput(BaseModel):
    """Input schema for the submit_official_offer tool."""
    price: Annotated[float, Field(gt=0, description="The price to offer the customer")]
    justification: Annotated[str, Field(description="Brief reason for this price point")]


def submit_official_offer(
    price: float,
    justification: str,
    floor_price: float,
    target_price: float,
    current_round: int = 1,
) -> OfferResult:
    """
    Validate and submit a price offer. This is the ONLY way the agent
    can communicate a price to the customer.
    
    Args:
        price: The proposed offer price
        justification: Agent's reasoning for this price
        floor_price: Absolute minimum acceptable price
        target_price: Ideal price point from ML model
        current_round: Current negotiation round
    
    Returns:
        OfferResult with approval status and feedback message
    """
    logger.info(
        f"[Round {current_round}] Offer submitted: ${price:.2f} "
        f"(floor: ${floor_price:.2f}, target: ${target_price:.2f}) "
        f"— {justification}"
    )

    # ── Guardrail 1: Below floor price ────────────────────────
    if price < floor_price:
        margin_below = ((floor_price - price) / floor_price) * 100
        logger.warning(f"  REJECTED: ${price:.2f} is {margin_below:.1f}% below floor")
        return OfferResult(
            approved=False,
            message=(
                f"Error: Offer REJECTED by the finance system. "
                f"${price:.2f} is below the ${floor_price:.2f} floor price. "
                f"You must offer at least ${floor_price:.2f}. "
                f"Consider offering a value-add (free shipping, warranty, "
                f"faster delivery) instead of dropping below the floor."
            ),
            offered_price=price,
            floor_price=floor_price,
        )

    # ── Guardrail 2: Sanity check — way above target ─────────
    if price > target_price * 1.5:
        logger.warning(f"  WARNING: ${price:.2f} is >50% above target, may lose the deal")
        return OfferResult(
            approved=True,
            message=(
                f"Warning: Offer of ${price:.2f} approved but is significantly "
                f"above the market target of ${target_price:.2f}. "
                f"The customer is likely to reject this. Consider a more "
                f"competitive offer to improve conversion probability."
            ),
            offered_price=price,
            floor_price=floor_price,
        )

    # ── Approved ──────────────────────────────────────────────
    margin_retained = ((price - floor_price) / (target_price - floor_price)) * 100
    margin_retained = max(0, min(margin_retained, 100))

    logger.info(f"  APPROVED: ${price:.2f} (margin retained: {margin_retained:.1f}%)")
    return OfferResult(
        approved=True,
        message=(
            f"Offer of ${price:.2f} APPROVED. "
            f"Margin retained: {margin_retained:.0f}%. "
            f"Proceed with generating the customer-facing response."
        ),
        offered_price=price,
        floor_price=floor_price,
    )


# ── LangChain Tool Definition ────────────────────────────────
# This is the schema that gets passed to the LLM via function calling

SUBMIT_OFFER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_official_offer",
        "description": (
            "Submit an official price offer to the customer. This is the ONLY way "
            "to propose a price. The finance system will validate the offer against "
            "the floor price. If rejected, you must generate a new offer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "price": {
                    "type": "number",
                    "description": "The price to offer the customer in USD. Must be above the floor price.",
                },
                "justification": {
                    "type": "string",
                    "description": (
                        "Brief explanation for why this price was chosen. "
                        "Reference the customer's message and negotiation context."
                    ),
                },
            },
            "required": ["price", "justification"],
        },
    },
}


def add_value_proposition(
    value_type: str,
    current_offer: float,
    floor_price: float,
) -> dict:
    """
    Instead of dropping price, offer value-adds.
    
    Args:
        value_type: One of 'free_shipping', 'extended_warranty', 'priority_support', 'bundle_discount'
        current_offer: The current price being discussed
        floor_price: Minimum acceptable price
    
    Returns:
        dict with the value proposition details
    """
    value_adds = {
        "free_shipping": {
            "name": "Free Express Shipping",
            "estimated_value": 25.00,
            "description": "We'll waive all shipping costs and upgrade to express delivery.",
        },
        "extended_warranty": {
            "name": "Extended 2-Year Warranty",
            "estimated_value": 50.00,
            "description": "Full coverage warranty extended to 2 years at no extra cost.",
        },
        "priority_support": {
            "name": "Priority Customer Support",
            "estimated_value": 30.00,
            "description": "Dedicated support line with guaranteed 4-hour response time.",
        },
        "bundle_discount": {
            "name": "Accessories Bundle",
            "estimated_value": 40.00,
            "description": "Complimentary accessories package included with your purchase.",
        },
    }

    if value_type not in value_adds:
        return {"error": f"Unknown value type: {value_type}"}

    proposition = value_adds[value_type]
    effective_discount = (proposition["estimated_value"] / current_offer) * 100

    return {
        "value_add": proposition,
        "effective_discount_percent": round(effective_discount, 1),
        "price_remains": current_offer,
        "total_value": round(current_offer + proposition["estimated_value"], 2),
    }


ADD_VALUE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "add_value_proposition",
        "description": (
            "Offer a value-add to the customer instead of dropping the price. "
            "Use this BEFORE dropping the price to maximize margin retention."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "value_type": {
                    "type": "string",
                    "enum": ["free_shipping", "extended_warranty", "priority_support", "bundle_discount"],
                    "description": "The type of value-add to offer.",
                },
            },
            "required": ["value_type"],
        },
    },
}
