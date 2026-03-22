"""
Competitor Analysis Skill: Data-driven competitor differentiation.

Instead of the LLM improvising competitor responses, this skill
provides structured feature comparison data that the LLM formats
into a natural response.
"""
from src.agent.skills.registry import Skill, SkillRegistry


# ── Competitor Feature Database ───────────────────────────────
# In production, this would be pulled from a CRM or product DB.
# For the portfolio project, we use a structured knowledge base.

COMPETITOR_DATA = {
    "default": {
        "our_advantages": [
            {"feature": "2-Year Extended Warranty", "value": "Included free", "competitor": "Extra $50-80"},
            {"feature": "Customer Support", "value": "4-hour response, dedicated line", "competitor": "24-48 hour email only"},
            {"feature": "Return Policy", "value": "30-day no-questions-asked", "competitor": "15-day with restocking fee"},
            {"feature": "Shipping", "value": "Free express available", "competitor": "Standard only, $15-25"},
        ],
        "our_weaknesses": [
            {"feature": "Brand Recognition", "note": "Competitor may have broader brand awareness"},
        ],
    },
    "dell": {
        "name": "Dell",
        "our_advantages": [
            {"feature": "Warranty", "value": "2-year included", "competitor": "1-year basic, paid upgrade"},
            {"feature": "Support", "value": "Priority 4-hour response", "competitor": "Standard phone queue"},
            {"feature": "Customization", "value": "Full spec customization", "competitor": "Limited configurations"},
            {"feature": "Price Match", "value": "We match + add value-adds", "competitor": "Fixed pricing"},
        ],
    },
    "amazon": {
        "name": "Amazon",
        "our_advantages": [
            {"feature": "Expertise", "value": "Dedicated product specialists", "competitor": "Self-service"},
            {"feature": "Warranty", "value": "2-year extended", "competitor": "Manufacturer warranty only"},
            {"feature": "Post-Sale Support", "value": "Dedicated account manager", "competitor": "Chatbot support"},
        ],
    },
    "best buy": {
        "name": "Best Buy",
        "our_advantages": [
            {"feature": "Online Experience", "value": "AI-powered personalization", "competitor": "Generic browsing"},
            {"feature": "Warranty", "value": "2-year included", "competitor": "Geek Squad paid add-on"},
            {"feature": "Delivery", "value": "Free express shipping", "competitor": "Paid or in-store pickup"},
        ],
    },
}


def _analyze_competitor(
    competitor_name: str = "generic",
    current_offer: float = 0,
    **kwargs,
) -> dict:
    """
    Get structured competitor differentiation data.
    Returns our advantages and suggested talking points.
    """
    # Find best match
    key = competitor_name.lower().strip()
    comp_data = COMPETITOR_DATA.get(key, COMPETITOR_DATA["default"])

    advantages = comp_data.get("our_advantages", COMPETITOR_DATA["default"]["our_advantages"])

    # Calculate total value of our advantages
    value_add_total = sum(
        _estimate_advantage_value(adv["feature"])
        for adv in advantages
    )

    talking_points = [
        f"Our {adv['feature']}: {adv['value']} (vs competitor: {adv.get('competitor', 'N/A')})"
        for adv in advantages[:3]
    ]

    return {
        "competitor": competitor_name,
        "advantages": advantages,
        "total_advantage_value": round(value_add_total, 2),
        "talking_points": talking_points,
        "strategy": (
            "Acknowledge the competitor respectfully, then highlight our "
            f"${value_add_total:.0f}+ in additional value through superior "
            "warranty, support, and shipping options."
        ),
        "recommended_response_tone": "confident_not_aggressive",
    }


def _estimate_advantage_value(feature: str) -> float:
    """Estimate dollar value of each competitive advantage."""
    values = {
        "2-Year Extended Warranty": 50,
        "Warranty": 50,
        "Customer Support": 30,
        "Support": 30,
        "Return Policy": 20,
        "Shipping": 25,
        "Delivery": 25,
        "Customization": 40,
        "Price Match": 0,
        "Expertise": 35,
        "Post-Sale Support": 30,
        "Online Experience": 15,
    }
    return values.get(feature, 20)


def register_competitor_skills(registry: SkillRegistry):
    """Register competitor analysis skills."""

    registry.register(Skill(
        name="analyze_competitor",
        description=(
            "Get structured competitive differentiation data when a customer "
            "mentions a competitor. Returns our advantages, their weaknesses, "
            "and recommended talking points with dollar values."
        ),
        category="competitor",
        execute=_analyze_competitor,
        tool_schema={
            "type": "function",
            "function": {
                "name": "analyze_competitor",
                "description": (
                    "Analyze a competitor and get differentiation talking points. "
                    "Call this when the customer mentions Dell, Amazon, Best Buy, "
                    "or any other competitor."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "competitor_name": {
                            "type": "string",
                            "description": "Name of the competitor mentioned by the customer",
                        },
                    },
                    "required": ["competitor_name"],
                },
            },
        },
    ))
