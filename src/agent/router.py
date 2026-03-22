"""
Intent Router (Improvisation 4: Real-Time Intent Routing).

Uses a lightweight LLM to classify user messages into intents,
routing to the appropriate prompt template and handling strategy.
"""
import logging
from src.utils.llm import get_router_llm
from langchain_core.messages import SystemMessage, HumanMessage

from src.agent.state import NegotiationState, IntentType
from src.utils.gemini import extract_json

logger = logging.getLogger(__name__)

# Router LLM imported from src.utils.llm


ROUTER_SYSTEM_PROMPT = """You are an intent classifier for a sales negotiation system. 
Classify the customer's message into exactly ONE intent.

Available intents:
- "price_objection": Customer objects to the price, asks for a discount, counter-offers
- "competitor_mention": Customer mentions a competitor or alternative product
- "shipping_query": Customer asks about shipping, delivery, or logistics
- "general_question": Customer asks about product features, specs, or general info
- "acceptance": Customer agrees to the price or wants to proceed with purchase
- "rejection": Customer explicitly rejects the offer or says they won't buy
- "greeting": Customer is greeting or starting the conversation
- "walkaway_threat": Customer threatens to leave or buy elsewhere without naming a specific competitor

Respond with ONLY a JSON object: {"intent": "<intent>", "confidence": <0.0-1.0>}"""


def classify_intent(state: NegotiationState) -> dict:
    """
    Node 1: Classify the user's latest message into an actionable intent.
    Returns a state update dict.
    """
    messages = state["messages"]
    if not messages:
        return {"intent": "greeting", "intent_confidence": 1.0}

    # Get the last user message
    last_message = None
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            last_message = msg.content
            break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            last_message = msg.get("content", "")
            break
        elif isinstance(msg, HumanMessage):
            last_message = msg.content
            break

    if not last_message:
        return {"intent": "greeting", "intent_confidence": 1.0}

    try:
        llm = get_router_llm()
        response = llm.invoke([
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"Customer message: \"{last_message}\""),
        ])

        result = extract_json(response.content)
        intent = result.get("intent", "general_question")
        confidence = float(result.get("confidence", 0.5))

        # Validate intent is a known type
        valid_intents = [
            "price_objection", "competitor_mention", "shipping_query",
            "general_question", "acceptance", "rejection", "greeting",
            "walkaway_threat",
        ]
        if intent not in valid_intents:
            intent = "general_question"
            confidence = 0.3

        logger.info(f"  Intent: {intent} (confidence: {confidence:.2f})")
        return {"intent": intent, "intent_confidence": confidence}

    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return {"intent": "general_question", "intent_confidence": 0.0}


def get_routing_decision(intent: IntentType) -> str:
    """
    Determine which prompt template / strategy to use based on intent.
    Returns the prompt key for the agentic core.
    """
    routing_map = {
        "price_objection": "negotiation",
        "competitor_mention": "competitor_differentiation",
        "shipping_query": "faq",
        "general_question": "product_info",
        "acceptance": "closing",
        "rejection": "retention",
        "greeting": "opening",
        "walkaway_threat": "urgency",
    }
    return routing_map.get(intent, "negotiation")
