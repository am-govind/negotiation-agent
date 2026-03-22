"""
LangGraph State Machine: Skills-augmented agentic negotiation pipeline.

Architecture (Skills + Prompts hybrid):
  Node 1: Router         → Classify intent (LLM)
  Node 2: Skill Selector → Run deterministic skills based on intent (Python)
  Node 3: Agentic Core   → LLM reasoning with skill results + tools
  Node 4: Tool Executor  → Validate tool calls + execute skills
  Node 5: Generator      → Produce final response (LLM)

The key difference from a prompt-only approach:
  - Prompt-only: LLM guesses at prices, product facts, and tactics
  - Skills hybrid: Python skills compute prices, fetch real data, and select
    tactics DETERMINISTICALLY, then the LLM formats the results conversationally

Usage:
    from src.agent.graph import create_negotiation_graph, run_negotiation
"""
import logging
import json
from typing import Literal

from src.utils.llm import get_core_llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END

from src.agent.state import NegotiationState, create_initial_state
from src.agent.router import classify_intent, get_routing_decision
from src.agent.prompts import get_system_prompt
from src.agent.skills.registry import get_registry
from src.utils.gemini import extract_text
from src.api.tools import (
    submit_official_offer, add_value_proposition,
    SUBMIT_OFFER_TOOL_SCHEMA, ADD_VALUE_TOOL_SCHEMA,
)

logger = logging.getLogger(__name__)

# Core LLM imported from src.utils.llm


# ── Node 1: Router ───────────────────────────────────────────

def router_node(state: NegotiationState) -> dict:
    """Classify user intent and update state."""
    logger.info("[Node 1] Classifying intent...")
    intent_update = classify_intent(state)
    round_num = state.get("negotiation_round", 0) + 1
    return {**intent_update, "negotiation_round": round_num}


# ── Node 2: Skill Selector (NEW — deterministic pre-computation) ──

def skill_selector_node(state: NegotiationState) -> dict:
    """
    Run deterministic skills BEFORE the LLM reasons.
    
    This is the key architectural upgrade: instead of the LLM guessing
    at prices, tactics, and product info, Python skills compute exact
    answers that get injected into the LLM's context.
    
    The LLM's job becomes: "Format these computed results into a
    natural conversation" rather than "Figure out the right price."
    """
    logger.info("[Node 2] Running skills based on intent...")
    registry = get_registry()
    intent = state.get("intent", "greeting")
    skill_results = {}

    # ── Always run: Negotiation Tactic Selector ───────────────
    tactic = registry.execute_skill(
        "select_tactic",
        intent=intent,
        negotiation_round=state.get("negotiation_round", 1),
        current_offer=state["current_offer"],
        floor_price=state["floor_price"],
        target_price=state["target_price"],
        value_adds_offered=state.get("value_adds_offered", []),
    )
    skill_results["tactic"] = tactic

    # ── Intent-specific skills ────────────────────────────────
    if intent in ("price_objection", "walkaway_threat"):
        # Extract user's counter-offer from the last message (if numeric)
        user_counter = _extract_price_from_messages(state["messages"])
        if user_counter > 0:
            counteroffer = registry.execute_skill(
                "calculate_counteroffer",
                user_counter=user_counter,
                current_offer=state["current_offer"],
                floor_price=state["floor_price"],
                target_price=state["target_price"],
                negotiation_round=state.get("negotiation_round", 1),
            )
            skill_results["counteroffer"] = counteroffer

            gap_analysis = registry.execute_skill(
                "analyze_price_gap",
                user_counter=user_counter,
                current_offer=state["current_offer"],
                floor_price=state["floor_price"],
                target_price=state["target_price"],
            )
            skill_results["gap_analysis"] = gap_analysis

    elif intent == "competitor_mention":
        # Extract competitor name from message
        competitor = _extract_competitor(state["messages"])
        comp_analysis = registry.execute_skill(
            "analyze_competitor",
            competitor_name=competitor,
            current_offer=state["current_offer"],
        )
        skill_results["competitor_analysis"] = comp_analysis

    elif intent == "shipping_query":
        shipping = registry.execute_skill(
            "estimate_shipping",
            customer_state=state.get("customer_state", "SP"),
            seller_state=state.get("seller_state", "SP"),
        )
        skill_results["shipping"] = shipping

    elif intent == "general_question":
        product_info = registry.execute_skill(
            "get_product_info",
            product_category=state["product_category"],
        )
        skill_results["product_info"] = product_info

    elif intent == "acceptance":
        deal = registry.execute_skill(
            "finalize_deal",
            final_price=state["current_offer"],
            floor_price=state["floor_price"],
            target_price=state["target_price"],
            opening_price=state["opening_price"],
            value_adds_offered=state.get("value_adds_offered", []),
            negotiation_round=state.get("negotiation_round", 1),
            product_category=state["product_category"],
        )
        skill_results["deal"] = deal

    elif intent == "rejection":
        walkaway = registry.execute_skill(
            "generate_walkaway_response",
            current_offer=state["current_offer"],
            floor_price=state["floor_price"],
            value_adds_offered=state.get("value_adds_offered", []),
        )
        skill_results["walkaway"] = walkaway

    # Build skill context string for the LLM
    skill_context = _format_skill_results(skill_results)
    logger.info(f"  Skills executed: {list(skill_results.keys())}")

    return {"last_tool_result": skill_context}


def _extract_price_from_messages(messages: list) -> float:
    """Extract a numeric price from the most recent user message."""
    import re
    for msg in reversed(messages):
        content = ""
        if isinstance(msg, HumanMessage):
            content = extract_text(msg.content)
        elif isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")

        if content:
            # Match $1,234 or $1234 or 1234 patterns
            matches = re.findall(r'\$?([\d,]+(?:\.\d{2})?)', content)
            for match in matches:
                try:
                    val = float(match.replace(",", ""))
                    if 10 < val < 50000:  # reasonable price range
                        return val
                except ValueError:
                    continue
            break
    return 0.0


def _extract_competitor(messages: list) -> str:
    """Extract competitor name from the most recent user message."""
    competitors = ["dell", "amazon", "best buy", "hp", "lenovo", "asus", "acer"]
    for msg in reversed(messages):
        content = ""
        if isinstance(msg, HumanMessage):
            content = extract_text(msg.content).lower()
        elif isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "").lower()

        if content:
            for comp in competitors:
                if comp in content:
                    return comp
            break
    return "generic"


def _format_skill_results(results: dict) -> str:
    """Format skill results as structured context for the LLM."""
    sections = []

    if "tactic" in results and "error" not in results["tactic"]:
        t = results["tactic"]
        sections.append(
            f"## Recommended Tactic: {t.get('name', 'unknown')}\n"
            f"- Action: {t.get('action', '')}\n"
            f"- Price Action: {t.get('price_action', 'hold')}\n"
            f"- Tone: {t.get('tone', 'professional')}\n"
            f"- Current Margin: {t.get('current_margin_pct', 0):.1f}%\n"
            + (f"- Suggested Value-Add: {t.get('suggested_value_add', '')}\n"
               if t.get('recommend_value_add') else "")
        )

    if "counteroffer" in results and "error" not in results["counteroffer"]:
        c = results["counteroffer"]
        sections.append(
            f"## Counteroffer Calculation\n"
            f"- Recommended Price: ${c.get('recommended_price', 0):.2f}\n"
            f"- Action: {c.get('action', '')}\n"
            f"- {c.get('message', '')}\n"
        )

    if "gap_analysis" in results and "error" not in results["gap_analysis"]:
        g = results["gap_analysis"]
        sections.append(
            f"## Price Gap Analysis\n"
            f"- Customer Position: {g.get('assessment', '')} "
            f"({g.get('user_position_in_range_pct', 0):.0f}% of range)\n"
            f"- Gap from Current Offer: ${g.get('gap_from_offer', 0):.2f}\n"
            f"- Negotiation Room: ${g.get('negotiation_room', 0):.2f}\n"
        )

    if "competitor_analysis" in results and "error" not in results["competitor_analysis"]:
        ca = results["competitor_analysis"]
        points = "\n".join(f"  - {p}" for p in ca.get("talking_points", []))
        sections.append(
            f"## Competitor Analysis: {ca.get('competitor', '')}\n"
            f"- Total Advantage Value: ${ca.get('total_advantage_value', 0):.0f}\n"
            f"- Strategy: {ca.get('strategy', '')}\n"
            f"- Key Points:\n{points}\n"
        )

    if "shipping" in results and "error" not in results["shipping"]:
        s = results["shipping"]
        sections.append(
            f"## Shipping Info\n"
            f"- {s.get('message', '')}\n"
            f"- Free Shipping Value: ${s.get('free_shipping_value', 0):.2f}\n"
        )

    if "product_info" in results and "error" not in results["product_info"]:
        pi = results["product_info"]
        if pi.get("found"):
            points = "\n".join(f"  - {p}" for p in pi.get("talking_points", []))
            sections.append(
                f"## Product Data (Real)\n"
                f"- Total Sales: {pi.get('total_sales', 0):,}\n"
                f"- Avg Review: {pi.get('avg_review_score', 0)}/5\n"
                f"- Demand: {pi.get('demand_level', 'unknown')}\n"
                f"- Talking Points:\n{points}\n"
            )

    if "deal" in results and "error" not in results["deal"]:
        d = results["deal"]
        sections.append(
            f"## Deal Summary\n"
            f"- {d.get('deal_summary', '')}\n"
            f"- Rating: {d.get('rating', '')}\n"
        )

    if "walkaway" in results and "error" not in results["walkaway"]:
        w = results["walkaway"]
        sections.append(
            f"## Walkaway Strategy\n"
            f"- {w.get('message_strategy', '')}\n"
            f"- Can Save: {w.get('should_attempt_save', False)}\n"
        )

    if not sections:
        return ""

    return "# SKILL RESULTS (use these facts, do NOT contradict)\n\n" + "\n".join(sections)


# ── Node 3: Agentic Core ────────────────────────────────────

def agentic_core_node(state: NegotiationState) -> dict:
    """
    Main reasoning node. Builds the intent-specific prompt,
    injects skill results, and invokes the LLM with tools.
    """
    logger.info("[Node 3] Agentic Core reasoning...")

    intent = state.get("intent", "greeting")
    intent_route = get_routing_decision(intent)

    # Build the system prompt with injected pricing context
    system_prompt = get_system_prompt(
        intent_route=intent_route,
        product_category=state["product_category"],
        opening_price=state["opening_price"],
        floor_price=state["floor_price"],
        target_price=state["target_price"],
        optimal_price=state["optimal_price"],
        current_offer=state["current_offer"],
        negotiation_round=state["negotiation_round"],
        value_adds_offered=state.get("value_adds_offered", []),
        conversion_probability=state.get("conversion_probability", 0.0),
    )

    # Inject skill results as immutable facts
    skill_context = state.get("last_tool_result", "")
    if skill_context:
        system_prompt += f"\n\n{skill_context}"
        system_prompt += (
            "\n\n## IMPORTANT: Skill Results Are Authoritative"
            "\nThe skill results above were computed by Python (deterministic)."
            "\nYou MUST follow the recommended tactic and use the calculated prices."
            "\nDo NOT invent your own prices — use calculate_counteroffer results."
            "\nDo NOT make up product facts — use Product Data if available."
        )

    # Prepare messages: system + conversation history
    llm_messages = [SystemMessage(content=system_prompt)]

    for msg in state["messages"]:
        if isinstance(msg, (HumanMessage, AIMessage, SystemMessage, ToolMessage)):
            llm_messages.append(msg)
        elif isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                llm_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                llm_messages.append(AIMessage(content=content))

    # Call LLM with tools (skills are pre-computed; tools are for offer submission)
    llm = get_core_llm()

    # Combine legacy tools + skill tool schemas
    registry = get_registry()
    tools = [SUBMIT_OFFER_TOOL_SCHEMA, ADD_VALUE_TOOL_SCHEMA] + registry.get_tool_schemas()

    response = llm.invoke(
        llm_messages,
        tools=tools,
    )

    return {"messages": [response]}


# ── Node 4: Tool Executor ────────────────────────────────────

def tool_executor_node(state: NegotiationState) -> dict:
    """
    Execute tool calls from the LLM response.
    Handles both legacy tools AND skill-based tools.
    """
    logger.info("[Node 4] Executing tools...")

    messages = state["messages"]
    last_msg = messages[-1] if messages else None

    if not last_msg or not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return {"last_tool_result": ""}

    tool_results = []
    current_offer = state["current_offer"]
    value_adds = list(state.get("value_adds_offered", []))
    registry = get_registry()

    for tool_call in last_msg.tool_calls:
        tool_name = tool_call["name"]
        args = tool_call["args"]

        # ── Legacy tools ──────────────────────────────────────
        if tool_name == "submit_official_offer":
            result = submit_official_offer(
                price=args["price"],
                justification=args.get("justification", ""),
                floor_price=state["floor_price"],
                target_price=state["target_price"],
                current_round=state["negotiation_round"],
            )

            if result.approved:
                current_offer = result.offered_price

            tool_msg = ToolMessage(
                content=result.message,
                tool_call_id=tool_call["id"],
            )
            tool_results.append(tool_msg)

        elif tool_name == "add_value_proposition":
            result = add_value_proposition(
                value_type=args["value_type"],
                current_offer=current_offer,
                floor_price=state["floor_price"],
            )
            if "error" not in result:
                value_adds.append(args["value_type"])

            tool_msg = ToolMessage(
                content=json.dumps(result),
                tool_call_id=tool_call["id"],
            )
            tool_results.append(tool_msg)

        # ── Skill-based tools (dynamic dispatch) ──────────────
        else:
            skill = registry.get(tool_name)
            if skill:
                # Inject state-level args for skills that need them
                if skill.requires_state:
                    args = {
                        **args,
                        "current_offer": state["current_offer"],
                        "floor_price": state["floor_price"],
                        "target_price": state["target_price"],
                        "opening_price": state.get("opening_price", 0),
                        "negotiation_round": state.get("negotiation_round", 1),
                        "value_adds_offered": state.get("value_adds_offered", []),
                        "customer_state": state.get("customer_state", "SP"),
                        "seller_state": state.get("seller_state", "SP"),
                        "product_category": state.get("product_category", ""),
                        "intent": state.get("intent", ""),
                    }

                result = registry.execute_skill(tool_name, **args)
                tool_msg = ToolMessage(
                    content=json.dumps(result, default=str),
                    tool_call_id=tool_call["id"],
                )
                tool_results.append(tool_msg)

                # Update state if skill returned a recommended price
                if "recommended_price" in result and result.get("action") == "accept":
                    current_offer = result["recommended_price"]
            else:
                tool_msg = ToolMessage(
                    content=json.dumps({"error": f"Unknown tool: {tool_name}"}),
                    tool_call_id=tool_call["id"],
                )
                tool_results.append(tool_msg)

    update = {
        "messages": tool_results,
        "current_offer": current_offer,
        "value_adds_offered": value_adds,
    }

    if tool_results:
        update["last_tool_result"] = tool_results[-1].content

    return update


# ── Node 5: Generator ────────────────────────────────────────

def generator_node(state: NegotiationState) -> dict:
    """
    Generate the final customer-facing response after tool execution.
    If tools were called and there's a tool result, the LLM
    generates text that naturally incorporates the approved price.
    """
    logger.info("[Node 5] Generating response...")

    messages = state["messages"]
    last_msg = messages[-1] if messages else None

    # If the last message is already a plain AI message (no tool calls),
    # just return it as-is
    if isinstance(last_msg, AIMessage) and (not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls):
        if extract_text(last_msg.content):
            return {}

    # If we have tool results, ask LLM to generate a natural response
    if isinstance(last_msg, ToolMessage):
        intent_route = get_routing_decision(state.get("intent", "negotiation"))
        system_prompt = get_system_prompt(
            intent_route=intent_route,
            product_category=state["product_category"],
            opening_price=state["opening_price"],
            floor_price=state["floor_price"],
            target_price=state["target_price"],
            optimal_price=state["optimal_price"],
            current_offer=state["current_offer"],
            negotiation_round=state["negotiation_round"],
            value_adds_offered=state.get("value_adds_offered", []),
            conversion_probability=state.get("conversion_probability", 0.0),
        )

        system_prompt += (
            "\n\n## Instruction"
            "\nGenerate a natural, conversational response to the customer "
            "that incorporates the tool/skill results above. Do NOT use "
            "submit_official_offer again — just write your response text."
            "\nAlways use the recommended tone from the tactic selector."
        )

        llm_messages = [SystemMessage(content=system_prompt)]
        for msg in messages:
            if isinstance(msg, (HumanMessage, AIMessage, ToolMessage)):
                llm_messages.append(msg)

        llm = get_core_llm()
        response = llm.invoke(llm_messages)

        deal_closed = state.get("intent") == "acceptance"

        return {
            "messages": [response],
            "deal_closed": deal_closed,
        }

    return {}


# ── Routing Logic ─────────────────────────────────────────────

def should_execute_tools(state: NegotiationState) -> Literal["tool_executor", "generator"]:
    """Determine if the LLM response contains tool calls."""
    messages = state["messages"]
    if not messages:
        return "generator"

    last_msg = messages[-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tool_executor"
    return "generator"


def should_continue_after_tools(state: NegotiationState) -> Literal["agentic_core", "generator"]:
    """After tool execution, check if the tool was rejected (needs retry)."""
    last_result = state.get("last_tool_result", "")
    if "REJECTED" in last_result or "Error:" in last_result:
        logger.info("  Tool rejected offer, routing back to agentic core for retry")
        return "agentic_core"
    return "generator"


# ── Graph Construction ────────────────────────────────────────

def create_negotiation_graph() -> StateGraph:
    """
    Build and compile the LangGraph negotiation pipeline.
    
    Flow:
        router → skill_selector → agentic_core → [tool_executor ↔ agentic_core] → generator
    """

    graph = StateGraph(NegotiationState)

    # Add nodes (5-node pipeline)
    graph.add_node("router", router_node)
    graph.add_node("skill_selector", skill_selector_node)
    graph.add_node("agentic_core", agentic_core_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("generator", generator_node)

    # Set entry point
    graph.set_entry_point("router")

    # Define edges
    graph.add_edge("router", "skill_selector")           # NEW: skills run before LLM
    graph.add_edge("skill_selector", "agentic_core")     # Skill results → LLM

    graph.add_conditional_edges(
        "agentic_core",
        should_execute_tools,
        {
            "tool_executor": "tool_executor",
            "generator": "generator",
        },
    )

    graph.add_conditional_edges(
        "tool_executor",
        should_continue_after_tools,
        {
            "agentic_core": "agentic_core",
            "generator": "generator",
        },
    )

    graph.add_edge("generator", END)

    return graph.compile()


# ── Convenience Runner ────────────────────────────────────────

_compiled_graph = None


def get_graph():
    """Get or create the compiled negotiation graph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = create_negotiation_graph()
    return _compiled_graph



def run_negotiation_turn(
    state: NegotiationState,
    user_message: str,
) -> tuple[NegotiationState, str]:
    """
    Process a single turn of negotiation.
    
    Args:
        state: Current negotiation state
        user_message: The customer's message
    
    Returns:
        Tuple of (updated_state, agent_response_text)
    """
    state["messages"].append(HumanMessage(content=user_message))

    graph = get_graph()
    result = graph.invoke(state)

    response_text = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            response_text = extract_text(msg.content)
            break

    return result, response_text

