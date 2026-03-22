---
description: Add a new executable skill to the agent's skill registry
---

# Add a New Skill

Skills are deterministic Python functions that the LLM orchestrates but does NOT execute. This ensures reliable, testable behavior for critical operations like pricing, product lookups, and competitive analysis.

## Architecture

```
Prompt-only:  LLM → generates text → hope it's correct
Skills + LLM: LLM → selects skill → Python executes → deterministic result → LLM formats naturally
```

## Steps

1. Create a new file in `src/agent/skills/`:
```python
# src/agent/skills/my_new_skill.py
from src.agent.skills.registry import Skill, SkillRegistry


def _my_skill_function(param1: str, param2: float, **kwargs) -> dict:
    """Pure Python logic — no LLM calls here."""
    result = do_computation(param1, param2)
    return {
        "output": result,
        "message": f"Computed: {result}",
    }


def register_my_skills(registry: SkillRegistry):
    registry.register(Skill(
        name="my_skill_name",
        description="What this skill does (shown to LLM for tool selection)",
        category="my_category",
        execute=_my_skill_function,
        requires_state=True,  # Set True if skill needs NegotiationState fields
        tool_schema={
            "type": "function",
            "function": {
                "name": "my_skill_name",
                "description": "When to call this skill",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": "..."},
                    },
                    "required": ["param1"],
                },
            },
        },
    ))
```

2. Register it in `src/agent/skills/registry.py`:
```python
# In _register_all_skills():
from src.agent.skills.my_new_skill import register_my_skills
register_my_skills(registry)
```

3. (Optional) Add intent-based auto-triggering in `graph.py` → `skill_selector_node()`:
```python
elif intent == "my_intent":
    result = registry.execute_skill("my_skill_name", param1=value)
    skill_results["my_key"] = result
```

## Existing Skills

| Skill | Category | What It Does |
|-------|----------|-------------|
| `calculate_counteroffer` | pricing | Concession curve algorithm |
| `analyze_price_gap` | pricing | Gap analysis between positions |
| `get_product_info` | product | Real dataset queries |
| `compare_to_category_avg` | product | Price percentile ranking |
| `estimate_shipping` | shipping | Distance-tier cost/time |
| `check_delivery_feasibility` | shipping | Urgency-based delivery check |
| `analyze_competitor` | competitor | Feature comparison data |
| `select_tactic` | negotiation | Decision tree tactic selection |
| `finalize_deal` | closing | Deal packaging with metrics |
| `generate_walkaway_response` | closing | Graceful exit strategy |

## Key Principle
> Skills handle the WHAT (deterministic computation).
> The LLM handles the HOW (natural language formatting).
