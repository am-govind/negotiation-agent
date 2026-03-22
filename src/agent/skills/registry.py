"""
Skills Registry: Central registry for all executable agent skills.

Skills are concrete Python functions that perform deterministic,
programmatic work — unlike prompts which are probabilistic.
The LLM orchestrates WHICH skill to call, but the skill execution
is pure Python with guaranteed behavior.

Architecture:
    Prompt-only approach:  LLM → generates text → hope it's correct
    Skills approach:       LLM → selects skill → Python executes → deterministic result → LLM formats

Each skill is a callable with:
    - name: unique identifier
    - description: what it does (fed to LLM for selection)
    - parameters: typed inputs
    - execute(): the actual Python logic
    - tool_schema: LangChain-compatible function calling schema
"""
import logging
from typing import Callable, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """A registered executable skill."""
    name: str
    description: str
    category: str  # pricing, product, shipping, negotiation, closing
    execute: Callable[..., dict]
    tool_schema: dict
    requires_state: bool = False  # Whether skill needs NegotiationState context


class SkillRegistry:
    """
    Central registry for all agent skills.
    
    Skills are loaded at startup and made available to the LangGraph
    pipeline as executable tools. The LLM chooses which skill to invoke,
    but the execution is deterministic Python.
    """

    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill):
        """Register a skill."""
        self._skills[skill.name] = skill
        logger.info(f"  Registered skill: {skill.name} [{skill.category}]")

    def get(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def get_all(self) -> list[Skill]:
        """Get all registered skills."""
        return list(self._skills.values())

    def get_by_category(self, category: str) -> list[Skill]:
        """Get skills filtered by category."""
        return [s for s in self._skills.values() if s.category == category]

    def get_tool_schemas(self) -> list[dict]:
        """Get all skill tool schemas for LLM function calling."""
        return [s.tool_schema for s in self._skills.values()]

    def execute_skill(self, name: str, **kwargs) -> dict:
        """Execute a skill by name with given arguments."""
        skill = self._skills.get(name)
        if not skill:
            return {"error": f"Unknown skill: {name}"}

        try:
            logger.info(f"  Executing skill: {name}")
            result = skill.execute(**kwargs)
            logger.info(f"  Skill {name} completed successfully")
            return result
        except Exception as e:
            logger.error(f"  Skill {name} failed: {e}")
            return {"error": f"Skill execution failed: {str(e)}"}

    def list_skills(self) -> list[dict]:
        """List all skills with metadata (for debugging/dashboard)."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "category": s.category,
                "requires_state": s.requires_state,
            }
            for s in self._skills.values()
        ]


# ── Global singleton ──────────────────────────────────────────
_registry: SkillRegistry | None = None


def get_registry() -> SkillRegistry:
    """Get or create the global skill registry."""
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
        _register_all_skills(_registry)
    return _registry


def _register_all_skills(registry: SkillRegistry):
    """Register all built-in skills."""
    logger.info("Registering agent skills...")

    from src.agent.skills.pricing import register_pricing_skills
    from src.agent.skills.product_knowledge import register_product_skills
    from src.agent.skills.shipping import register_shipping_skills
    from src.agent.skills.competitor import register_competitor_skills
    from src.agent.skills.negotiation_tactics import register_negotiation_skills
    from src.agent.skills.deal_closer import register_deal_skills

    register_pricing_skills(registry)
    register_product_skills(registry)
    register_shipping_skills(registry)
    register_competitor_skills(registry)
    register_negotiation_skills(registry)
    register_deal_skills(registry)

    logger.info(f"  Total skills registered: {len(registry.get_all())}")
