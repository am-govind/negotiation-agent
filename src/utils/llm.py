"""
Centralized LLM Factory: Single source for all LLM instances.

Supports HuggingFace Inference API (primary) with Gemini fallback.
Configure via environment variables in .env:
  HF_API_TOKEN  — HuggingFace API token (enables HF models)
  HF_MODEL      — Primary model (default: Qwen/Qwen2.5-72B-Instruct)
  HF_BACKUP_MODEL — Backup model (default: meta-llama/Llama-3.1-70B-Instruct)
  GOOGLE_API_KEY — Gemini API key (fallback if HF is not configured)

Usage:
    from src.utils.llm import get_router_llm, get_core_llm, get_buyer_llm
"""
import os
import logging

logger = logging.getLogger(__name__)

# ── Cached instances ──────────────────────────────────────────
_router_llm = None
_core_llm = None
_buyer_llm = None


def _get_hf_token() -> str | None:
    """Return HF API token if set."""
    return os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")


def _get_hf_model() -> str:
    return os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")


def _get_hf_backup_model() -> str:
    return os.getenv("HF_BACKUP_MODEL", "meta-llama/Llama-3.1-70B-Instruct")


def _create_hf_chat(model: str, temperature: float, max_tokens: int):
    """Create a HuggingFace chat model via Inference API."""
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    endpoint = HuggingFaceEndpoint(
        repo_id=model,
        huggingfacehub_api_token=_get_hf_token(),
        task="text-generation",
        max_new_tokens=max_tokens,
        temperature=max(temperature, 0.01),  # HF doesn't allow exactly 0
    )
    return ChatHuggingFace(
        llm=endpoint,
        model_id=model,
        verbose=False,
    )


def _create_gemini_chat(model: str, temperature: float, max_tokens: int):
    """Create a Gemini chat model (fallback)."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )


def _create_chat(role: str, temperature: float, max_tokens: int):
    """
    Create the best available chat model.
    Priority: HuggingFace → Gemini → Error
    """
    hf_token = _get_hf_token()

    if hf_token:
        model = _get_hf_model()
        try:
            llm = _create_hf_chat(model, temperature, max_tokens)
            logger.info(f"[{role}] Using HuggingFace model: {model}")
            return llm
        except Exception as e:
            logger.warning(f"[{role}] HF primary model failed ({model}): {e}")
            # Try backup
            backup = _get_hf_backup_model()
            try:
                llm = _create_hf_chat(backup, temperature, max_tokens)
                logger.info(f"[{role}] Using HuggingFace backup: {backup}")
                return llm
            except Exception as e2:
                logger.warning(f"[{role}] HF backup failed ({backup}): {e2}")

    # Fallback to Gemini
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        gemini_model = os.getenv("CORE_MODEL", "gemini-2.5-flash")
        logger.info(f"[{role}] Falling back to Gemini: {gemini_model}")
        return _create_gemini_chat(gemini_model, temperature, max_tokens)

    raise RuntimeError(
        "No LLM backend configured. Set HF_API_TOKEN for HuggingFace "
        "or GOOGLE_API_KEY for Gemini in your .env file."
    )


# ── Public API ────────────────────────────────────────────────

def get_router_llm():
    """Get the lightweight router LLM (intent classification)."""
    global _router_llm
    if _router_llm is None:
        _router_llm = _create_chat("router", temperature=0.01, max_tokens=100)
    return _router_llm


def get_core_llm():
    """Get the main negotiation LLM (reasoning + response generation)."""
    global _core_llm
    if _core_llm is None:
        _core_llm = _create_chat("core", temperature=0.7, max_tokens=500)
    return _core_llm


def get_buyer_llm():
    """Get the buyer simulation LLM (arena testing)."""
    global _buyer_llm
    if _buyer_llm is None:
        _buyer_llm = _create_chat("buyer", temperature=0.8, max_tokens=150)
    return _buyer_llm


def get_provider_info() -> dict:
    """Return info about the currently configured LLM provider."""
    hf_token = _get_hf_token()
    if hf_token:
        return {
            "provider": "huggingface",
            "model": _get_hf_model(),
            "backup_model": _get_hf_backup_model(),
        }
    elif os.getenv("GOOGLE_API_KEY"):
        return {
            "provider": "gemini",
            "model": os.getenv("CORE_MODEL", "gemini-2.5-flash"),
        }
    return {"provider": "none", "model": "not configured"}
