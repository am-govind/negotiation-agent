"""
Gemini Response Utilities.

gemini-2.5-flash (and other thinking models) return response.content as a
list of parts instead of a plain string:

    [
      {"type": "thinking", "thinking": "Let me consider..."},
      {"type": "text", "text": "Hello, welcome!"}
    ]

This module provides helpers to normalize any Gemini response format
into plain strings or parsed JSON.
"""
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def extract_text(content: Any) -> str:
    """Extract plain text from a Gemini response content field.

    Handles:
      - Plain strings (returned as-is)
      - List of parts (thinking + text dicts)
      - Nested or unexpected formats (converted via str())

    Args:
        content: The `.content` attribute from a LangChain message or
                 Gemini API response.

    Returns:
        A plain text string with thinking tokens removed.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                # Skip thinking tokens
                if part.get("type") == "thinking":
                    continue
                # Extract text from text-type parts
                if "text" in part:
                    text_parts.append(part["text"])
                elif "content" in part:
                    # Nested content field
                    text_parts.append(extract_text(part["content"]))
            elif isinstance(part, str):
                text_parts.append(part)
            else:
                text_parts.append(str(part))
        return "".join(text_parts)
    return str(content)


def extract_json(content: Any) -> dict:
    """Extract and parse JSON from a Gemini response content field.

    Handles:
      - Plain JSON strings
      - List-of-parts content (extracts text, skips thinking)
      - Markdown code fences (```json ... ```)
      - Leading/trailing whitespace

    Args:
        content: The `.content` attribute from a LangChain message.

    Returns:
        Parsed dict from the JSON content.

    Raises:
        json.JSONDecodeError: If no valid JSON can be extracted.
    """
    raw = extract_text(content).strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        # Remove opening fence (```json or ```)
        raw = raw.split("\n", 1)[-1] if "\n" in raw else raw[3:]
        # Remove closing fence
        raw = raw.rsplit("```", 1)[0].strip()

    # Try to extract JSON object if there's surrounding text
    if not raw.startswith("{"):
        match = re.search(r'\{[^{}]*\}', raw)
        if match:
            raw = match.group(0)

    return json.loads(raw)


def safe_content(msg: Any) -> str:
    """Normalize a LangChain message's .content to a plain string.

    Convenience wrapper that reads the `.content` attribute from a
    message object and passes it through `extract_text()`.

    Args:
        msg: A LangChain message object (HumanMessage, AIMessage, etc.)
             or a dict with a "content" key.

    Returns:
        Plain text string.
    """
    if hasattr(msg, "content"):
        return extract_text(msg.content)
    if isinstance(msg, dict) and "content" in msg:
        return extract_text(msg["content"])
    return str(msg)
