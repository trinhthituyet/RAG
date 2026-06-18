"""Helpers for parsing JSON out of LLM replies."""


def extract_json(text: str) -> str:
    """Extract a JSON object from an LLM reply that may wrap it in prose or ``` fences."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text.strip()
