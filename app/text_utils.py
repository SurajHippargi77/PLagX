"""Text processing utilities: sentence splitting and normalization."""

import re
from typing import List


def normalize_text(text: str) -> str:
    """Normalize whitespace and strip leading/trailing spaces."""
    if not text or not isinstance(text, str):
        return ""
    return " ".join(text.split()).strip()


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences. Handles common abbreviations and decimal numbers.
    Returns non-empty, normalized sentences.
    """
    if not text or not isinstance(text, str):
        return []

    normalized = normalize_text(text)
    if not normalized:
        return []

    pattern = r"(?<![A-Za-z])(?<!\d)([.!?])\s+(?=[A-Z])"
    parts = re.split(pattern, normalized)

    sentences: List[str] = []
    current = ""

    for i, part in enumerate(parts):
        if part in ".!?":
            current = (current + part).strip()
            if current:
                sentences.append(current)
            current = ""
        else:
            current = (current + " " + part).strip() if current else part.strip()

    if current:
        sentences.append(current)

    return [s for s in sentences if len(s) > 2]
