"""
Text processing utilities.
Sentence splitting that handles multiple languages including Japanese.
"""

import re
from typing import List


def normalize(text: str) -> str:
    """Collapse whitespace, strip edges."""
    if not text:
        return ""
    return " ".join(text.split()).strip()


def split_sentences(text: str, lang: str = "en") -> List[str]:
    """
    Split text into sentences.

    For Japanese (lang='ja'): split on Japanese sentence-ending markers (。！？)
    and also on standard punctuation followed by space + uppercase.

    For all other languages: regex-based split on sentence boundaries.

    Returns list of non-trivial sentences (len > 5 chars).
    """
    if not text or not isinstance(text, str):
        return []

    text = normalize(text)
    if not text:
        return []

    if lang == "ja":
        # Japanese uses 。！？ as sentence enders
        # Also handle mixed content with Western punctuation
        raw = re.split(r'(?<=[。！？])\s*|(?<=[.!?])\s+(?=[A-Z\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff])', text)
    else:
        # Western languages: split on .!? followed by whitespace + uppercase letter
        # Handles abbreviations (e.g., Dr., U.S.) and decimal numbers better
        raw = re.split(r'(?<=[.!?])\s+(?=[A-Z\u00C0-\u00DC])', text)

    sentences = [normalize(s) for s in raw if s]
    return [s for s in sentences if len(s) > 5]
