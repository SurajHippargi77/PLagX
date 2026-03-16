"""
Reference document store. Static Dutch papers loaded from disk at startup.
Embeddings computed once; reference documents are read-only (no API modifications).
"""

from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

# Structure: reference_documents[filename] = {"sentences": [...], "embeddings": np.ndarray}
reference_documents: Dict[str, Dict[str, Any]] = {}


def load_document(file_path: Path) -> Optional[str]:
    """Load text from a .txt file. Returns None on error."""
    try:
        return file_path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return None


def add_document(filename: str, sentences: List[str], embeddings: np.ndarray) -> None:
    """
    Add a precomputed document to the store. Internal use only (startup).
    Reference documents cannot be modified via API.
    """
    reference_documents[filename] = {
        "sentences": list(sentences),
        "embeddings": embeddings,
    }


def get_all_for_comparison() -> List[tuple[str, List[str], np.ndarray]]:
    """Return list of (filename, sentences, embeddings) for all reference documents."""
    out: List[tuple[str, List[str], np.ndarray]] = []
    for filename, data in reference_documents.items():
        s = data.get("sentences", [])
        e = data.get("embeddings")
        if s and e is not None:
            out.append((filename, s, e))
    return out


def list_documents() -> List[Dict[str, Any]]:
    """List all reference documents (read-only info for API)."""
    return [
        {
            "filename": name,
            "sentence_count": len(data.get("sentences", [])),
        }
        for name, data in reference_documents.items()
    ]
