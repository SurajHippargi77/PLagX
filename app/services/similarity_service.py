"""Cosine similarity using sklearn. Threshold configurable in config.py (SIMILARITY_THRESHOLD)."""

from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.config import get_settings


def _threshold() -> float:
    """Similarity threshold from config (default 0.80). Config key: similarity_threshold."""
    return get_settings().similarity_threshold


def cosine_similarity_score(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between sets of vectors using sklearn.
    a: (n_a, dim), b: (n_b, dim) -> (n_a, n_b).
    Returns float similarity scores in [0, 1] when normalized.
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    return cosine_similarity(a, b)


def find_matches(
    query_embeddings: np.ndarray,
    query_sentences: List[str],
    ref_embeddings: np.ndarray,
    ref_sentences: List[str],
    ref_paper_id: str,
    threshold: float | None = None,
) -> List[Tuple[str, str, float]]:
    """
    For each query sentence, find best-matching reference sentence if above threshold.
    Returns list of (query_sentence, ref_sentence, similarity: float).
    """
    th = threshold if threshold is not None else _threshold()
    if query_embeddings.size == 0 or ref_embeddings.size == 0:
        return []

    sim = cosine_similarity_score(query_embeddings, ref_embeddings)
    matches: List[Tuple[str, str, float]] = []

    for i in range(sim.shape[0]):
        j = int(np.argmax(sim[i]))
        score = float(sim[i, j])
        if score >= th and i < len(query_sentences) and j < len(ref_sentences):
            matches.append((query_sentences[i], ref_sentences[j], score))

    return matches
