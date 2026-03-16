"""Sentence embedding using sentence-transformers. Model loaded ONCE globally at module level."""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings

# Load model ONCE globally. Do NOT load inside functions.
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Return the globally loaded model (lazy load on first use)."""
    global _model
    if _model is None:
        model_name = get_settings().embedding_model
        _model = SentenceTransformer(model_name)
    return _model


def encode_sentences(sentences: List[str]) -> np.ndarray:
    """
    Encode a list of sentences into embeddings using the global model.
    Returns numpy array of shape (n_sentences, embedding_dim).
    """
    if not sentences:
        return np.array([]).reshape(0, 384)
    model = _get_model()
    embeddings = model.encode(
        sentences,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.atleast_2d(embeddings)
