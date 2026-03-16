"""
Embedding service using sentence-transformers.
Uses paraphrase-multilingual-MiniLM-L12-v2 which supports 50+ languages
and maps semantically similar sentences (even across languages) to nearby vectors.

Model is loaded ONCE and cached globally.
"""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from plagx.config import EMBEDDING_MODEL

LOG = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the multilingual sentence embedding model."""
    global _model
    if _model is None:
        LOG.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def encode(sentences: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Encode sentences into normalized embedding vectors.

    Args:
        sentences: List of text strings (any language).
        batch_size: Encoding batch size.

    Returns:
        np.ndarray of shape (len(sentences), 384) with L2-normalized embeddings.
        Cosine similarity between two normalized vectors = their dot product.
    """
    if not sentences:
        return np.empty((0, 384), dtype=np.float32)

    model = _get_model()
    embeddings = model.encode(
        sentences,
        normalize_embeddings=True,   # L2-normalize -> dot product = cosine sim
        show_progress_bar=False,
        batch_size=batch_size,
    )
    return np.atleast_2d(embeddings).astype(np.float32)
