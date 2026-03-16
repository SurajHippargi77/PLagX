"""
Load reference Dutch documents from disk, split into sentences, generate embeddings in batch.
Called once at application startup. Embeddings computed only once.
"""

import logging
from pathlib import Path

from app.text_utils import split_into_sentences
from app.services import embedding_service
from app.reference_store import add_document, load_document

LOG = logging.getLogger(__name__)


def load_reference_documents_from_directory(directory: Path) -> int:
    """
    Load all .txt files from directory, extract text, split sentences, batch-embed, store.
    Returns count of documents loaded. Embeddings computed once.
    """
    if not directory.is_dir():
        LOG.warning("Reference directory is not a directory: %s", directory)
        return 0

    count = 0
    txt_files = sorted(directory.glob("*.txt"))

    if not txt_files:
        LOG.warning("No .txt files found in %s", directory)
        return 0

    all_sentences_by_file: list[tuple[str, list[str]]] = []
    for fp in txt_files:
        text = load_document(fp)
        if not text:
            continue
        sentences = split_into_sentences(text)
        if not sentences:
            continue
        all_sentences_by_file.append((fp.stem, sentences))

    if not all_sentences_by_file:
        return 0

    # Batch embed all sentences across all documents (single model call for efficiency)
    all_sentences = []
    file_boundaries: list[tuple[str, int, int]] = []  # (filename, start_idx, end_idx)
    idx = 0
    for filename, sents in all_sentences_by_file:
        start = idx
        idx += len(sents)
        all_sentences.extend(sents)
        file_boundaries.append((filename, start, idx))

    try:
        all_embeddings = embedding_service.encode_sentences(all_sentences)
    except Exception as e:
        LOG.error("Failed to generate embeddings: %s", e)
        return 0

    for filename, start, end in file_boundaries:
        doc_sentences = all_sentences[start:end]
        doc_embeddings = all_embeddings[start:end]
        add_document(filename, doc_sentences, doc_embeddings)
        count += 1

    LOG.info("Loaded %d reference document(s) with precomputed embeddings.", count)
    return count
