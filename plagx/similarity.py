"""
Similarity computation module.

Three mathematical approaches combined:

1. **Semantic Cosine Similarity** (cross-lingual, embedding space)
   - Uses multilingual sentence embeddings directly.
   - Works across languages WITHOUT translation.
   - sim(a, b) = (a · b) / (||a|| ||b||)  — but since embeddings are L2-normalized,
     this simplifies to sim(a, b) = a · b (dot product).

2. **TF-IDF Cosine Similarity** (on translated English text)
   - Translates reference sentences → English, then computes TF-IDF vectors.
   - Captures lexical overlap that embedding models might miss.
   - TF-IDF(t,d) = tf(t,d) × log(N / df(t))
   - cosine_sim(d1, d2) = (d1 · d2) / (||d1|| ||d2||)

3. **Combined Score** (weighted fusion)
   - final = W_SEMANTIC × semantic_sim + W_TFIDF × tfidf_sim
   - Default: 0.7 × semantic + 0.3 × tfidf
   - Captures both deep meaning (semantic) and surface form (tfidf).
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple


def semantic_similarity_matrix(query_emb: np.ndarray,
                               ref_emb: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity between query and reference embeddings.

    Since embeddings are L2-normalized:
        cosine_sim(q, r) = q · r^T ∈ [-1, 1]

    Args:
        query_emb: (n_query, dim) normalized embeddings
        ref_emb:   (n_ref, dim) normalized embeddings

    Returns:
        (n_query, n_ref) similarity matrix
    """
    q = np.atleast_2d(query_emb)
    r = np.atleast_2d(ref_emb)
    return cosine_similarity(q, r)


def tfidf_similarity_matrix(query_sentences: List[str],
                            ref_sentences_english: List[str]) -> np.ndarray:
    """
    Compute TF-IDF cosine similarity between query (English) and
    reference sentences (translated to English).

    TF-IDF weighting:
        w(t,d) = (1 + log(tf(t,d))) × log(N / df(t))

    Cosine similarity on TF-IDF vectors captures lexical/surface-form overlap
    that neural embeddings might miss (e.g., exact technical terms).

    Args:
        query_sentences: English input sentences
        ref_sentences_english: Reference sentences translated to English

    Returns:
        (n_query, n_ref) similarity matrix with values in [0, 1]
    """
    if not query_sentences or not ref_sentences_english:
        return np.zeros((len(query_sentences), len(ref_sentences_english)))

    all_sentences = query_sentences + ref_sentences_english
    n_q = len(query_sentences)

    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,       # Use 1 + log(tf) instead of raw tf
        analyzer="word",
        ngram_range=(1, 2),      # Unigrams + bigrams for phrase matching
        min_df=1,
        # Keep shared terms in tiny corpora (e.g., query vs single translation pair).
        # With max_df=0.95 and only 2 documents, overlap terms are dropped (df=1.0).
        max_df=1.0,
        strip_accents="unicode",
        lowercase=True,
    )

    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_df=1.0,
        lowercase=True,
    )

    tfidf_word = word_vectorizer.fit_transform(all_sentences)
    q_word = tfidf_word[:n_q]
    r_word = tfidf_word[n_q:]
    word_sim = cosine_similarity(q_word, r_word).astype(np.float32)

    tfidf_char = char_vectorizer.fit_transform(all_sentences)
    q_char = tfidf_char[:n_q]
    r_char = tfidf_char[n_q:]
    char_sim = cosine_similarity(q_char, r_char).astype(np.float32)

    # Use the stronger lexical signal per pair to stay robust to paraphrasing.
    return np.maximum(word_sim, char_sim)


def combined_similarity_matrix(semantic_sim: np.ndarray,
                               tfidf_sim: np.ndarray,
                               w_semantic: float = 0.7,
                               w_tfidf: float = 0.3) -> np.ndarray:
    """
    Weighted fusion of semantic and TF-IDF similarity matrices.

    final(i,j) = w_semantic × semantic(i,j) + w_tfidf × tfidf(i,j)

    The combined score leverages:
    - Semantic similarity: captures MEANING even across languages/paraphrases
    - TF-IDF similarity: captures LEXICAL overlap (exact terms, names, numbers)

    Args:
        semantic_sim: (n_q, n_r) semantic similarities
        tfidf_sim: (n_q, n_r) tf-idf similarities (must be same shape)
        w_semantic: weight for semantic component (default 0.7)
        w_tfidf: weight for tfidf component (default 0.3)

    Returns:
        (n_q, n_r) combined similarity matrix
    """
    # Ensure shapes match
    assert semantic_sim.shape == tfidf_sim.shape, \
        f"Shape mismatch: semantic {semantic_sim.shape} vs tfidf {tfidf_sim.shape}"

    return (w_semantic * semantic_sim + w_tfidf * tfidf_sim).astype(np.float32)


def find_best_matches(sim_matrix: np.ndarray,
                      query_sentences: List[str],
                      ref_sentences: List[str],
                      threshold: float) -> List[Tuple[int, int, float]]:
    """
    For each query sentence, find the best-matching reference sentence
    if the similarity exceeds the threshold.

    Args:
        sim_matrix: (n_query, n_ref) similarity scores
        query_sentences: input sentences
        ref_sentences: reference sentences
        threshold: minimum similarity to flag as match

    Returns:
        List of (query_idx, ref_idx, score) tuples
    """
    matches = []
    n_q = min(sim_matrix.shape[0], len(query_sentences))
    n_r = min(sim_matrix.shape[1], len(ref_sentences))

    if n_q == 0 or n_r == 0:
        return matches

    for i in range(n_q):
        j = int(np.argmax(sim_matrix[i, :n_r]))
        score = float(sim_matrix[i, j])
        if score >= threshold:
            matches.append((i, j, score))

    return matches
