"""
PLagX Detector – Full cross-lingual plagiarism detection pipeline.

Given an English text (or PDF), this module:
1. Loads reference documents in 5 languages (Dutch, Japanese, German, Spanish, Portuguese)
2. Splits all texts into sentences
3. Computes multilingual sentence embeddings (cross-lingual semantic similarity)
4. Finds candidate matches above a low semantic threshold
5. Translates ONLY the matched reference sentences → English (lazy translation)
6. Refines with TF-IDF cosine similarity on translated text
7. Computes combined score: 0.7×semantic + 0.3×tfidf
8. Reports per-language, per-document, and per-sentence plagiarism results

Mathematics:
    Semantic:   sim_s(q, r) = q·r / (‖q‖‖r‖)  (cosine on multilingual embeddings)
    Lexical:    sim_t(q, r') = tfidf(q)·tfidf(r') / (‖tfidf(q)‖‖tfidf(r')‖)
                where r' = translate(r → English)  [only matched candidates]
    Combined:   sim(q, r) = 0.7·sim_s + 0.3·sim_t
    Plagiarism% = |{q ∈ Q : max_r sim(q,r) ≥ threshold}| / |Q| × 100
"""

import logging
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyPDF2 import PdfReader

from plagx.config import (
    LANGUAGES, DATA_DIR, COMBINED_THRESHOLD, W_SEMANTIC, W_TFIDF,
    MAX_SENTENCES_PER_DOC,
)
from plagx.text_utils import split_sentences
from plagx.embeddings import encode
from plagx.translation import translate_to_english
from plagx.similarity import (
    semantic_similarity_matrix,
    tfidf_similarity_matrix,
    find_best_matches,
)

LOG = logging.getLogger(__name__)


# ── Data classes for results ─────────────────────────────────────────────────

@dataclass
class SentenceMatch:
    """A single flagged sentence match."""
    query_sentence: str          # English input sentence
    ref_sentence_original: str   # Reference sentence in original language
    ref_sentence_english: str    # Reference sentence translated to English
    language: str                # Source language code (e.g. 'nl')
    language_name: str           # Source language name (e.g. 'Dutch')
    document: str                # Source document filename
    semantic_score: float        # Cross-lingual embedding similarity
    tfidf_score: float           # TF-IDF similarity (after translation)
    combined_score: float        # Weighted fusion score


@dataclass
class DocumentResult:
    """Plagiarism result for one reference document."""
    filename: str
    language: str
    language_name: str
    num_ref_sentences: int
    matches: List[SentenceMatch] = field(default_factory=list)

    @property
    def match_count(self) -> int:
        return len(self.matches)

    @property
    def avg_score(self) -> float:
        if not self.matches:
            return 0.0
        return sum(m.combined_score for m in self.matches) / len(self.matches)


@dataclass
class LanguageResult:
    """Plagiarism result aggregated per language."""
    lang_code: str
    lang_name: str
    documents: List[DocumentResult] = field(default_factory=list)

    @property
    def total_matches(self) -> int:
        return sum(d.match_count for d in self.documents)

    @property
    def matched_query_sentences(self) -> set:
        s = set()
        for d in self.documents:
            for m in d.matches:
                s.add(m.query_sentence)
        return s


@dataclass
class PlagiarismReport:
    """Complete plagiarism analysis report."""
    input_sentences: List[str]
    total_input_sentences: int
    languages: Dict[str, LanguageResult] = field(default_factory=dict)
    threshold_used: float = COMBINED_THRESHOLD

    @property
    def all_matches(self) -> List[SentenceMatch]:
        out = []
        for lr in self.languages.values():
            for d in lr.documents:
                out.extend(d.matches)
        return out

    @property
    def flagged_sentences(self) -> set:
        """Set of unique input sentences that were flagged."""
        return {m.query_sentence for m in self.all_matches}

    @property
    def overall_plagiarism_pct(self) -> float:
        """% of input sentences that matched ANY reference document."""
        if self.total_input_sentences == 0:
            return 0.0
        return len(self.flagged_sentences) / self.total_input_sentences * 100

    def per_language_pct(self) -> Dict[str, float]:
        """% of input sentences that matched each language."""
        result = {}
        for code, lr in self.languages.items():
            unique = lr.matched_query_sentences
            pct = len(unique) / self.total_input_sentences * 100 if self.total_input_sentences else 0.0
            result[code] = pct
        return result


# ── Reference document loading ───────────────────────────────────────────────

@dataclass
class _RefDoc:
    """Internal: loaded reference document with precomputed embeddings only."""
    filename: str
    lang_code: str
    lang_name: str
    sentences: List[str]
    embeddings: np.ndarray
    # translations populated lazily (only for matched sentences)


def _read_reference_text(path: Path) -> str:
    """Read text from a reference file (.pdf or .txt)."""
    if path.suffix.lower() == ".pdf":
        try:
            reader = PdfReader(str(path))
            parts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
            return " ".join(parts).strip()
        except Exception as exc:
            LOG.warning("Failed to read reference PDF %s: %s", path.name, exc)
            return ""

    return path.read_text(encoding="utf-8", errors="replace").strip()


def _load_reference_docs(lang_codes: Optional[List[str]] = None) -> List[_RefDoc]:
    """
    Load and prepare all reference documents:
    1. Read .pdf files from plagx/data/reference_papers/<lang>/ (fallback .txt)
    2. Split into sentences
    3. Compute multilingual embeddings

    Translation is done LAZILY — only for sentences that matched a query.
    This avoids translating thousands of sentences upfront.
    """
    lang_codes = lang_codes or list(LANGUAGES.keys())
    docs: List[_RefDoc] = []

    for lc in lang_codes:
        if lc not in LANGUAGES:
            LOG.warning("Unknown language code: %s, skipping", lc)
            continue

        lang_info = LANGUAGES[lc]
        lang_dir = DATA_DIR / lc

        if not lang_dir.is_dir():
            LOG.warning("No data directory for %s: %s", lc, lang_dir)
            continue

        pdf_files = sorted(lang_dir.glob("*.pdf"))
        txt_files = sorted(lang_dir.glob("*.txt"))
        # Prefer text exports when present; PDF extraction (especially CJK) can
        # collapse punctuation/newlines and hurt sentence-level matching quality.
        ref_files = txt_files if txt_files else pdf_files
        if not ref_files:
            LOG.warning("No reference files found for %s in %s", lc, lang_dir)
            continue

        for fp in ref_files:
            text = _read_reference_text(fp)
            if not text:
                continue

            sentences = split_sentences(text, lang=lc)
            if not sentences:
                continue

            # Limit sentences per doc for performance
            sentences = sentences[:MAX_SENTENCES_PER_DOC]

            LOG.info("[%s] %s: %d sentences", lc, fp.name, len(sentences))

            # Compute multilingual embeddings (fast — no network needed)
            emb = encode(sentences)

            docs.append(_RefDoc(
                filename=fp.stem,
                lang_code=lc,
                lang_name=lang_info["name"],
                sentences=sentences,
                embeddings=emb,
            ))

    return docs


@lru_cache(maxsize=8)
def _load_reference_docs_cached(lang_codes_key: Tuple[str, ...]) -> Tuple[_RefDoc, ...]:
    """Cached wrapper to avoid rebuilding reference embeddings every request."""
    docs = _load_reference_docs(list(lang_codes_key))
    return tuple(docs)


def _get_reference_docs(lang_codes: Optional[List[str]] = None) -> List[_RefDoc]:
    """Fetch reference documents from cache using normalized language key."""
    if lang_codes:
        key = tuple(sorted(set(lang_codes)))
    else:
        key = tuple(sorted(LANGUAGES.keys()))
    return list(_load_reference_docs_cached(key))


def warmup_reference_cache(lang_codes: Optional[List[str]] = None) -> int:
    """Preload and cache reference documents + embeddings at startup."""
    docs = _get_reference_docs(lang_codes)
    return len(docs)


# ── Main detection function ──────────────────────────────────────────────────

def detect(text: str,
           lang_codes: Optional[List[str]] = None,
           threshold: Optional[float] = None) -> PlagiarismReport:
    """
    Run full cross-lingual plagiarism detection on English input text.

    Pipeline:
      1. Split input into sentences
      2. Embed all input sentences (multilingual model)
      3. For each reference doc: compute semantic similarity (cross-lingual, no translation)
      4. Find candidate matches above a low semantic pre-filter threshold
      5. Translate ONLY the matched reference sentences → English  (lazy)
      6. Recompute TF-IDF similarity on translated candidates
      7. Combine: final = 0.7×semantic + 0.3×tfidf
      8. Flag sentences above final threshold

    Args:
        text: English text to check for plagiarism.
        lang_codes: Languages to check against (default: all 5).
        threshold: Override combined similarity threshold.

    Returns:
        PlagiarismReport with per-sentence, per-document, per-language results.
    """
    threshold = threshold if threshold is not None else COMBINED_THRESHOLD

    # 1. Split input into sentences
    input_sentences = split_sentences(text, lang="en")
    report = PlagiarismReport(
        input_sentences=input_sentences,
        total_input_sentences=len(input_sentences),
        threshold_used=threshold,
    )

    if not input_sentences:
        LOG.warning("No sentences extracted from input text.")
        return report

    effective_threshold = threshold
    if len(input_sentences) <= 2:
        # Single/very short inputs are more sensitive to wording variation.
        effective_threshold = max(threshold - 0.06, 0.50)

    # Pre-filter with slightly lower semantic threshold to capture candidates.
    semantic_prefilter = max(effective_threshold - 0.12, 0.45)
    # Fallback gate for externally translated/paraphrased inputs where lexical
    # overlap can be weak but multilingual semantic similarity is still strong.
    high_semantic_gate = max(threshold + 0.20, 0.78)

    LOG.info("Input: %d sentences", len(input_sentences))

    # 2. Embed input sentences
    query_emb = encode(input_sentences)

    # 3. Load reference documents (embed only, no translation)
    ref_docs = _get_reference_docs(lang_codes)
    if not ref_docs:
        LOG.warning("No reference documents loaded. Run 'python -m plagx download' first.")
        return report

    LOG.info("Loaded %d reference docs across %d languages",
             len(ref_docs), len({d.lang_code for d in ref_docs}))

    # 4-7. Compare against each reference document
    for doc in ref_docs:
        # 4a. Semantic similarity matrix (cross-lingual, no translation needed)
        sem_sim = semantic_similarity_matrix(query_emb, doc.embeddings)

        # 4b. Find semantic candidates (pre-filter)
        candidates: List[Tuple[int, int, float]] = find_best_matches(
            sem_sim, input_sentences, doc.sentences, semantic_prefilter)

        if not candidates:
            # No candidate matches → skip translation entirely
            if doc.lang_code not in report.languages:
                report.languages[doc.lang_code] = LanguageResult(
                    lang_code=doc.lang_code, lang_name=doc.lang_name)
            report.languages[doc.lang_code].documents.append(DocumentResult(
                filename=doc.filename,
                language=doc.lang_code,
                language_name=doc.lang_name,
                num_ref_sentences=len(doc.sentences),
            ))
            continue

        # 4c. Collect unique candidate reference indices for translation
        ref_indices = sorted({ri for _, ri, _ in candidates})
        candidate_ref_sentences = [doc.sentences[i] for i in ref_indices]

        # 4d. Translate ONLY the matched reference sentences → English (lazy)
        LOG.info("[%s] %s: translating %d candidate sentences → EN",
                 doc.lang_code, doc.filename, len(candidate_ref_sentences))
        candidate_translations = translate_to_english(candidate_ref_sentences, doc.lang_code)

        # Build index map: original ref_idx -> position in candidate list
        ref_idx_to_pos = {ri: pos for pos, ri in enumerate(ref_indices)}

        # 4e. For each candidate, compute TF-IDF on translated text and combine
        doc_result = DocumentResult(
            filename=doc.filename,
            language=doc.lang_code,
            language_name=doc.lang_name,
            num_ref_sentences=len(doc.sentences),
        )

        for q_idx, r_idx, sem_score in candidates:
            pos = ref_idx_to_pos[r_idx]
            ref_en = candidate_translations[pos]
            ref_orig = doc.sentences[r_idx]
            q_sent = input_sentences[q_idx]

            # TF-IDF on this pair (single pair comparison)
            tfidf_pair = tfidf_similarity_matrix([q_sent], [ref_en])
            tfidf_score = float(tfidf_pair[0, 0])

            # Combined weighted score
            combined = W_SEMANTIC * sem_score + W_TFIDF * tfidf_score

            if combined >= effective_threshold or sem_score >= high_semantic_gate:
                doc_result.matches.append(SentenceMatch(
                    query_sentence=q_sent,
                    ref_sentence_original=ref_orig,
                    ref_sentence_english=ref_en,
                    language=doc.lang_code,
                    language_name=doc.lang_name,
                    document=doc.filename,
                    semantic_score=sem_score,
                    tfidf_score=tfidf_score,
                    combined_score=round(combined, 4),
                ))

        # Add to language result
        if doc.lang_code not in report.languages:
            report.languages[doc.lang_code] = LanguageResult(
                lang_code=doc.lang_code, lang_name=doc.lang_name)
        report.languages[doc.lang_code].documents.append(doc_result)

    return report


def detect_from_pdf(pdf_path: str,
                    lang_codes: Optional[List[str]] = None,
                    threshold: Optional[float] = None) -> PlagiarismReport:
    """
    Run plagiarism detection on a PDF file.
    Extracts text from the PDF then runs detect().
    """
    from PyPDF2 import PdfReader

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(path))
    text_parts = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text_parts.append(t)

    full_text = " ".join(text_parts).strip()
    if not full_text:
        raise ValueError("No text could be extracted from the PDF.")

    return detect(full_text, lang_codes=lang_codes, threshold=threshold)
