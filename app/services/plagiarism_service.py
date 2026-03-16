"""Frontend service wrapper around the multilingual PLagX detector."""

import uuid
from typing import Dict, List

from app.config import get_settings
from app.schemas import (
    MatchedSentence,
    PerDocumentSimilarity,
    PlagiarismAnalysisResponse,
)
from app.services import pdf_service
from plagx.config import LANGUAGES
from plagx.detector import detect


def _threshold() -> float:
    return get_settings().similarity_threshold


def _empty_response() -> PlagiarismAnalysisResponse:
    return PlagiarismAnalysisResponse(
        overall_plagiarism_percentage=0.0,
        cross_language_similarity_percentage=0.0,
        dutch_similarity_percentage=0.0,
        per_language_similarity={},
        per_document_similarity=[],
        matched_sentences=[],
        total_sentences_analyzed=0,
        analysis_id=str(uuid.uuid4()),
    )


def analyze_text(text: str) -> PlagiarismAnalysisResponse:
    """Analyze plain English text using the multilingual `plagx` detector."""
    text = (text or "").strip()
    if not text:
        return _empty_response()

    report = detect(text, threshold=_threshold())
    all_matches = report.all_matches

    # Prefer a dominant source language when a single language covers at least
    # half the input sentences. This stabilizes language attribution for
    # translated/copied passages that originate from one language corpus.
    lang_query_semantic: Dict[str, Dict[str, float]] = {}
    for m in all_matches:
        per_query = lang_query_semantic.setdefault(m.language, {})
        prev = per_query.get(m.query_sentence)
        if prev is None or m.semantic_score > prev:
            per_query[m.query_sentence] = m.semantic_score

    dominant_language = None
    dominant_candidates = []
    for lang, q_scores in lang_query_semantic.items():
        coverage = len(q_scores)
        if report.total_input_sentences and coverage / report.total_input_sentences >= 0.5:
            avg_sem = sum(q_scores.values()) / coverage if coverage else 0.0
            dominant_candidates.append((coverage, avg_sem, lang))

    # Apply dominant-language consolidation only for short inputs where users
    # typically paste one translated paragraph from a single source document.
    if dominant_candidates and report.total_input_sentences <= 3:
        dominant_candidates.sort(reverse=True)
        dominant_language = dominant_candidates[0][2]

    # Keep one strongest match per input sentence. If a dominant language is
    # detected, first try selecting from that language for each sentence.
    best_match_per_query: Dict[str, object] = {}
    matches_by_query: Dict[str, List[object]] = {}
    for m in all_matches:
        matches_by_query.setdefault(m.query_sentence, []).append(m)

    for q, q_matches in matches_by_query.items():
        candidates = q_matches
        if dominant_language is not None:
            dominant_only = [m for m in q_matches if m.language == dominant_language]
            if dominant_only:
                candidates = dominant_only

        best = max(candidates, key=lambda x: (x.combined_score, x.semantic_score))
        best_match_per_query[q] = best

    selected_matches = list(best_match_per_query.values())

    matched_sentences: List[MatchedSentence] = []
    for m in selected_matches:
        matched_sentences.append(
            MatchedSentence(
                source_sentence=m.query_sentence,
                reference_sentence=m.ref_sentence_original,
                translated_reference_sentence=m.ref_sentence_english,
                similarity=round(m.combined_score, 4),
                reference_document=m.document,
                language_code=m.language,
                language_name=m.language_name,
            )
        )

    total = report.total_input_sentences

    # Recompute per-language percentages from selected best matches only.
    lang_to_queries: Dict[str, set] = {code: set() for code in LANGUAGES.keys()}
    for m in selected_matches:
        lang_to_queries.setdefault(m.language, set()).add(m.query_sentence)

    per_language_named: Dict[str, float] = {
        LANGUAGES.get(code, {}).get("name", code): round((len(queries) / total * 100) if total else 0.0, 2)
        for code, queries in lang_to_queries.items()
    }

    # Recompute per-document percentages from selected best matches only.
    doc_groups: Dict[tuple, List[object]] = {}
    for m in selected_matches:
        key = (m.document, m.language, m.language_name)
        doc_groups.setdefault(key, []).append(m)

    per_document: List[PerDocumentSimilarity] = []
    for (document, lang_code, lang_name), matches in doc_groups.items():
        unique_sources = len({m.query_sentence for m in matches})
        pct = (unique_sources / total * 100) if total else 0.0
        avg_sim = (sum(m.combined_score for m in matches) / len(matches) * 100) if matches else 0.0
        per_document.append(
            PerDocumentSimilarity(
                document=document,
                language_code=lang_code,
                language_name=lang_name,
                plagiarism_percentage=round(pct, 2),
                match_count=len(matches),
                average_similarity=round(avg_sim, 2),
            )
        )
    per_document.sort(key=lambda x: (-x.plagiarism_percentage, -x.average_similarity))

    # Cross-language similarity should reflect multilingual semantic closeness,
    # not lexical overlap after translation. For each input sentence, use the
    # strongest semantic match across all reference documents.
    best_semantic_per_query: Dict[str, float] = {}
    for m in selected_matches:
        key = m.query_sentence
        if key not in best_semantic_per_query or m.semantic_score > best_semantic_per_query[key]:
            best_semantic_per_query[key] = m.semantic_score

    cross_avg = (
        sum(best_semantic_per_query.values()) / len(best_semantic_per_query) * 100
        if best_semantic_per_query
        else 0.0
    )

    return PlagiarismAnalysisResponse(
        overall_plagiarism_percentage=round(report.overall_plagiarism_pct, 2),
        cross_language_similarity_percentage=round(cross_avg, 2),
        # Backward compatibility field used by current frontend gauge card.
        dutch_similarity_percentage=round(cross_avg, 2),
        per_language_similarity=per_language_named,
        per_document_similarity=per_document,
        matched_sentences=matched_sentences,
        total_sentences_analyzed=report.total_input_sentences,
        analysis_id=str(uuid.uuid4()),
    )


def analyze_pdf_bytes(pdf_bytes: bytes) -> PlagiarismAnalysisResponse:
    """Extract text from PDF bytes and run multilingual analysis."""
    text = pdf_service.extract_text_from_bytes(pdf_bytes)
    return analyze_text(text)
