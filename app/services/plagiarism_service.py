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

    per_language_pct = report.per_language_pct()
    per_language_named: Dict[str, float] = {
        LANGUAGES.get(code, {}).get("name", code): round(pct, 2)
        for code, pct in per_language_pct.items()
    }

    matched_sentences: List[MatchedSentence] = []
    for m in all_matches:
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

    per_document: List[PerDocumentSimilarity] = []
    for lang_code, lr in report.languages.items():
        for d in lr.documents:
            if d.match_count == 0:
                continue
            unique_sources = len({m.query_sentence for m in d.matches})
            pct = (unique_sources / report.total_input_sentences * 100) if report.total_input_sentences else 0.0
            avg_sim = (sum(m.combined_score for m in d.matches) / d.match_count * 100) if d.match_count else 0.0
            per_document.append(
                PerDocumentSimilarity(
                    document=d.filename,
                    language_code=lang_code,
                    language_name=d.language_name,
                    plagiarism_percentage=round(pct, 2),
                    match_count=d.match_count,
                    average_similarity=round(avg_sim, 2),
                )
            )
    per_document.sort(key=lambda x: (-x.plagiarism_percentage, -x.average_similarity))

    # For each unique input sentence, take only its BEST match score across all
    # reference documents.  Averaging every duplicate match dilutes the result
    # because a sentence that matches 10 docs also accumulates many weak
    # (barely-above-threshold) duplicates that drag the mean down.
    best_per_query: Dict[str, float] = {}
    for m in matched_sentences:
        key = m.source_sentence
        if key not in best_per_query or m.similarity > best_per_query[key]:
            best_per_query[key] = m.similarity

    cross_avg = (
        sum(best_per_query.values()) / len(best_per_query) * 100
        if best_per_query
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
