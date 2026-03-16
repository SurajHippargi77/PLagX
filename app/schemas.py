"""Pydantic schemas for API request/response and internal data."""

from typing import Optional
from pydantic import BaseModel, Field


# --- Plagiarism API ---


class PlagiarismAnalysisRequest(BaseModel):
    """Request body for plagiarism analysis (text only; PDF handled via multipart)."""

    text: Optional[str] = Field(None, description="Raw English text to analyze")


class MatchedSentence(BaseModel):
    """A single sentence match between source and reference."""

    source_sentence: str = Field(..., description="Sentence from submitted text")
    reference_sentence: str = Field(..., description="Matching sentence from reference document")
    translated_reference_sentence: str = Field("", description="English translation of the matched reference sentence")
    similarity: float = Field(..., ge=0, le=1, description="Combined similarity score")
    reference_document: str = Field(..., description="Filename of the reference document containing the match")
    language_code: str = Field(..., description="Reference language code")
    language_name: str = Field(..., description="Reference language name")


class PerDocumentSimilarity(BaseModel):
    """Per-document plagiarism similarity."""

    document: str = Field(..., description="Reference document filename")
    language_code: str = Field(..., description="Reference language code")
    language_name: str = Field(..., description="Reference language name")
    plagiarism_percentage: float = Field(..., ge=0, le=100, description="% of input sentences matching this document")
    match_count: int = Field(..., ge=0)
    average_similarity: float = Field(..., ge=0, le=100)


class PlagiarismAnalysisResponse(BaseModel):
    """Full plagiarism analysis result."""

    overall_plagiarism_percentage: float = Field(..., ge=0, le=100)
    cross_language_similarity_percentage: float = Field(..., ge=0, le=100)
    dutch_similarity_percentage: float = Field(..., ge=0, le=100)
    per_language_similarity: dict[str, float] = Field(default_factory=dict)
    per_document_similarity: list[PerDocumentSimilarity] = Field(
        default_factory=list,
        description="Per-document plagiarism breakdown",
    )
    matched_sentences: list[MatchedSentence] = Field(default_factory=list)
    total_sentences_analyzed: int = Field(..., ge=0)
    analysis_id: Optional[str] = None


# --- Reference documents API (read-only) ---


class ReferenceDocumentInfo(BaseModel):
    """Summary of a static reference document (read-only)."""

    filename: str
    sentence_count: int
