"""Plagiarism analysis API routes. Error handling: empty file, unsupported format, embedding errors."""

from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from app.schemas import PlagiarismAnalysisResponse
from app.services import plagiarism_service

router = APIRouter(prefix="/api/plagiarism", tags=["plagiarism"])

ALLOWED_EXTENSIONS = (".pdf",)
MAX_FILENAME_LEN = 255


@router.post("/analyze", response_model=PlagiarismAnalysisResponse)
async def analyze_plagiarism(
    text: str | None = Form(None, description="English text to analyze"),
    file: UploadFile | None = File(None, description="PDF file (English) to analyze"),
):
    """
    Analyze submitted text or PDF for cross-language plagiarism against stored Dutch papers.
    Provide either `text` or `file` (PDF). If both are provided, file takes precedence.
    """
    # File path: validate format and empty
    if file and file.filename:
        if len(file.filename) > MAX_FILENAME_LEN:
            raise HTTPException(status_code=400, detail="Filename too long.")
        if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are accepted. Unsupported file format.",
            )
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")
        try:
            return plagiarism_service.analyze_pdf_bytes(content)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            if "Embedding" in str(e):
                raise HTTPException(status_code=503, detail=str(e)) from e
            raise HTTPException(status_code=500, detail=str(e)) from e

    if text and text.strip():
        try:
            return plagiarism_service.analyze_text(text.strip())
        except RuntimeError as e:
            if "Embedding" in str(e):
                raise HTTPException(status_code=503, detail=str(e)) from e
            raise HTTPException(status_code=500, detail=str(e)) from e

    raise HTTPException(
        status_code=400,
        detail="Provide either 'text' (plain English text) or 'file' (PDF file).",
    )
