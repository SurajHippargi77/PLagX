"""
Reference documents API (read-only). Reference documents are static files on disk.
No upload or modification via API.
"""

from fastapi import APIRouter

from app.schemas import ReferenceDocumentInfo
from app.reference_store import list_documents

router = APIRouter(prefix="/api/reference-documents", tags=["reference-documents"])


@router.get("/", response_model=list[ReferenceDocumentInfo])
async def list_reference_documents():
    """List all 10 static reference documents (read-only). Reference documents cannot be modified via API."""
    docs = list_documents()
    return [ReferenceDocumentInfo(filename=d["filename"], sentence_count=d["sentence_count"]) for d in docs]
