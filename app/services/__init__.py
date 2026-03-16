# Services module
from app.services import pdf_service
from app.services import embedding_service
from app.services import similarity_service
from app.services import plagiarism_service

__all__ = [
    "pdf_service",
    "embedding_service",
    "similarity_service",
    "plagiarism_service",
]
