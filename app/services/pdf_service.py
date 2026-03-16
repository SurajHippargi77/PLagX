"""PDF text extraction using PyPDF2."""

from pathlib import Path
from io import BytesIO

from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

from app.text_utils import normalize_text


def extract_text_from_file(file_path: Path) -> str:
    """
    Extract all text from a PDF file.
    Returns normalized concatenated text from all pages.
    Raises ValueError on invalid/empty PDF.
    """
    try:
        reader = PdfReader(str(file_path))
        chunks = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                chunks.append(text)
        result = normalize_text(" ".join(chunks))
        if not result:
            raise ValueError("No text could be extracted from the PDF.")
        return result
    except (PdfReadError, OSError) as e:
        raise ValueError(f"Failed to read PDF: {e}") from e


def extract_text_from_bytes(content: bytes) -> str:
    """Extract text from PDF bytes (e.g. uploaded file). Raises ValueError on invalid/empty PDF."""
    if not content:
        raise ValueError("PDF content is empty.")
    try:
        reader = PdfReader(BytesIO(content))
        chunks = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                chunks.append(text)
        result = normalize_text(" ".join(chunks))
        if not result:
            raise ValueError("No text could be extracted from the PDF.")
        return result
    except (PdfReadError, OSError) as e:
        raise ValueError(f"Failed to read PDF: {e}") from e
