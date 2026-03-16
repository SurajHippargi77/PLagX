"""Application configuration."""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    app_name: str = "PLagX"
    app_version: str = "1.0.0"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    # Similarity threshold (0–1). Matches above this are considered plagiarism.
    # Env: PLAGX_SIMILARITY_THRESHOLD
    similarity_threshold: float = 0.58  # aligned with CLI defaults
    max_file_size_mb: int = 20
    # Optional: override directory of Dutch .txt reference documents (default: app/data/dutch_papers/)
    dutch_papers_dir: str | None = None

    class Config:
        env_prefix = "PLAGX_"
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
