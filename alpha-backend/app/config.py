"""
Application Configuration

Responsibilities
----------------
- Load environment variables from .env
- Validate application configuration
- Provide a single Settings object for the entire application
- Keep ALL configurable values in one place

DO NOT:
- Initialize models
- Create database connections
- Configure logging
- Create FastAPI objects
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "data"

DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SETTINGS
# =============================================================================


class Settings(BaseSettings):
    """
    Global application settings.

    Automatically loads values from `.env`.

    Access anywhere using:

        from app.config import settings
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # APPLICATION
    # =========================================================================

    APP_NAME: str = "Advanced RAG Platform"

    DATA_DIR: str = str(DEFAULT_DATA_DIR)

    APP_VERSION: str = "1.0.0"

    ENVIRONMENT: Literal[
        "development",
        "testing",
        "production",
    ] = "development"

    DEBUG: bool = True

    # =========================================================================
    # LOGGING
    # =========================================================================

    LOG_LEVEL: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"

    LOG_FORMAT: str = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    # =========================================================================
    # API KEYS
    # =========================================================================

    GOOGLE_API_KEY: str = ""

    GROQ_API_KEY: str = ""

    NVIDIA_API_KEY: str = ""

    COHERE_API_KEY: str = ""

    # =========================================================================
    # DEFAULT LLM
    # =========================================================================

    DEFAULT_LLM_PROVIDER: Literal[
        "gemini",
        "groq",
        "nvidia",
    ] = "gemini"

    GEMINI_MODEL: str = "gemini-2.5-flash"

    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    NVIDIA_MODEL: str = "meta/llama-3.3-70b-instruct"

    LLM_TEMPERATURE: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )

    LLM_TIMEOUT: int = 60

    MAX_LLM_RETRIES: int = 2

    MAX_CONCURRENT_LLM_REQUESTS: int = 3

    # =========================================================================
    # EMBEDDINGS
    # =========================================================================

    EMBEDDING_MODEL: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # =========================================================================
    # RERANKER
    # =========================================================================

    CROSS_ENCODER_MODEL: str = (
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # =========================================================================
    # VECTOR DATABASE
    # =========================================================================

    CHROMA_COLLECTION_NAME: str = "rag_chunks"

    CHROMA_DB_PATH: str = str(DEFAULT_DATA_DIR / "chroma")

    # =========================================================================
    # SQLITE
    # =========================================================================

    SQLITE_DB_PATH: str = str(DEFAULT_DATA_DIR / "rag.db")

    # =========================================================================
    # CHUNKING
    # =========================================================================

    CHUNK_SIZE: int = 1000

    CHUNK_OVERLAP: int = 200

    # =========================================================================
    # RETRIEVAL
    # =========================================================================

    RETRIEVAL_TOP_K: int = 20

    FINAL_TOP_K: int = 5

    BM25_WEIGHT: float = 0.5

    VECTOR_WEIGHT: float = 0.5

    # =========================================================================
    # FILE UPLOAD
    # =========================================================================

    MAX_UPLOAD_SIZE_MB: int = 50

    ALLOWED_EXTENSIONS: tuple[str, ...] = (
        ".pdf",
        ".txt",
        ".docx",
        ".md",
    )

    # =========================================================================
    # EVALUATION
    # =========================================================================

    EVALUATION_BATCH_SIZE: int = 5

    # =========================================================================
    # API
    # =========================================================================

    API_PREFIX: str = "/api"

    # =========================================================================
    # CORS
    # =========================================================================

    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


# =============================================================================
# SETTINGS INSTANCE
# =============================================================================


@lru_cache
def get_settings() -> Settings:
    """
    Returns a singleton Settings instance.

    Using lru_cache ensures the settings are created only once
    during the application's lifetime.
    """
    s = Settings()
    import os
    if s.GOOGLE_API_KEY:
        os.environ["GOOGLE_API_KEY"] = s.GOOGLE_API_KEY
    if s.GROQ_API_KEY:
        os.environ["GROQ_API_KEY"] = s.GROQ_API_KEY
    if s.NVIDIA_API_KEY:
        os.environ["NVIDIA_API_KEY"] = s.NVIDIA_API_KEY
    if s.COHERE_API_KEY:
        os.environ["COHERE_API_KEY"] = s.COHERE_API_KEY
    return s


settings = get_settings()