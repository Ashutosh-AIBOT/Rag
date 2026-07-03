import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    GOOGLE_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    NVIDIA_API_KEY: str = ""

    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    CHROMA_PERSIST_DIRECTORY: str = str(BASE_DIR / "chroma_db")
    CHROMA_COLLECTION_NAME: str = "rag_documents"

    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR / 'rag_database.db'}"

    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 4096
    LLM_CONCURRENCY_LIMIT: int = 5

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
