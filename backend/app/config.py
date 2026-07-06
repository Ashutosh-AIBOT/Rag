from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Settings(BaseSettings):
    GOOGLE_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    NVIDIA_API_KEY: str = ""
    EMBEDDING_MODEL_NAME: str = ""
    CHROMA_PERSIST_DIRECTORY: str = ""
    CHROMA_COLLECTION_NAME: str = ""
    DATABASE_URL: str = ""
    REDIS_URL: str = "redis://localhost:6379/0"

    BM25_VECTOR_WEIGHT: float = 0.5
    BM25_SPARSE_WEIGHT: float = 0.5
    RRF_K: int = 60

    model_config = {"env_file": str(env_path), "extra": "ignore"}


settings = Settings()
