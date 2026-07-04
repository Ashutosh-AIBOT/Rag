from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    GOOGLE_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    NVIDIA_API_KEY: str = ""
    EMBEDDING_MODEL_NAME: str = ""
    CHROMA_PERSIST_DIRECTORY: str = ""
    CHROMA_COLLECTION_NAME: str = ""
    DATABASE_URL: str = ""

    BM25_VECTOR_WEIGHT: float = 0.5
    BM25_SPARSE_WEIGHT: float = 0.5
    RRF_K: int = 60

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
