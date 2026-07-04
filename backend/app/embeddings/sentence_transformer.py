from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_embedding_instance = None


def load_embedding_model():
    global _embedding_instance
    if _embedding_instance is not None:
        print("[stage01 | embeddings | 005-A] OK: Embedding model already loaded")
        return _embedding_instance

    try:
        print(f"[stage01 | embeddings | 005-A] OK: Loading model - {settings.EMBEDDING_MODEL_NAME}")
        _embedding_instance = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("[stage01 | embeddings | 005-B] OK: Embedding model loaded")
        return _embedding_instance
    except Exception as e:
        print(f"[stage01 | embeddings | 005-B] FAIL: Embedding model load failed - {e}")
        raise
