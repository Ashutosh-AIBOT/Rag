from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_embedding_instance = None


def load_embedding_model():
    global _embedding_instance
    if _embedding_instance is not None:
        logger.info("Embedding model already loaded")
        return _embedding_instance

    try:
        logger.info(f"Loading model: {settings.EMBEDDING_MODEL_NAME}")
        _embedding_instance = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded")
        return _embedding_instance
    except Exception as e:
        logger.error(f"Embedding model load failed: {e}")
        raise
