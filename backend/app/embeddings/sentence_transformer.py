from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_embedding_instance = None


def load_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_instance
    if _embedding_instance is not None:
        return _embedding_instance

    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
    _embedding_instance = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("Embedding model loaded successfully")
    return _embedding_instance
