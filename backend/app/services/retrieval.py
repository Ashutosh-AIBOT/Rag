from langchain_chroma import Chroma
from app.core.logging import get_logger

logger = get_logger(__name__)


def search_chunks(vectorstore: Chroma, query: str, k: int = 5, filter_dict: dict | None = None) -> list:
    try:
        kwargs = {"k": k}
        if filter_dict:
            kwargs["filter"] = filter_dict
        results = vectorstore.similarity_search_with_score(query, **kwargs)
        logger.info(f"Found {len(results)} chunks")
        return results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise
