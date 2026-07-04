from langchain_chroma import Chroma
from app.core.logging import get_logger
from app.database.database import get_parent_document

logger = get_logger(__name__)


def search_chunks(vectorstore: Chroma, query: str, k: int = 5, filter_dict: dict | None = None) -> list:
    try:
        kwargs = {"k": k}
        if filter_dict:
            kwargs["filter"] = filter_dict
        results = vectorstore.similarity_search_with_score(query, **kwargs)

        swapped_results = []
        for doc, score in results:
            if doc.metadata.get("strategy") == "parent-child" and doc.metadata.get("parent_id"):
                parent_content = get_parent_document(doc.metadata["parent_id"])
                if parent_content:
                    doc.page_content = parent_content
                    doc.metadata["swapped"] = True
            swapped_results.append((doc, score))

        logger.info(f"Found {len(swapped_results)} chunks")
        return swapped_results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise
