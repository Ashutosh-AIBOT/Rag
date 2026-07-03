from app.core.logging import get_logger
from app.vectorstore.chroma import search_chroma
from app.database.database import get_parent_document

logger = get_logger(__name__)


def retrieve_chunks(chroma_store, query: str, k: int = 5, strategy: str | None = None) -> list[dict]:
    filter_dict = None
    if strategy:
        filter_dict = {"strategy": strategy}

    results = search_chroma(chroma_store, query, k=k, filter_dict=filter_dict)

    chunks = []
    for doc, score in results:
        chunk_data = {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score),
        }

        if doc.metadata.get("strategy") == "parent-child":
            parent_id = doc.metadata.get("parent_id")
            if parent_id:
                parent_content = get_parent_document(parent_id)
                if parent_content:
                    chunk_data["content"] = parent_content
                    chunk_data["swapped_to_parent"] = True
                    logger.info(f"Swapped child chunk to parent: {parent_id}")

        chunks.append(chunk_data)

    logger.info(f"Retrieved {len(chunks)} chunks for query")
    return chunks
