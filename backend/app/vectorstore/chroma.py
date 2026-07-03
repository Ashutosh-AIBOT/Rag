import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def init_chroma(embeddings) -> Chroma:
    logger.info(f"Initializing ChromaDB at: {settings.CHROMA_PERSIST_DIRECTORY}")

    chroma_client = chromadb.PersistentClient(
        path=settings.CHROMA_PERSIST_DIRECTORY,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    vectorstore = Chroma(
        client=chroma_client,
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
    )

    logger.info(f"ChromaDB collection '{settings.CHROMA_COLLECTION_NAME}' ready")
    return vectorstore


def add_documents_to_chroma(vectorstore: Chroma, texts: list[str], metadatas: list[dict], ids: list[str]) -> None:
    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    logger.info(f"Added {len(texts)} chunks to ChromaDB")


def search_chroma(vectorstore: Chroma, query: str, k: int = 5, filter_dict: dict | None = None) -> list:
    kwargs = {"k": k}
    if filter_dict:
        kwargs["filter"] = filter_dict
    results = vectorstore.similarity_search_with_score(query, **kwargs)
    logger.info(f"ChromaDB search returned {len(results)} results")
    return results


def delete_from_chroma(vectorstore: Chroma, filter_dict: dict) -> None:
    vectorstore._collection.delete(where=filter_dict)
    logger.info(f"Deleted documents from ChromaDB with filter: {filter_dict}")
