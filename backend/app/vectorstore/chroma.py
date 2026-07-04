import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def initialize_chroma(embeddings) -> Chroma:
    try:
        logger.info(f"Connecting to ChromaDB at {settings.CHROMA_PERSIST_DIRECTORY}")
        chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIRECTORY,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB connected")
    except Exception as e:
        logger.error(f"ChromaDB connection failed: {e}")
        raise

    try:
        vectorstore = Chroma(
            client=chroma_client,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
        )
        logger.info(f"Collection '{settings.CHROMA_COLLECTION_NAME}' ready")
        return vectorstore
    except Exception as e:
        logger.error(f"Collection creation failed: {e}")
        raise


def add_documents_to_chroma(vectorstore: Chroma, texts: list[str], metadatas: list[dict], ids: list[str]) -> None:
    try:
        vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        logger.info(f"Added {len(texts)} chunks to ChromaDB")
    except Exception as e:
        logger.error(f"Add documents failed: {e}")
        raise


def search_chroma(vectorstore: Chroma, query: str, k: int = 5, filter_dict: dict | None = None) -> list:
    try:
        kwargs = {"k": k}
        if filter_dict:
            kwargs["filter"] = filter_dict
        results = vectorstore.similarity_search_with_score(query, **kwargs)
        logger.info(f"Search returned {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise


def delete_from_chroma(vectorstore: Chroma, filter_dict: dict) -> None:
    try:
        vectorstore._collection.delete(where=filter_dict)
        logger.info(f"Deleted chunks with filter {filter_dict}")
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise


def initialize_chroma_gemini() -> Chroma:
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIRECTORY,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        gemini_embed = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = Chroma(
            client=chroma_client,
            collection_name=settings.CHROMA_COLLECTION_NAME + "_gemini",
            embedding_function=gemini_embed,
        )
        logger.info("Gemini Chroma collection ready")
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to initialize Gemini Chroma: {e}")
        return None
