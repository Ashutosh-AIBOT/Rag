import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def initialize_chroma(embeddings) -> Chroma:
    try:
        print(f"[stage01 | chroma | 006-A] OK: Connecting to ChromaDB at {settings.CHROMA_PERSIST_DIRECTORY}")
        chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIRECTORY,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        print("[stage01 | chroma | 006-B] OK: ChromaDB connected")
    except Exception as e:
        print(f"[stage01 | chroma | 006-B] FAIL: ChromaDB connection failed - {e}")
        raise

    try:
        vectorstore = Chroma(
            client=chroma_client,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
        )
        print(f"[stage01 | chroma | 006-C] OK: Collection '{settings.CHROMA_COLLECTION_NAME}' ready")
        return vectorstore
    except Exception as e:
        print(f"[stage01 | chroma | 006-C] FAIL: Collection creation failed - {e}")
        raise


def add_documents_to_chroma(vectorstore: Chroma, texts: list[str], metadatas: list[dict], ids: list[str]) -> None:
    try:
        vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        print(f"[stage01 | chroma | 006-D] OK: Added {len(texts)} chunks to ChromaDB")
    except Exception as e:
        print(f"[stage01 | chroma | 006-D] FAIL: Add documents failed - {e}")
        raise


def search_chroma(vectorstore: Chroma, query: str, k: int = 5, filter_dict: dict | None = None) -> list:
    try:
        kwargs = {"k": k}
        if filter_dict:
            kwargs["filter"] = filter_dict
        results = vectorstore.similarity_search_with_score(query, **kwargs)
        print(f"[stage01 | chroma | 006-E] OK: Search returned {len(results)} results")
        return results
    except Exception as e:
        print(f"[stage01 | chroma | 006-E] FAIL: Search failed - {e}")
        raise


def delete_from_chroma(vectorstore: Chroma, filter_dict: dict) -> None:
    try:
        vectorstore._collection.delete(where=filter_dict)
        print(f"[stage01 | chroma | 006-F] OK: Deleted chunks with filter {filter_dict}")
    except Exception as e:
        print(f"[stage01 | chroma | 006-F] FAIL: Delete failed - {e}")
        raise


def get_vectorstore():
    from app.core.lifespan import app
    return app.state.vectorstore
