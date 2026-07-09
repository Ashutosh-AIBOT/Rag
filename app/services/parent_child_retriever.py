"""
Wires a real LangChain `ParentDocumentRetriever` for the parent-child
chunking strategy, replacing what used to be a fully hand-rolled
implementation (children denormalized a copy of the parent's text directly
onto their own metadata, with no LangChain retriever object involved at
all).

Design:
  - Children still get embedded into the SAME shared Chroma collection as
    every other strategy (tagged strategy="parent_child"), so cross-strategy
    A/B comparison keeps working without a second vector store.
  - Parents are persisted in a `LocalFileStore`-backed `docstore` (LangChain's
    key/value document store abstraction), so parent lookups go through a
    real LangChain storage component instead of a denormalized metadata
    field, and parents survive process restarts.
  - `retrieval.parent_child_search()` still runs a *scored* child search
    (via basic_vector_search) so the UI can show per-chunk semantic scores
    and ranks -- `ParentDocumentRetriever.invoke()` alone doesn't expose
    those, so we use the retriever's `.docstore` directly for the
    child -> parent swap while keeping score transparency, and the fully
    "pure" `.invoke()` path is available via `get_parent_child_retriever()`
    for anyone who wants the plain LangChain interface.
"""
import logging
from functools import lru_cache
from typing import List, Dict

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

HAS_PARENT_CHILD = False
ParentDocumentRetriever = None
LocalFileStore = None
create_kv_docstore = None

try:
    from langchain.retrievers import ParentDocumentRetriever
    from langchain.storage import LocalFileStore, create_kv_docstore
    HAS_PARENT_CHILD = True
except ImportError:
    try:
        from langchain_community.retrievers import ParentDocumentRetriever
        from langchain_community.storage import LocalFileStore, create_kv_docstore
        HAS_PARENT_CHILD = True
    except (ImportError, AttributeError):
        logger.warning(
            "ParentDocumentRetriever not available. "
            "Install langchain-storage and langchain-retrievers for parent-child support."
        )

from app.config import get_settings
from app.services.vector_store import get_vector_store

settings = get_settings()

ID_KEY = "parent_id"  # matches the rest of the system's existing parent_id metadata field


@lru_cache
def _get_docstore():
    byte_store = LocalFileStore(f"{settings.chroma_persist_dir}/parent_docstore")
    return create_kv_docstore(byte_store)


@lru_cache
def get_parent_child_retriever():
    """The real, plain LangChain ParentDocumentRetriever. `.invoke(query)`
    returns full parent Documents directly (no scores) -- used when pure
    LangChain-native retrieval is wanted. See module docstring for why
    `retrieval.parent_child_search()` calls the vector store + docstore
    directly instead, to preserve score transparency for the UI."""
    if not HAS_PARENT_CHILD:
        raise RuntimeError("ParentDocumentRetriever not available")
    return ParentDocumentRetriever(
        vectorstore=get_vector_store(),
        docstore=_get_docstore(),
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=200 * 4, chunk_overlap=40),
        parent_splitter=RecursiveCharacterTextSplitter(chunk_size=1000 * 4, chunk_overlap=0),
        id_key=ID_KEY,
    )


def persist_parents(parent_lookup: Dict[str, LCDocument]) -> None:
    """Writes parent documents produced by chunking_strategies.parent_child_chunks
    into the retriever's real docstore, keyed by parent_id."""
    if not parent_lookup:
        return
    if not HAS_PARENT_CHILD:
        logger.warning("Cannot persist parents: ParentDocumentRetriever not available")
        return
    docstore = _get_docstore()
    docstore.mset(list(parent_lookup.items()))


def get_parent_text(parent_id: str) -> str:
    """Fetches a parent's full text from the persistent LangChain docstore
    given the parent_id stashed on a child chunk's metadata."""
    if not parent_id or not HAS_PARENT_CHILD:
        return ""
    docstore = _get_docstore()
    docs = docstore.mget([parent_id])
    if docs and docs[0] is not None:
        return docs[0].page_content
    return ""
