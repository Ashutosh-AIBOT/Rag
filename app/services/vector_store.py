"""
Thin wrapper around a single persistent Chroma collection that stores
chunks from ALL chunking strategies together, distinguished by the
`strategy` metadata field. This is what enables A/B comparison across
strategies without re-indexing.
"""
import logging
import pickle
import types
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document as LCDocument
from langchain_chroma import Chroma

from app.config import get_settings
from app.services.llm_factory import get_embeddings

logger = logging.getLogger("vector_store")
settings = get_settings()
COLLECTION_NAME = "rag_chunks"

TAG_PREFIX = "tag:"  # per-tag boolean metadata field, e.g. "tag:finance" -> True


def auto_migrate_chroma(persist_dir: str):
    """
    Self-healing migration that checks for any legacy index_metadata.pickle files
    written as a standard dict by older Chroma versions, and upgrades them to SimpleNamespace
    so that newer Chroma versions don't crash with AttributeError: 'dict' object has no attribute 'dimensionality'.
    """
    try:
        persist_path = Path(persist_dir)
        if not persist_path.exists():
            return
        for pickle_path in persist_path.rglob("index_metadata.pickle"):
            try:
                with open(pickle_path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    logger.info("auto_migrate_chroma: Found legacy dict metadata in %s. Migrating to SimpleNamespace...", pickle_path)
                    new_data = types.SimpleNamespace(**data)
                    with open(pickle_path, "wb") as f:
                        pickle.dump(new_data, f)
                    logger.info("auto_migrate_chroma: Migration successful for %s", pickle_path)
            except Exception as e:
                logger.error("auto_migrate_chroma: Failed to migrate %s: %s", pickle_path, e)
    except Exception as e:
        logger.error("auto_migrate_chroma: General failure: %s", e)


_vector_store_instance = None

def get_vector_store() -> Chroma:
    global _vector_store_instance
    if _vector_store_instance is not None:
        return _vector_store_instance
    
    logger.info("get_vector_store: Initializing/fetching Chroma instance. Collection=%s, Dir=%s", COLLECTION_NAME, settings.chroma_persist_dir)
    
    # Ensure directory exists
    persist_path = Path(settings.chroma_persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    
    auto_migrate_chroma(settings.chroma_persist_dir)
    
    _vector_store_instance = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )
    logger.info("get_vector_store: Chroma initialized successfully")
    return _vector_store_instance


def reset_vector_store():
    """Reset the vector store instance (useful after document deletion)."""
    global _vector_store_instance
    _vector_store_instance = None
    logger.info("reset_vector_store: Vector store instance reset")


def add_chunks(chunks: List[LCDocument]) -> None:
    if not chunks:
        logger.info("add_chunks: No chunks provided to index.")
        return
    uid = chunks[0].metadata.get("user_id", "-") if chunks else "-"
    logger.info("[uid:%s] add_chunks: Formatting metadata for %d chunks", uid, len(chunks))
    store = get_vector_store()
    # Chroma metadata must be flat scalars -> stringify tags/list metadata,
    # but expand tags into individual boolean flags FIRST so multi-tag
    # filtering (AND/OR across tags) is actually possible (see
    # build_chroma_where below). The comma-joined "tags" string is kept
    # too, purely for display in the UI.
    for c in chunks:
        tags = c.metadata.get("tags")
        if isinstance(tags, list):
            for t in tags:
                c.metadata[f"{TAG_PREFIX}{t}"] = True
        c.metadata = {
            k: (", ".join(v) if isinstance(v, list) else v)
            for k, v in c.metadata.items()
            if v is not None
        }
    ids = [c.metadata.get("chunk_id") for c in chunks]
    store.add_documents(chunks, ids=ids)
    logger.info("[uid:%s] add_chunks: Wrote %d documents to Chroma", uid, len(chunks))


def delete_document_chunks(doc_id: str, user_id: Optional[str] = None) -> None:
    logger.info("[uid:%s] delete_document_chunks: Deleting all chunks for doc_id=%s", user_id or "-", doc_id)
    store = get_vector_store()
    store._collection.delete(where={"doc_id": doc_id})
    logger.info("[uid:%s] delete_document_chunks: Deleted chunks for doc_id=%s", user_id or "-", doc_id)


def _date_to_epoch(date_str: str, end_of_day: bool = False) -> Optional[float]:
    import datetime as dt
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            d = dt.datetime.strptime(date_str, fmt)
            if end_of_day:
                d = d + dt.timedelta(hours=23, minutes=59, seconds=59)
            return d.replace(tzinfo=dt.timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def build_chroma_where(filters, strategy: Optional[str] = None, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Translate our MetadataFilters schema into a Chroma `where` clause.

    User-facing filters (source, section, tags, page range, date range)
    combine using `filters.filter_logic` ("and" | "or"), so
    "source=X OR section=Y" and "source=X AND section=Y" are both
    expressible -- previously only AND was possible.

    The `strategy` tag (which strategy's chunks to search) is a separate,
    always-AND concern: it scopes *which chunk set* we're searching within,
    it isn't part of the user's AND/OR filter combination.

    The `user_id` is always ANDed in to enforce per-user data isolation.
    """
    user_conditions: List[Dict[str, Any]] = []

    if user_id:
        user_conditions.append({"user_id": user_id})

    if filters:
        def get_val(name: str) -> Any:
            if hasattr(filters, name):
                return getattr(filters, name)
            if isinstance(filters, dict):
                return filters.get(name)
            return None

        source = get_val("source")
        section = get_val("section")
        tags = get_val("tags")
        page_min = get_val("page_min")
        page_max = get_val("page_max")
        date_from = get_val("date_from")
        date_to = get_val("date_to")
        filter_logic = get_val("filter_logic") or "and"

        if source:
            if isinstance(source, list):
                if len(source) == 1:
                    user_conditions.append({"source": source[0]})
                elif len(source) > 1:
                    user_conditions.append({"source": {"$in": source}})
            else:
                user_conditions.append({"source": source})
        if section:
            user_conditions.append({"section": section})
        if tags:
            # Each tag is its own boolean metadata field (see add_chunks),
            # so a chunk can match >1 tag simultaneously -- fixes both the
            # "tags never match" bug and the "2+ tags = impossible" bug.
            tag_conditions = [{f"{TAG_PREFIX}{tag}": True} for tag in tags]
            if len(tag_conditions) == 1:
                user_conditions.append(tag_conditions[0])
            else:
                op = "$or" if filter_logic == "or" else "$and"
                user_conditions.append({op: tag_conditions})
        if page_min is not None:
            user_conditions.append({"page": {"$gte": page_min}})
        if page_max is not None:
            user_conditions.append({"page": {"$lte": page_max}})
        if date_from:
            epoch = _date_to_epoch(date_from)
            if epoch is not None:
                user_conditions.append({"upload_ts": {"$gte": epoch}})
        if date_to:
            epoch = _date_to_epoch(date_to, end_of_day=True)
            if epoch is not None:
                user_conditions.append({"upload_ts": {"$lte": epoch}})

    # Separate user_id (always AND) from user-facing filters (AND/OR per logic)
    always_and: List[Dict[str, Any]] = [c for c in user_conditions if "user_id" in c]
    user_facing = [c for c in user_conditions if "user_id" not in c]

    combined_user_facing = None
    if len(user_facing) == 1:
        combined_user_facing = user_facing[0]
    elif len(user_facing) > 1:
        op = "$or" if filter_logic == "or" else "$and"
        combined_user_facing = {op: user_facing}

    strategy_condition = {"strategy": strategy} if strategy else None

    # Build final $and list: user_id + user-facing filters + strategy
    and_parts = always_and[:]
    if combined_user_facing:
        and_parts.append(combined_user_facing)
    if strategy_condition:
        and_parts.append(strategy_condition)

    if len(and_parts) == 0:
        return None
    if len(and_parts) == 1:
        return and_parts[0]
    return {"$and": and_parts}


def get_chunks_for_document(doc_id: str, limit: int = 2000) -> Dict[str, List[Dict[str, Any]]]:
    """Returns every chunk for a document, grouped by chunking strategy,
    with text previews -- powers the Document Management 'click to see all
    chunks' screen."""
    store = get_vector_store()
    result = store._collection.get(
        where={"doc_id": doc_id}, limit=limit, include=["documents", "metadatas"]
    )
    by_strategy: Dict[str, List[Dict[str, Any]]] = {}
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    for text, meta in zip(documents, metadatas):
        strategy = meta.get("strategy", "unknown")
        by_strategy.setdefault(strategy, []).append({
            "chunk_id": meta.get("chunk_id", ""),
            "text": text,
            "page": meta.get("page"),
            "section": meta.get("section"),
            "strategy": strategy,
        })
    # Stable ordering within each strategy (by page then chunk_id) so the
    # UI doesn't reshuffle chunks on every refresh.
    for strategy in by_strategy:
        by_strategy[strategy].sort(key=lambda c: (c.get("page") or 0, c["chunk_id"]))
    return by_strategy
