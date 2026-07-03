import sqlite3
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from app.config import settings
from app.core.logging import get_logger
from app.vectorstore.chroma import get_chroma_db

logger = get_logger(__name__)

def rebuild_bm25_index(app) -> Optional[BM25Retriever]:
    """
    Fetches all chunks from ChromaDB and builds a new BM25 index.
    Updates app.state.bm25_index.
    """
    logger.info("Rebuilding BM25 index from ChromaDB chunks...")
    try:
        db = get_chroma_db(app.state.embeddings)
        # Fetch all documents in ChromaDB
        collection_data = db._collection.get()
        documents_text = collection_data.get("documents", [])
        metadatas = collection_data.get("metadatas", [])
        
        if not documents_text:
            logger.warning("No documents found in ChromaDB to build BM25 index.")
            app.state.bm25_index = None
            return None

        docs = []
        for text, meta in zip(documents_text, metadatas):
            docs.append(Document(page_content=text, metadata=meta))

        bm25_retriever = BM25Retriever.from_documents(docs)
        # Set default retrieve top_k
        bm25_retriever.k = settings.RETRIEVAL_TOP_K
        app.state.bm25_index = bm25_retriever
        logger.info(f"BM25 index successfully rebuilt with {len(docs)} chunks.")
        return bm25_retriever
    except Exception as e:
        logger.error(f"Failed to rebuild BM25 index: {e}", exc_info=True)
        app.state.bm25_index = None
        return None

def get_bm25_retriever(app) -> Optional[BM25Retriever]:
    """Gets the preloaded BM25 retriever or rebuilds it if None."""
    if not hasattr(app.state, "bm25_index") or app.state.bm25_index is None:
        return rebuild_bm25_index(app)
    return app.state.bm25_index

def filter_documents_by_metadata(docs: List[Document], filters: Dict[str, Any]) -> List[Document]:
    """
    Applies custom filters to retrieved documents.
    Supported filters:
      - doc_id: str
      - tags: str (comma separated tags to match)
      - page: int
      - chunk_strategy: str
    """
    if not filters:
        return docs

    filtered = []
    for doc in docs:
        keep = True
        
        # Filter by doc_id
        if "doc_id" in filters and filters["doc_id"]:
            if doc.metadata.get("doc_id") != filters["doc_id"]:
                keep = False

        # Filter by chunk_strategy
        if "chunk_strategy" in filters and filters["chunk_strategy"]:
            if doc.metadata.get("chunk_strategy") != filters["chunk_strategy"]:
                keep = False

        # Filter by page number
        if "page" in filters and filters["page"]:
            try:
                page_val = int(filters["page"])
                if int(doc.metadata.get("page", -1)) != page_val:
                    keep = False
            except (ValueError, TypeError):
                pass

        # Filter by tags
        if "tags" in filters and filters["tags"]:
            filter_tags = [t.strip().lower() for t in filters["tags"].split(",") if t.strip()]
            doc_tags_str = doc.metadata.get("tags", "")
            doc_tags = [t.strip().lower() for t in doc_tags_str.split(",") if t.strip()]
            
            # Check if there is any intersection
            if not any(t in doc_tags for t in filter_tags):
                keep = False

        if keep:
            filtered.append(doc)

    return filtered

def rrf_merge(dense_results: List[Document], sparse_results: List[Document], k: int = 60, top_n: int = 20) -> List[Document]:
    """
    Merges dense and sparse search results using Reciprocal Rank Fusion (RRF).
    Score = sum(1 / (k + rank))
    """
    rrf_scores = {}
    doc_map = {}

    def process_results(results):
        for rank, doc in enumerate(results):
            # Use content and metadata to identify uniqueness
            doc_id = doc.metadata.get("doc_id", "")
            chunk_idx = doc.metadata.get("chunk_index", -1)
            key = f"{doc_id}_{chunk_idx}_{hash(doc.page_content)}"
            
            if key not in rrf_scores:
                rrf_scores[key] = 0.0
                doc_map[key] = doc
                
            rrf_scores[key] += 1.0 / (k + (rank + 1))

    process_results(dense_results)
    process_results(sparse_results)

    # Sort keys by score descending
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    # Select top_n and assign rrf_score to metadata
    final_docs = []
    for key in sorted_keys[:top_n]:
        doc = doc_map[key]
        doc.metadata["rrf_score"] = rrf_scores[key]
        final_docs.append(doc)
        
    return final_docs
