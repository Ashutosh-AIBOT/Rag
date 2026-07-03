import asyncio
from typing import List
from langchain_core.documents import Document
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

async def rerank_documents(
    query: str,
    documents: List[Document],
    app,
    top_n: int = 5
) -> List[Document]:
    """
    Reranks a list of retrieved documents using a Cross-Encoder or Cohere Rerank API.
    Falls back to local Cross-Encoder if Cohere fails or is not configured.
    """
    if not documents:
        return []

    # 1. Attempt Cohere Rerank if API key is configured
    if settings.COHERE_API_KEY:
        try:
            import cohere
            logger.info("Attempting Cohere Rerank API...")
            co = cohere.Client(api_key=settings.COHERE_API_KEY)
            
            # Format documents for Cohere API
            docs_content = [doc.page_content for doc in documents]
            # Since Cohere is network-bound, wrap in executor thread
            response = await asyncio.to_thread(
                co.rerank,
                model="rerank-english-v3.0",
                query=query,
                documents=docs_content,
                top_n=top_n
            )
            
            reranked_docs = []
            for result in response.results:
                orig_doc = documents[result.index]
                # Store rerank score in metadata
                orig_doc.metadata["rerank_score"] = result.relevance_score
                orig_doc.metadata["rerank_rank"] = result.index
                reranked_docs.append(orig_doc)
                
            logger.info("Cohere Rerank completed successfully.")
            return reranked_docs
        except Exception as e:
            logger.error(f"Cohere Rerank failed, falling back to local Cross-Encoder: {e}")

    # 2. Local Cross-Encoder Fallback
    cross_encoder = getattr(app.state, "cross_encoder", None)
    if cross_encoder is not None:
        try:
            logger.info("Running local Cross-Encoder reranking...")
            # Prepare pairs of (query, document_text)
            pairs = [[query, doc.page_content] for doc in documents]
            # Offload CPU-heavy cross-encoder model inference to a worker thread
            scores = await asyncio.to_thread(cross_encoder.predict, pairs)
            
            # Attach scores to documents
            for doc, score in zip(documents, scores):
                doc.metadata["rerank_score"] = float(score)
                
            # Sort by score descending
            sorted_docs = sorted(documents, key=lambda x: x.metadata["rerank_score"], reverse=True)
            logger.info("Local Cross-Encoder reranking complete.")
            return sorted_docs[:top_n]
        except Exception as e:
            logger.error(f"Local Cross-Encoder reranking failed: {e}", exc_info=True)
    
    # 3. If all else fails, return original top_n results with a default rerank_score
    logger.warning("No reranking applied (both Cohere and local cross-encoder failed/not loaded).")
    for doc in documents[:top_n]:
        doc.metadata["rerank_score"] = 1.0
    return documents[:top_n]

