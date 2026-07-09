"""
Re-ranks an initial retrieval list (e.g. top-20 from hybrid search) using
a more precise cross-encoder that jointly encodes (query, chunk) pairs --
much more accurate than bi-encoder cosine similarity, at the cost of being
too slow to run over the whole corpus (hence: retrieve top-20 cheaply,
then rerank down to top-5).
"""
import logging
from functools import lru_cache
from typing import List, Dict, Any

from app.config import get_settings

logger = logging.getLogger("reranker_service")
settings = get_settings()


@lru_cache
def _get_cross_encoder():
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    return HuggingFaceCrossEncoder(
        model_name=settings.cross_encoder_model,
        model_kwargs={"device": "cpu"}
    )


def rerank_local(query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    if not chunks:
        return []
    model = _get_cross_encoder()
    pairs = [(query, c["text"]) for c in chunks]

    batch_size = 32
    all_scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        prediction = model.score(batch)
        all_scores.extend(prediction)

    for c, score in zip(chunks, all_scores):
        c["rerank_score"] = float(score)

    reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
    for i, c in enumerate(reranked, start=1):
        c["final_rank"] = i
    return reranked


def rerank_cohere(query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    try:
        import cohere
    except ImportError:
        logger.warning("cohere package not installed; falling back to local reranker.")
        return rerank_local(query, chunks, top_k)

    if not settings.cohere_api_key:
        logger.warning("Cohere API key not configured; falling back to local reranker.")
        return rerank_local(query, chunks, top_k)

    try:
        co = cohere.Client(settings.cohere_api_key)
        docs = [c["text"] for c in chunks]
        resp = co.rerank(query=query, documents=docs, top_n=top_k, model="rerank-english-v3.0")
        reranked = []
        for i, result in enumerate(resp.results, start=1):
            chunk = dict(chunks[result.index])
            chunk["rerank_score"] = float(result.relevance_score)
            chunk["final_rank"] = i
            reranked.append(chunk)
        return reranked
    except Exception as e:
        logger.warning("Cohere rerank failed: %s; falling back to local reranker.", e)
        return rerank_local(query, chunks, top_k)


def rerank(query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    if not chunks:
        return []
    if settings.reranker_provider.lower() == "cohere" and settings.cohere_api_key:
        return rerank_cohere(query, chunks, top_k)
    return rerank_local(query, chunks, top_k)
