import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel

from app.config import get_settings
from app.models.database import get_db, QueryLog, Document, User
from app.models.schemas import RETRIEVAL_STRATEGIES
from app.routers.auth import _get_current_user
from app.services.llm_factory import llm_provider_status, estimate_cost_usd

logger = logging.getLogger("misc_router")
settings = get_settings()

router = APIRouter(prefix="/api", tags=["misc"])

STRATEGY_DESCRIPTIONS = {
    "basic_vector": "Standard cosine-similarity vector search (baseline).",
    "hybrid": "Semantic + BM25 combined via Reciprocal Rank Fusion.",
    "hybrid_rerank": "Hybrid retrieval, top-20, then cross-encoder rerank to top-5. Recommended default.",
    "parent_child": "Search small child chunks, return larger parent chunks for context.",
    "multi_query": "LLM generates query variants, merges results across all of them.",
    "hyde": "Embeds a hypothetical LLM-generated answer instead of the raw query.",
    "decomposition": "Breaks complex questions into sub-questions, retrieves for each.",
    "step_back": "Generates a broader 'step-back' question alongside the specific one for extra recall.",
    "auto": "Picks the best strategy automatically based on the shape of the query.",
    "multi_vector": "[Bonus] Indexes summaries + hypothetical questions per chunk; retrieval uses richer semantic signals than raw chunk text.",
    "section_search": "Search over section-based chunks (one chunk per H1/H2 section). Best for structured documents.",
}


@router.get("/strategies")
def list_strategies():
    logger.info("Strategies list requested: count=%d", len(RETRIEVAL_STRATEGIES))
    return [{"id": s, "description": STRATEGY_DESCRIPTIONS.get(s, "")} for s in RETRIEVAL_STRATEGIES]


@router.get("/stats")
def get_stats(db: Session = Depends(get_db), current_user: User = Depends(_get_current_user)):
    logger.info("Stats requested: querying database for user %s", current_user.email)
    total_queries = db.query(func.count(QueryLog.id)).filter(QueryLog.user_id == current_user.id).scalar() or 0
    avg_latency = db.query(func.avg(QueryLog.latency_ms)).filter(QueryLog.user_id == current_user.id).scalar() or 0.0
    total_docs = db.query(func.count(Document.id)).filter(Document.user_id == current_user.id).scalar() or 0
    total_input_tokens = db.query(func.sum(QueryLog.input_tokens)).filter(QueryLog.user_id == current_user.id).scalar() or 0
    total_output_tokens = db.query(func.sum(QueryLog.output_tokens)).filter(QueryLog.user_id == current_user.id).scalar() or 0
    estimated_total_cost_usd = estimate_cost_usd(total_input_tokens, total_output_tokens)

    by_strategy = (
        db.query(QueryLog.strategy, func.count(QueryLog.id), func.avg(QueryLog.latency_ms))
        .filter(QueryLog.user_id == current_user.id)
        .group_by(QueryLog.strategy)
        .all()
    )

    return {
        "total_queries": total_queries,
        "total_documents": total_docs,
        "avg_latency_ms": round(avg_latency, 2),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "estimated_total_cost_usd": estimated_total_cost_usd,
        "by_strategy": [
            {"strategy": s, "count": c, "avg_latency_ms": round(l, 2) if l else 0}
            for s, c, l in by_strategy
        ],
    }


@router.get("/health")
def health():
    logger.debug("Health check requested")
    return {
        "status": "ok",
        "llm": llm_provider_status(),
        "concurrency": {
            "max_concurrent_queries": settings.max_concurrent_queries,
            "ingestion_worker_threads": settings.ingestion_worker_threads,
            "eval_worker_threads": settings.eval_worker_threads,
        },
    }


class EmbeddingCompareRequest(BaseModel):
    text: str
    model_a: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_b: str = "sentence-transformers/all-MiniLM-L6-v2"


@router.post("/embeddings/compare")
def compare_embeddings(body: EmbeddingCompareRequest):
    """Compare embeddings from two models side-by-side (Bonus #6)."""
    logger.info("Embedding compare requested: text_len=%d model_a=%s model_b=%s", len(body.text), body.model_a, body.model_b)
    import numpy as np

    def _get_emb_for_model(model_name: str):
        provider = settings.embedding_provider.lower()
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=model_name, api_key=settings.openai_api_key)
        try:
            from langchain_community.embeddings import FastEmbedEmbeddings
            return FastEmbedEmbeddings(model_name=model_name)
        except Exception:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})

    try:
        emb_a = _get_emb_for_model(body.model_a).embed_query(body.text)
        emb_b = _get_emb_for_model(body.model_b).embed_query(body.text)

        def model_info(emb, name):
            arr = np.array(emb)
            return {
                "model": name,
                "dimensions": len(emb),
                "magnitude": float(np.linalg.norm(arr)),
                "embedding": [round(x, 6) for x in emb[:20]],  # First 20 values
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
            }

        return {
            "text": body.text,
            "model_a": model_info(emb_a, body.model_a),
            "model_b": model_info(emb_b, body.model_b),
            "cosine_similarity": float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))) if np.linalg.norm(emb_a) > 0 and np.linalg.norm(emb_b) > 0 else 0.0,
        }
    except Exception as e:
        logger.error("Embedding comparison failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Embedding comparison failed: {e}")
