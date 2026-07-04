from fastapi import APIRouter, Request
from pydantic import BaseModel
from app.core.logging import get_logger
from app.services.reranker import get_reranker
from app.services.retrieval import retrieval_service
from app.services.hybrid_retriever import get_hybrid_retriever
import asyncio

logger = get_logger(__name__)
router = APIRouter(prefix="/rerank", tags=["rerank"])


class RerankRequest(BaseModel):
    question: str
    k: int = 5
    top_k: int = 3
    strategy: str = "vector"


class RerankResponse(BaseModel):
    results: list[dict]
    original_count: int
    reranked_count: int


@router.post("", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest, req: Request):
    try:
        if request.strategy == "hybrid":
            hybrid_retriever = get_hybrid_retriever(req.app.state.vectorstore)
            docs = await asyncio.to_thread(hybrid_retriever.search, request.question, request.k)
        else:
            docs = retrieval_service.invoke({"query": request.question, "k": request.k})

        reranker = get_reranker()
        reranked = await asyncio.to_thread(
            reranker.rerank, request.question, docs, request.top_k
        )

        results = [
            {
                "content": doc.page_content[:200],
                "metadata": doc.metadata,
                "original_rank": doc.metadata.get("original_rank", 0),
                "reranked_position": doc.metadata.get("reranked_position", 0),
                "relevance_score": doc.metadata.get("relevance_score", 0),
            }
            for doc in reranked
        ]

        logger.info(f"Reranked {len(docs)} docs to {len(reranked)} docs")
        return RerankResponse(
            results=results,
            original_count=len(docs),
            reranked_count=len(reranked),
        )
    except Exception as e:
        logger.error(f"Rerank failed: {e}")
        raise
