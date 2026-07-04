from fastapi import APIRouter, Request, HTTPException
from app.core.logging import get_logger
from app.services.search_service import search_documents
from app.database.database import get_retrieval_stats
import asyncio

logger = get_logger(__name__)
router = APIRouter(tags=["stats"])


@router.get("/chunks/search")
async def search_chunks_direct(
    q: str,
    k: int = 5,
    strategy: str = "vector",
    rerank: bool = False,
    rerank_top_k: int = 3,
    source: str = None,
    page: int = None,
    tags: str = None,
    section: str = None,
    req: Request = None,
):
    try:
        vectorstore = req.app.state.vectorstore
        filters = {}
        if source:
            filters["source"] = source
        if page is not None:
            filters["page"] = page
        if tags:
            filters["tags"] = tags
        if section:
            filters["section"] = section
            
        search_res = await asyncio.to_thread(
            search_documents,
            query=q,
            k=k,
            strategy=strategy,
            filters=filters,
            rerank=rerank,
            rerank_top_k=rerank_top_k,
            vectorstore=vectorstore,
        )
        return {
            "retrieved_chunks": search_res["trace"].get("retrieved_chunks", []),
            "reranked_chunks": search_res["trace"].get("reranked_chunks", []),
        }
    except Exception as e:
        logger.error(f"Direct chunk search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats_endpoint():
    try:
        stats = get_retrieval_stats()
        return stats
    except Exception as e:
        logger.error(f"Get stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
