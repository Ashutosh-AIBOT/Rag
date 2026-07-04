from fastapi import APIRouter, Request
from app.core.logging import get_logger
from app.services.bm25_retriever import get_bm25_retriever
from app.services.hybrid_retriever import get_hybrid_retriever
from app.services.rrf import get_rrf_retriever

logger = get_logger(__name__)
router = APIRouter(prefix="/strategies", tags=["strategies"])


@router.get("")
async def get_strategies():
    return {
        "strategies": [
            {
                "name": "vector",
                "description": "Pure vector similarity search using ChromaDB",
                "status": "available",
            },
            {
                "name": "bm25",
                "description": "BM25 keyword-based retrieval using rank_bm25",
                "status": "available",
            },
            {
                "name": "hybrid",
                "description": "Ensemble of vector + BM25 with configurable weights",
                "status": "available",
            },
            {
                "name": "rrf",
                "description": "Reciprocal Rank Fusion combining multiple retrievers",
                "status": "available",
            },
        ]
    }


@router.get("/bm25/status")
async def bm25_status():
    retriever = get_bm25_retriever()
    return {
        "initialized": retriever.bm25 is not None,
        "document_count": len(retriever.documents),
    }


@router.get("/hybrid/status")
async def hybrid_status():
    try:
        retriever = get_hybrid_retriever(None)
        return {
            "initialized": True,
            "weights": retriever.weights,
        }
    except Exception as e:
        return {
            "initialized": False,
            "error": str(e),
        }


@router.get("/rrf/status")
async def rrf_status():
    try:
        retriever = get_rrf_retriever([])
        return {
            "initialized": True,
            "k": retriever.k,
        }
    except Exception as e:
        return {
            "initialized": False,
            "error": str(e),
        }
