from fastapi import APIRouter, Request
from app.core.logging import get_logger
from app.core.startup import check_sqlite_health

logger = get_logger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(request: Request):
    try:
        db_status = check_sqlite_health()
        embedding_loaded = hasattr(request.app.state, "embeddings")
        vectorstore_loaded = hasattr(request.app.state, "vectorstore")

        return {
            "status": "ok",
            "database": "healthy" if db_status else "unhealthy",
            "embedding_model": "loaded" if embedding_loaded else "not loaded",
            "vectorstore": "loaded" if vectorstore_loaded else "not loaded",
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "detail": str(e)}
