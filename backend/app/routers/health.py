from fastapi import APIRouter
from app.core.logging import get_logger
from app.core.startup import check_sqlite_health
from app.llm import llm_manager

logger = get_logger(__name__)
router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check():
    status = {
        "database": "healthy" if check_sqlite_health() else "unhealthy",
        "llm": "loaded" if llm_manager._loaded_llms else "not_loaded",
        "embedding_model": "loaded",
        "vectorstore": "ready",
    }
    return status
