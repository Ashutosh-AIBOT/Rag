import sqlite3
from fastapi import APIRouter, Request
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Health"])

@router.get("/health")
async def health_check(request: Request):
    """
    Simple, beginner-friendly health check endpoint.
    Checks database connection and model loading states.
    """
    logger.info("Health check endpoint hit")
    app = request.app
    
    # 1. Check SQLite connection
    db_healthy = True
    try:
        conn = sqlite3.connect(settings.SQLITE_DB_PATH)
        conn.execute("SELECT 1;")
        conn.close()
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_healthy = False

    # 2. Check preloaded ML models on app.state
    embeddings_loaded = getattr(app.state, "embeddings", None) is not None
    cross_encoder_loaded = getattr(app.state, "cross_encoder", None) is not None
    
    # Determine overall status
    is_healthy = db_healthy and embeddings_loaded and cross_encoder_loaded
    status = "healthy" if is_healthy else "degraded"
    
    logger.info(f"System status: {status}")
    
    return {
        "status": status,
        "database": "connected" if db_healthy else "disconnected",
        "models": {
            "embeddings": "loaded" if embeddings_loaded else "missing",
            "cross_encoder": "loaded" if cross_encoder_loaded else "missing"
        }
    }
