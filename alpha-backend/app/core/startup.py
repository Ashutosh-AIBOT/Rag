import asyncio
import os
from fastapi import FastAPI
from app.config import settings
from app.core.logging import get_logger
from app.database.database import init_db as database_init

logger = get_logger(__name__)

def init_db() -> None:
    """
    Initializes the SQLite database tables and enables WAL mode.
    """
    # Ensure database folder exists
    db_path = settings.SQLITE_DB_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Run the database schema initialization
    database_init()


async def load_ml_models(app: FastAPI) -> None:
    """
    Preloads ML models once at startup and stores them on app.state
    to make them available for all request handlers without reloading.
    """
    logger.info("Preloading ML models at startup...")

    # 1. Load Sentence-Transformers Embedding Model
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        app.state.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        logger.info("Embedding model preloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to preload embedding model: {e}")
        app.state.embeddings = None

    # 2. Load Cross-Encoder Re-ranker Model
    try:
        from sentence_transformers import CrossEncoder
        app.state.cross_encoder = CrossEncoder(settings.CROSS_ENCODER_MODEL)
        logger.info("Cross-Encoder model preloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to preload Cross-Encoder model: {e}")
        app.state.cross_encoder = None

    # 3. Initialize placeholders for dynamic states
    app.state.bm25_index = None

    # 4. Initialize LLM Manager and Fallback Handler
    try:
        from app.llm.manager import llm_manager
        from app.llm.models import ProviderName
        llm_manager.initialize(primary=ProviderName(settings.DEFAULT_LLM_PROVIDER))
        app.state.llm_manager = llm_manager
        logger.info("LLM Manager initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize LLM Manager: {e}")
        app.state.llm_manager = None

    # 5. Initialize Semaphore to restrict max concurrent LLM API requests
    app.state.llm_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_LLM_REQUESTS)
    
    logger.info("ML models loading complete.")
