from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.logging import get_logger
from app.core.startup import init_db, load_ml_models

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of the FastAPI application.
    Executes startup hooks to initialize connections and load models,
    and handles graceful shutdown/cleanup.
    """
    # ------------------ STARTUP ------------------
    logger.info("Executing lifespan startup hooks...")
    
    try:
        # Initialize SQLite with WAL mode
        init_db()
        
        # Preload Embedding models, Cross-encoders, and LLM semaphore
        await load_ml_models(app)
        
        logger.info("Lifespan startup hooks completed successfully. App is ready to receive requests.")
    except Exception as e:
        logger.critical(f"App initialization failed during startup hooks: {e}", exc_info=True)
        raise e

    yield

    # ----------------- SHUTDOWN ------------------
    logger.info("Executing lifespan shutdown hooks...")
    
    # Clean up resources stored on app state
    try:
        if hasattr(app.state, "embeddings"):
            del app.state.embeddings
        if hasattr(app.state, "cross_encoder"):
            del app.state.cross_encoder
        if hasattr(app.state, "llm_semaphore"):
            del app.state.llm_semaphore
            
        logger.info("Lifespan shutdown hooks completed. Resources released.")
    except Exception as e:
        logger.error(f"Error encountered during lifespan shutdown cleanup: {e}", exc_info=True)
