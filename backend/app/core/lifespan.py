import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.core.logging import get_logger
from app.core.startup import init_sqlite_wal
from app.embeddings.sentence_transformer import load_embedding_model
from app.vectorstore.chroma import init_chroma
from app.llm.manager import llm_manager
from app.llm.models import ProviderName

logger = get_logger(__name__)

_semaphore: asyncio.Semaphore | None = None


def get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(settings.LLM_CONCURRENCY_LIMIT)
    return _semaphore


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Starting RAG Platform ===")

    init_sqlite_wal()
    logger.info("SQLite initialized with WAL mode")

    embeddings = load_embedding_model()
    app.state.embeddings = embeddings
    logger.info("Embedding model loaded and attached to app.state")

    chroma_store = init_chroma(embeddings)
    app.state.chroma_store = chroma_store
    logger.info("ChromaDB initialized")

    llm_manager.initialize(primary=ProviderName.NVIDIA)
    llm_manager.load_all_models()
    app.state.llm_manager = llm_manager
    logger.info("LLM Manager initialized with fallback chain")

    app.state.semaphore = get_semaphore()
    logger.info(f"Semaphore created with limit: {settings.LLM_CONCURRENCY_LIMIT}")

    logger.info("=== RAG Platform Ready ===")

    yield

    logger.info("=== Shutting down RAG Platform ===")


def get_health_status(app: FastAPI) -> dict:
    from app.core.startup import check_sqlite_health

    return {
        "status": "healthy",
        "sqlite": check_sqlite_health(),
        "embeddings_loaded": hasattr(app.state, "embeddings") and app.state.embeddings is not None,
        "chroma_ready": hasattr(app.state, "chroma_store") and app.state.chroma_store is not None,
        "llm_manager": app.state.llm_manager.status if hasattr(app.state, "llm_manager") else {},
        "semaphore_available": hasattr(app.state, "semaphore"),
    }
