from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.logging import get_logger
from app.core.startup import init_sqlite_wal, check_sqlite_health
from app.embeddings.sentence_transformer import load_embedding_model
from app.vectorstore.chroma import initialize_chroma
from app.llm.manager import llm_manager
from app.services.retrieval import set_vectorstore

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup starting...")
    init_sqlite_wal()

    if not check_sqlite_health():
        raise RuntimeError("SQLite health check failed")

    embedding_model = load_embedding_model()
    app.state.embeddings = embedding_model

    vectorstore = initialize_chroma(embedding_model)
    app.state.vectorstore = vectorstore
    set_vectorstore(vectorstore)

    llm_manager.initialize()
    llm_manager.load_all_models()

    logger.info("App ready")
    yield
    logger.info("App stopped")
