from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.logging import get_logger
from app.core.startup import init_sqlite_wal, check_sqlite_health
from app.embeddings.sentence_transformer import load_embedding_model
from app.vectorstore.chroma import initialize_chroma

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[stage00 | lifespan | 004-A] OK: Startup starting...")
    init_sqlite_wal()

    if not check_sqlite_health():
        raise RuntimeError("SQLite health check failed")

    # Load embedding model
    embedding_model = load_embedding_model()
    app.state.embedding_model = embedding_model

    # Initialize ChromaDB
    vectorstore = initialize_chroma(embedding_model)
    app.state.vectorstore = vectorstore

    print("[stage00 | lifespan | 004-B] OK: App ready")
    yield
    print("[stage00 | lifespan | 004-C] OK: App stopped")
