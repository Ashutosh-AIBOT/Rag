from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.core.lifespan import lifespan
from app.routers import documents, query, health, hybrid, rerank, evaluation, stats, jobs
from app.core.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(title="RAG Platform", lifespan=lifespan)
logger.info("FastAPI app created")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.core.middleware import TokenGatingMiddleware
app.add_middleware(TokenGatingMiddleware)

app.include_router(documents.router, prefix="/api")
app.include_router(query.router, prefix="/api")
app.include_router(health.router, prefix="/api")
app.include_router(hybrid.router, prefix="/api")
app.include_router(rerank.router, prefix="/api")
app.include_router(evaluation.router, prefix="/api")
app.include_router(stats.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")
