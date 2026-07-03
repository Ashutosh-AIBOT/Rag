from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.core.lifespan import lifespan
from app.routers import health, documents, query

app = FastAPI(
    title="Advanced RAG Platform",
    description="Production-grade RAG with hybrid search, re-ranking, and multi-strategy retrieval",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)


@app.get("/")
async def root():
    return {"message": "Advanced RAG Platform", "docs": "/docs"}
