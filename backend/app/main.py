from fastapi import FastAPI
from app.config import settings
from app.core.lifespan import lifespan


app = FastAPI(title="RAG Platform", lifespan=lifespan)
print("[stage00 | main | 001] OK: FastAPI app created")


@app.get("/health")
def health_check():
    return {"status": "ok"}