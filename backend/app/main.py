cd /home/creator/Desktop/ExcellenceTechnology/06.Rag
from fastapi import FastAPI
from app.config import settings
from app.core.lifespan import lifespan
from app.routers import documents


app = FastAPI(title="RAG Platform", lifespan=lifespan)
print("[stage00 | main | 001-A] OK: FastAPI app created")

app.include_router(documents.router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
