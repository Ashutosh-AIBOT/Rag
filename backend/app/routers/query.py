import asyncio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.services.rag_chain import get_rag_chain
from app.llm.manager import _semaphore
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    query: str
    document_ids: list[str] | None = None


class QueryResponse(BaseModel):
    answer: str
    query: str


@router.post("", response_model=QueryResponse)
async def query_documents(request: Request, body: QueryRequest):
    try:
        vectorstore = request.app.state.vectorstore
        chain = get_rag_chain(vectorstore, body.document_ids)

        async def run_with_semaphore():
            async with _semaphore:
                return await asyncio.to_thread(chain.invoke, {"question": body.query})

        answer = await run_with_semaphore()
        logger.info("Query answered")
        return QueryResponse(answer=answer, query=body.query)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Query failed")


@router.post("/stream")
async def query_stream(request: Request, body: QueryRequest):
    try:
        vectorstore = request.app.state.vectorstore
        chain = get_rag_chain(vectorstore, body.document_ids)

        async def generate():
            async for chunk in chain.astream({"question": body.query}):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        logger.info("Stream started")
        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Stream failed: {e}")
        raise HTTPException(status_code=500, detail="Stream failed")
