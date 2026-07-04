from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from app.models.schemas import QueryRequest, QueryResponse
from app.core.logging import get_logger
from app.llm import get_llm_chain, invoke_with_semaphore
from app.services.retrieval import retrieval_service
from app.services.rag_chain import build_rag_chain
import asyncio

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        docs = retrieval_service.search(
            query=request.question,
            k=request.k,
        )
        context = "\n\n".join([doc.page_content for doc in docs])
        chain = build_rag_chain()
        answer = await invoke_with_semaphore(chain, {
            "context": context,
            "question": request.question,
        })
        return QueryResponse(
            answer=answer,
            sources=[doc.metadata.get("source", "") for doc in docs],
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    async def event_generator():
        try:
            docs = retrieval_service.search(
                query=request.question,
                k=request.k,
            )
            context = "\n\n".join([doc.page_content for doc in docs])
            chain = build_rag_chain()
            async for chunk in chain.astream({
                "context": context,
                "question": request.question,
            }):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            logger.error(f"Stream failed: {e}")
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
