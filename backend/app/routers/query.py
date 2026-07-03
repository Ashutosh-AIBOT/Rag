from fastapi import APIRouter, Request

from app.core.logging import get_logger
from app.core.dependencies import get_chroma_store, get_llm_manager, get_semaphore
from app.database.schemas import QueryRequest, QueryResponse
from app.services.retrieval import retrieve_chunks
from app.services.rag_chain import query_rag

logger = get_logger(__name__)

router = APIRouter(prefix="/api/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query_documents(request: Request, body: QueryRequest):
    chroma_store = get_chroma_store(request)
    llm_manager = get_llm_manager(request)
    semaphore = get_semaphore(request)

    chunks = retrieve_chunks(chroma_store, body.query, k=body.top_k, strategy=body.strategy)

    if not chunks:
        return QueryResponse(
            answer="No relevant documents found for your query.",
            source_chunks=[],
            provider="none",
            model="none",
        )

    result = await query_rag(llm_manager, semaphore, body.query, chunks)

    return QueryResponse(
        answer=result["answer"],
        source_chunks=[
            {"content": c["content"], "metadata": c["metadata"], "score": c["score"]}
            for c in chunks
        ],
        provider=result["provider"],
        model=result["model"],
    )
