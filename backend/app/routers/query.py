from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.models.schemas import QueryRequest, QueryResponse
from app.core.logging import get_logger
from app.llm import get_llm_chain, invoke_with_semaphore
from app.services.search_service import search_documents
from app.services.rag_chain import build_rag_chain
from app.database.database import (
    insert_query_history,
    insert_pipeline_trace,
    get_pipeline_trace_by_query_id,
    get_query_history,
)
import asyncio
import time
import uuid
import json

logger = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


class CompareRequest(BaseModel):
    question: str
    k: int = 5
    strategy_a: str = "vector"
    strategy_b: str = "hybrid"
    rerank_a: bool = False
    rerank_b: bool = False
    rerank_top_k_a: int = 3
    rerank_top_k_b: int = 3
    filters: dict = None


class CompareResponse(BaseModel):
    query_id_a: str
    answer_a: str
    sources_a: list[str]
    trace_a: dict
    latency_ms_a: int

    query_id_b: str
    answer_b: str
    sources_b: list[str]
    trace_b: dict
    latency_ms_b: int


@router.post("", response_model=QueryResponse)
async def query_documents(request: QueryRequest, req: Request):
    try:
        start_time = time.time()
        query_id = str(uuid.uuid4())
        vectorstore = req.app.state.vectorstore

        # 1. Advanced Retrieval Strategy and Rerank
        search_res = await asyncio.to_thread(
            search_documents,
            query=request.question,
            k=request.k,
            strategy=request.strategy,
            filters=request.filters,
            rerank=request.rerank,
            rerank_top_k=request.rerank_top_k,
            vectorstore=vectorstore,
        )

        docs = search_res["documents"]
        trace = search_res["trace"]

        # 2. Context Assembly
        context = "\n\n".join([doc.page_content for doc in docs])

        # 3. LLM Answer Generation
        chain = build_rag_chain()
        answer = await invoke_with_semaphore(chain, {
            "context": context,
            "question": request.question,
        })

        latency_ms = int((time.time() - start_time) * 1000)

        # 4. Save Query History and Trace
        insert_query_history(query_id, request.question, answer, request.strategy, latency_ms)
        
        trace_id = str(uuid.uuid4())
        # Attach generated answer and latency details to trace
        trace["answer"] = answer
        trace["latency_ms"] = latency_ms
        insert_pipeline_trace(trace_id, query_id, json.dumps(trace))

        return QueryResponse(
            query_id=query_id,
            answer=answer,
            sources=[doc.metadata.get("source", "") for doc in docs],
            trace=trace,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=CompareResponse)
async def compare_strategies(request: CompareRequest, req: Request):
    try:
        vectorstore = req.app.state.vectorstore

        # Strategy A execution
        start_a = time.time()
        search_res_a = await asyncio.to_thread(
            search_documents,
            query=request.question,
            k=request.k,
            strategy=request.strategy_a,
            filters=request.filters,
            rerank=request.rerank_a,
            rerank_top_k=request.rerank_top_k_a,
            vectorstore=vectorstore,
        )
        docs_a = search_res_a["documents"]
        trace_a = search_res_a["trace"]
        context_a = "\n\n".join([doc.page_content for doc in docs_a])
        chain = build_rag_chain()
        answer_a = await invoke_with_semaphore(chain, {
            "context": context_a,
            "question": request.question,
        })
        latency_a = int((time.time() - start_a) * 1000)
        query_id_a = str(uuid.uuid4())
        insert_query_history(query_id_a, request.question, answer_a, request.strategy_a, latency_a)
        trace_id_a = str(uuid.uuid4())
        trace_a["answer"] = answer_a
        trace_a["latency_ms"] = latency_a
        insert_pipeline_trace(trace_id_a, query_id_a, json.dumps(trace_a))

        # Strategy B execution
        start_b = time.time()
        search_res_b = await asyncio.to_thread(
            search_documents,
            query=request.question,
            k=request.k,
            strategy=request.strategy_b,
            filters=request.filters,
            rerank=request.rerank_b,
            rerank_top_k=request.rerank_top_k_b,
            vectorstore=vectorstore,
        )
        docs_b = search_res_b["documents"]
        trace_b = search_res_b["trace"]
        context_b = "\n\n".join([doc.page_content for doc in docs_b])
        answer_b = await invoke_with_semaphore(chain, {
            "context": context_b,
            "question": request.question,
        })
        latency_b = int((time.time() - start_b) * 1000)
        query_id_b = str(uuid.uuid4())
        insert_query_history(query_id_b, request.question, answer_b, request.strategy_b, latency_b)
        trace_id_b = str(uuid.uuid4())
        trace_b["answer"] = answer_b
        trace_b["latency_ms"] = latency_b
        insert_pipeline_trace(trace_id_b, query_id_b, json.dumps(trace_b))

        return CompareResponse(
            query_id_a=query_id_a,
            answer_a=answer_a,
            sources_a=[doc.metadata.get("source", "") for doc in docs_a],
            trace_a=trace_a,
            latency_ms_a=latency_a,
            query_id_b=query_id_b,
            answer_b=answer_b,
            sources_b=[doc.metadata.get("source", "") for doc in docs_b],
            trace_b=trace_b,
            latency_ms_b=latency_b,
        )
    except Exception as e:
        logger.error(f"Compare failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{query_id}/pipeline")
async def get_query_pipeline_trace(query_id: str):
    try:
        trace_row = get_pipeline_trace_by_query_id(query_id)
        if not trace_row:
            raise HTTPException(status_code=404, detail="Trace not found")
        # trace_row["steps"] is the JSON-serialized string of the trace
        steps = json.loads(trace_row["steps"])
        return {"query_id": query_id, "pipeline_trace": steps}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get pipeline trace failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{query_id}/chunks")
async def get_query_chunks(query_id: str):
    try:
        trace_row = get_pipeline_trace_by_query_id(query_id)
        if not trace_row:
            raise HTTPException(status_code=404, detail="Trace not found")
        steps = json.loads(trace_row["steps"])
        
        # Return both retrieved and reranked chunks if present
        return {
            "query_id": query_id,
            "retrieved_chunks": steps.get("retrieved_chunks", []),
            "reranked_chunks": steps.get("reranked_chunks", []),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get query chunks failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def query_stream(request: QueryRequest, req: Request):
    async def event_generator():
        try:
            vectorstore = req.app.state.vectorstore
            search_res = await asyncio.to_thread(
                search_documents,
                query=request.question,
                k=request.k,
                strategy=request.strategy,
                filters=request.filters,
                rerank=request.rerank,
                rerank_top_k=request.rerank_top_k,
                vectorstore=vectorstore,
            )
            docs = search_res["documents"]
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
