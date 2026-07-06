from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
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
    list_recent_queries,
    create_job,
    update_job_status,
    get_job_status,
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
    embedding_model_a: str = "huggingface"
    embedding_model_b: str = "huggingface"
    compress_a: bool = False
    compress_b: bool = False


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


def get_vectorstore_by_embedding(embedding_name: str, req: Request):
    if embedding_name == "gemini":
        from app.vectorstore.chroma import initialize_chroma_gemini
        if not hasattr(req.app.state, "vectorstore_gemini") or req.app.state.vectorstore_gemini is None:
            req.app.state.vectorstore_gemini = initialize_chroma_gemini()
        return req.app.state.vectorstore_gemini
    return req.app.state.vectorstore


@router.post("", response_model=QueryResponse)
async def query_documents(request: QueryRequest, req: Request):
    try:
        start_time = time.time()
        query_id = str(uuid.uuid4())
        vectorstore = get_vectorstore_by_embedding(request.embedding_model, req)

        # Check semantic cache first
        from app.services.semantic_cache import semantic_cache
        cached_result = await asyncio.to_thread(semantic_cache.get, request.question)
        if cached_result:
            latency_ms = int((time.time() - start_time) * 1000)
            insert_query_history(query_id, request.question, cached_result["answer"], request.strategy, latency_ms)
            
            trace_id = str(uuid.uuid4())
            cached_result["trace"]["cached_hit"] = True
            cached_result["trace"]["latency_ms"] = latency_ms
            insert_pipeline_trace(trace_id, query_id, json.dumps(cached_result["trace"]))
            
            return QueryResponse(
                query_id=query_id,
                answer=cached_result["answer"],
                sources=cached_result["sources"],
                trace=cached_result["trace"],
            )

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
            compress=request.compress,
        )

        docs = search_res["documents"]
        trace = search_res["trace"]

        # 2. Context Assembly
        context = "\n\n".join([doc.page_content for doc in docs])

        # 3. LLM Answer Generation
        from app.core.callbacks import MetricsTrackingCallbackHandler
        handler = MetricsTrackingCallbackHandler()
        chain = build_rag_chain()
        answer = await invoke_with_semaphore(chain, {
            "context": context,
            "question": request.question,
        }, config={"callbacks": [handler]})

        latency_ms = int((time.time() - start_time) * 1000)

        # 4. Deduct User Token Budget
        user_id = req.headers.get("X-User-ID", "default_user")
        tokens_consumed = handler.total_tokens if handler.total_tokens > 0 else (len(request.question) + len(answer)) // 4
        from app.database.database import consume_user_tokens
        await asyncio.to_thread(consume_user_tokens, user_id, tokens_consumed)

        # 5. Save Query History and Trace
        insert_query_history(query_id, request.question, answer, request.strategy, latency_ms)
        
        trace_id = str(uuid.uuid4())
        # Attach generated answer and latency details to trace
        trace["answer"] = answer
        trace["latency_ms"] = latency_ms
        insert_pipeline_trace(trace_id, query_id, json.dumps(trace))

        # Store in semantic cache asynchronously/thread
        sources = [doc.metadata.get("source", "") for doc in docs]
        await asyncio.to_thread(semantic_cache.set, request.question, answer, sources, trace)

        return QueryResponse(
            query_id=query_id,
            answer=answer,
            sources=sources,
            trace=trace,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_single_strategy(
    question: str,
    k: int,
    strategy: str,
    filters: dict,
    rerank: bool,
    rerank_top_k: int,
    vectorstore,
    compress: bool = False,
) -> dict:
    start_time = time.time()
    search_res = await asyncio.to_thread(
        search_documents,
        query=question,
        k=k,
        strategy=strategy,
        filters=filters,
        rerank=rerank,
        rerank_top_k=rerank_top_k,
        vectorstore=vectorstore,
        compress=compress,
    )
    docs = search_res["documents"]
    trace = search_res["trace"]
    context = "\n\n".join([doc.page_content for doc in docs])
    from app.core.callbacks import MetricsTrackingCallbackHandler
    handler = MetricsTrackingCallbackHandler()
    chain = build_rag_chain()
    answer = await invoke_with_semaphore(chain, {
        "context": context,
        "question": question,
    }, config={"callbacks": [handler]})
    latency_ms = int((time.time() - start_time) * 1000)
    query_id = str(uuid.uuid4())
    
    # Run SQLite database inserts in a threadpool to prevent blocking the event loop
    await asyncio.to_thread(insert_query_history, query_id, question, answer, strategy, latency_ms)
    trace["answer"] = answer
    trace["latency_ms"] = latency_ms
    await asyncio.to_thread(insert_pipeline_trace, str(uuid.uuid4()), query_id, json.dumps(trace))
    
    return {
        "query_id": query_id,
        "answer": answer,
        "docs": docs,
        "trace": trace,
        "latency_ms": latency_ms
    }


async def process_comparison_job(
    job_id: str,
    question: str,
    k: int,
    strategy_a: str,
    strategy_b: str,
    rerank_a: bool,
    rerank_b: bool,
    rerank_top_k_a: int,
    rerank_top_k_b: int,
    filters: dict,
    vectorstore_a,
    vectorstore_b,
    compress_a: bool = False,
    compress_b: bool = False,
):
    try:
        # Check cancellation state before commencing
        job = await asyncio.to_thread(get_job_status, job_id)
        if job and job["status"] == "cancelled":
            logger.info(f"Comparison job {job_id} cancelled before start.")
            return

        await asyncio.to_thread(update_job_status, job_id, "processing", 0.1)

        # Run concurrent strategy tasks
        task_a = run_single_strategy(
            question=question,
            k=k,
            strategy=strategy_a,
            filters=filters,
            rerank=rerank_a,
            rerank_top_k=rerank_top_k_a,
            vectorstore=vectorstore_a,
            compress=compress_a,
        )
        task_b = run_single_strategy(
            question=question,
            k=k,
            strategy=strategy_b,
            filters=filters,
            rerank=rerank_b,
            rerank_top_k=rerank_top_k_b,
            vectorstore=vectorstore_b,
            compress=compress_b,
        )

        res_a, res_b = await asyncio.gather(task_a, task_b)

        # Check cancellation before finalizing
        job = await asyncio.to_thread(get_job_status, job_id)
        if job and job["status"] == "cancelled":
            logger.info(f"Comparison job {job_id} cancelled during processing.")
            return

        result_payload = {
            "query_id_a": res_a["query_id"],
            "answer_a": res_a["answer"],
            "sources_a": [doc.metadata.get("source", "") for doc in res_a["docs"]],
            "trace_a": res_a["trace"],
            "latency_ms_a": res_a["latency_ms"],
            "query_id_b": res_b["query_id"],
            "answer_b": res_b["answer"],
            "sources_b": [doc.metadata.get("source", "") for doc in res_b["docs"]],
            "trace_b": res_b["trace"],
            "latency_ms_b": res_b["latency_ms"],
        }

        await asyncio.to_thread(update_job_status, job_id, "completed", 1.0, json.dumps(result_payload))
    except Exception as e:
        logger.error(f"Async comparison job {job_id} failed: {e}")
        await asyncio.to_thread(update_job_status, job_id, "failed", 1.0, error=str(e))


@router.post("/compare", response_model=CompareResponse)
async def compare_strategies(request: CompareRequest, req: Request):
    try:
        vectorstore_a = get_vectorstore_by_embedding(request.embedding_model_a, req)
        vectorstore_b = get_vectorstore_by_embedding(request.embedding_model_b, req)

        # Execute Strategy A and Strategy B in parallel
        task_a = run_single_strategy(
            question=request.question,
            k=request.k,
            strategy=request.strategy_a,
            filters=request.filters,
            rerank=request.rerank_a,
            rerank_top_k=request.rerank_top_k_a,
            vectorstore=vectorstore_a,
            compress=request.compress_a,
        )
        task_b = run_single_strategy(
            question=request.question,
            k=request.k,
            strategy=request.strategy_b,
            filters=request.filters,
            rerank=request.rerank_b,
            rerank_top_k=request.rerank_top_k_b,
            vectorstore=vectorstore_b,
            compress=request.compress_b,
        )

        res_a, res_b = await asyncio.gather(task_a, task_b)

        return CompareResponse(
            query_id_a=res_a["query_id"],
            answer_a=res_a["answer"],
            sources_a=[doc.metadata.get("source", "") for doc in res_a["docs"]],
            trace_a=res_a["trace"],
            latency_ms_a=res_a["latency_ms"],
            query_id_b=res_b["query_id"],
            answer_b=res_b["answer"],
            sources_b=[doc.metadata.get("source", "") for doc in res_b["docs"]],
            trace_b=res_b["trace"],
            latency_ms_b=res_b["latency_ms"],
        )
    except Exception as e:
        logger.error(f"Compare failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare/async", status_code=202)
async def compare_strategies_async(request: CompareRequest, req: Request, background_tasks: BackgroundTasks):
    try:
        job_id = str(uuid.uuid4())
        await asyncio.to_thread(create_job, job_id, "comparison")

        vectorstore_a = get_vectorstore_by_embedding(request.embedding_model_a, req)
        vectorstore_b = get_vectorstore_by_embedding(request.embedding_model_b, req)

        try:
            from app.tasks.evaluation import process_comparison_job_task
            process_comparison_job_task.delay(
                job_id=job_id,
                question=request.question,
                k=request.k,
                strategy_a=request.strategy_a,
                strategy_b=request.strategy_b,
                rerank_a=request.rerank_a,
                rerank_b=request.rerank_b,
                rerank_top_k_a=request.rerank_top_k_a,
                rerank_top_k_b=request.rerank_top_k_b,
                filters=request.filters,
                embedding_model_a=request.embedding_model_a,
                embedding_model_b=request.embedding_model_b,
                compress_a=request.compress_a,
                compress_b=request.compress_b,
            )
        except Exception:
            background_tasks.add_task(
                process_comparison_job,
                job_id=job_id,
                question=request.question,
                k=request.k,
                strategy_a=request.strategy_a,
                strategy_b=request.strategy_b,
                rerank_a=request.rerank_a,
                rerank_b=request.rerank_b,
                rerank_top_k_a=request.rerank_top_k_a,
                rerank_top_k_b=request.rerank_top_k_b,
                filters=request.filters,
                vectorstore_a=vectorstore_a,
                vectorstore_b=vectorstore_b,
                compress_a=request.compress_a,
                compress_b=request.compress_b,
            )

        return {"job_id": job_id, "status": "pending"}
    except Exception as e:
        logger.error(f"Failed to submit comparison job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recent")
async def get_recent_queries(limit: int = 10):
    try:
        queries = list_recent_queries(limit)
        return {"queries": queries}
    except Exception as e:
        logger.error(f"Get recent queries failed: {e}")
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
            start_time = time.time()
            query_id = str(uuid.uuid4())
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
            trace = search_res["trace"]
            context = "\n\n".join([doc.page_content for doc in docs])
            chain = build_rag_chain()
            full_answer = ""
            async for chunk in chain.astream({
                "context": context,
                "question": request.question,
            }):
                full_answer += chunk
                yield f"data: {chunk}\n\n"

            latency_ms = int((time.time() - start_time) * 1000)

            # Save query history and pipeline trace
            insert_query_history(query_id, request.question, full_answer, request.strategy, latency_ms)
            trace["answer"] = full_answer
            trace["latency_ms"] = latency_ms
            trace_id = str(uuid.uuid4())
            insert_pipeline_trace(trace_id, query_id, json.dumps(trace))
        except Exception as e:
            logger.error(f"Stream failed: {e}")
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

