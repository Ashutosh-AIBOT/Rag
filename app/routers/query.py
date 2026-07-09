import json
import time
import anyio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from app.models.database import get_db, QueryLog, User
from app.models.schemas import (
    QueryRequest, QueryResponse, ChunkScore, PipelineStep,
    CompareRequest, CompareResponse, QueryLogOut,
)
from app.services.rag_chain import run_rag_query, assemble_context, stream_rag_answer, count_tokens
from app.services.retrieval import (
    basic_vector_search, hybrid_search, parent_child_search,
)
from app.services.evaluator import evaluate_faithfulness, evaluate_relevancy
from app.services.pipeline_tracer import PipelineTracer
from app.services.rag_chain import retrieve
from app.services.job_manager import get_query_semaphore
from app.services.cache import get_cache, set_cache, query_result_key
from app.services.llm_factory import current_primary_provider
from app.routers.auth import _get_current_user

logger = logging.getLogger("query_router")
router = APIRouter(prefix="/api", tags=["query"])


async def _run_rag_query_bounded(*args, **kwargs):
    """Runs the (synchronous, CPU/IO heavy) RAG pipeline off the event loop
    in FastAPI's worker threadpool, gated by a semaphore so at most
    `MAX_CONCURRENT_QUERIES` run at once. Extra concurrent users simply wait
    a beat here instead of the request queue exhausting the LLM provider's
    rate limit or the machine's CPU all at once -- this is what lets 5-10
    concurrent users hit the app smoothly without the UI stalling for
    everyone."""
    sem = get_query_semaphore()
    async with sem:
        return await anyio.to_thread.run_sync(lambda: run_rag_query(*args, **kwargs))


def _to_chunk_scores(chunks):
    return [
        ChunkScore(
            chunk_id=c.get("chunk_id", ""),
            text=c.get("text", ""),
            source=c.get("source", ""),
            page=c.get("page"),
            section=c.get("section"),
            strategy=c.get("strategy", ""),
            semantic_score=c.get("semantic_score"),
            bm25_score=c.get("bm25_score"),
            rrf_score=c.get("rrf_score"),
            rerank_score=c.get("rerank_score"),
            original_rank=c.get("original_rank"),
            final_rank=c.get("final_rank"),
            token_count=count_tokens(c.get("text", "")),
            child_text=c.get("child_text"),
        )
        for c in chunks
    ]


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest, db: Session = Depends(get_db), current_user: User = Depends(_get_current_user)):
    # Check cache first
    import hashlib
    filters_hash = hashlib.md5(str(req.filters.model_dump()).encode()).hexdigest()[:8]
    cache_key = query_result_key(req.query, req.strategy, filters_hash)
    cached = get_cache(cache_key)
    if cached:
        logger.info("[%s] Query cache HIT: strategy=%s query=%r", current_user.email, req.strategy, req.query[:50])
        return QueryResponse(**cached)

    logger.info("[%s] Query started: strategy=%s query=%r filters=%s", current_user.email, req.strategy, req.query[:80], req.filters.model_dump())
    result = await _run_rag_query_bounded(
        query=req.query,
        strategy=req.strategy,
        filters=req.filters,
        top_k_initial=req.top_k_initial,
        top_k_final=req.top_k_final,
        semantic_weight=req.semantic_weight,
        bm25_weight=req.bm25_weight,
        compress_context=req.compress_context,
        user_id=current_user.id,
    )

    log = QueryLog(
        query=req.query,
        strategy=req.strategy,
        filters=req.filters.model_dump(),
        answer=result["answer"],
        trace=result["pipeline"],
        chunks=result["chunks"],
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        latency_ms=result["latency_ms"],
        user_id=current_user.id,
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    logger.info("[%s] Query completed: strategy=%s latency=%.2fms chunks=%d in_tokens=%d out_tokens=%d cost=$%.6f",
                current_user.email, req.strategy, result["latency_ms"], len(result["chunks"]), result["input_tokens"], result["output_tokens"], result.get("estimated_cost_usd", 0.0))

    response = QueryResponse(
        query_id=log.id,
        query=req.query,
        strategy=req.strategy,
        answer=result["answer"],
        chunks=_to_chunk_scores(result["chunks"]),
        pipeline=[PipelineStep(name=s["name"], detail=s["detail"], duration_ms=s["duration_ms"])
                  for s in result["pipeline"]],
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        latency_ms=result["latency_ms"],
        estimated_cost_usd=result.get("estimated_cost_usd", 0.0),
    )
    # Cache the result for 5 minutes
    set_cache(cache_key, response.model_dump(), ttl_seconds=300)
    return response


@router.post("/query/stream")
async def query_stream_endpoint(req: QueryRequest, db: Session = Depends(get_db), current_user: User = Depends(_get_current_user)):
    """SSE streaming variant: retrieval happens synchronously first (so the
    client can render chunks immediately), then tokens stream as generated.
    The final answer is still persisted to QueryLog once streaming
    completes, so streamed queries show up in history/pipeline lookups
    exactly like non-streamed ones."""
    logger.info("=== [STREAM REQUEST START] ===")
    logger.info("User: %s", current_user.email)
    logger.info("Strategy: %s, Query: %r", req.strategy, req.query)
    
    try:
        tracer = PipelineTracer()
        stream_start = time.perf_counter()
        logger.info("Calling retrieve()...")
        chunks = retrieve(req.strategy, req.query, req.filters, req.top_k_initial,
                           req.top_k_final, req.semantic_weight, req.bm25_weight, tracer, user_id=current_user.id)
        logger.info("retrieve() completed. Found %d chunks.", len(chunks))
        
        logger.info("Calling assemble_context()...")
        context = assemble_context(chunks, max_chunks=req.top_k_final)
        logger.info("Context assembled (length: %d chars).", len(context))
    except Exception as e:
        err_str = str(e)
        logger.exception("CRITICAL ERROR during synchronous retrieval stage of streaming request: %s", e)
        headers = {
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
        }
        return EventSourceResponse(error_generator(), headers=headers)

    if not chunks:
        logger.warning("[%s] Stream query: no chunks matched. strategy=%s query=%r", current_user.email, req.strategy, req.query[:80])
        async def empty_generator():
            yield {"event": "chunks", "data": "[]"}
            notice = "No relevant document chunks found matching the query/filters. Please adjust your query or clear active metadata filters."
            yield {"event": "token", "data": notice}
            
            from app.models.database import SessionLocal
            bg_db = SessionLocal()
            try:
                log = QueryLog(
                    query=req.query, strategy=req.strategy, filters=req.filters.model_dump(),
                    answer=notice, trace=tracer.as_list(), chunks=[],
                    input_tokens=0, output_tokens=0, latency_ms=0.0,
                    user_id=current_user.id,
                )
                bg_db.add(log)
                bg_db.commit()
                bg_db.refresh(log)
                query_id = log.id
            finally:
                bg_db.close()
                
        headers = {
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
        }
        return EventSourceResponse(empty_generator(), headers=headers)

    logger.info("[%s] Stream retrieved %d chunks. Starting LLM generation.", current_user.email, len(chunks))

    async def event_generator():
        logger.info("event_generator: yielding event 'chunks'")
        yield {"event": "chunks", "data": json.dumps([
            {**{k: v for k, v in c.items() if k != "parent_text"}, "token_count": count_tokens(c.get("text", ""))} for c in chunks
        ])}
        full_answer = []
        token_count = 0
        try:
            provider = current_primary_provider()
            logger.info("event_generator: invoking stream_rag_answer with provider='%s'", provider)
            async for token in stream_rag_answer(req.query, context, provider=provider, tracer=tracer):
                token_count += 1
                if token_count % 10 == 0 or token_count < 5:
                    logger.info("event_generator: received token #%d: %r", token_count, token)
                full_answer.append(token)
                yield {"event": "token", "data": token}
            logger.info("event_generator: stream_rag_answer finished. Total tokens: %d", token_count)
        except Exception as e:
            logger.exception("CRITICAL ERROR during async streaming generation stage: %s", e)
            error_msg = f"\n\n[Generation Error: {e}]"
            full_answer.append(error_msg)
            yield {"event": "token", "data": error_msg}

        answer = "".join(full_answer)
        latency_ms = round((time.perf_counter() - stream_start) * 1000, 2)
        
        logger.info("event_generator: persisting QueryLog to DB...")
        from app.models.database import SessionLocal
        bg_db = SessionLocal()
        try:
            log = QueryLog(
                query=req.query, strategy=req.strategy, filters=req.filters.model_dump(),
                answer=answer, trace=tracer.as_list(), chunks=chunks,
                input_tokens=count_tokens(context) + count_tokens(req.query),
                output_tokens=count_tokens(answer), latency_ms=latency_ms,
                user_id=current_user.id,
            )
            bg_db.add(log)
            bg_db.commit()
            bg_db.refresh(log)
            query_id = log.id
            input_tokens = log.input_tokens
            output_tokens = log.output_tokens
            logger.info("event_generator: QueryLog persisted successfully. Log ID: %s", query_id)
        except Exception as db_err:
            logger.exception("Failed to write query log to database: %s", db_err)
            query_id = "error"
            input_tokens = 0
            output_tokens = 0
        finally:
            bg_db.close()

        logger.info("[%s] Stream completed: chunks=%d in_tokens=%d out_tokens=%d latency=%.2fms", current_user.email, len(chunks), input_tokens, output_tokens, latency_ms)

        yield {"event": "done", "data": json.dumps({
            "query_id": query_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
        })}

    headers = {
        "X-Accel-Buffering": "no",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
    }
    return EventSourceResponse(event_generator(), headers=headers)


@router.post("/query/compare", response_model=CompareResponse)
async def compare_endpoint(req: CompareRequest, db: Session = Depends(get_db), current_user: User = Depends(_get_current_user)):
    logger.info("[%s] Compare started: strategies=%s vs %s query=%r", current_user.email, req.strategy_a, req.strategy_b, req.query[:80])
    import asyncio

    async def run_safe(strategy):
        try:
            return await _run_rag_query_bounded(req.query, strategy, req.filters, user_id=current_user.id)
        except Exception as e:
            logger.error("RAG strategy %s execution failed in compare endpoint: %s", strategy, e, exc_info=True)
            return {
                "answer": f"Error running strategy {strategy}: {str(e)}",
                "chunks": [],
                "pipeline": [{"name": "error", "detail": {"error": str(e)}, "duration_ms": 0.0}],
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": 0.0,
                "estimated_cost_usd": 0.0,
                "context": "",
            }

    result_a, result_b = await asyncio.gather(
        run_safe(req.strategy_a),
        run_safe(req.strategy_b),
    )

    def score_quality(result):
        """A/B compare has no ground-truth reference answer (it's an ad hoc
        query, not an eval-dataset run), so only the two metrics that don't
        need one -- faithfulness (answer vs. context) and relevancy
        (answer vs. question) -- are computed here, directly via the judges
        rather than the full 4-metric evaluate_full pass."""
        if not req.score_quality or not result.get("context") or not result.get("answer"):
            return {}
        try:
            faithfulness = evaluate_faithfulness(result["context"], result["answer"])
            relevancy = evaluate_relevancy(req.query, result["answer"])
            return {"faithfulness": faithfulness.get("score", 0.0), "answer_relevancy": relevancy.get("score", 0.0)}
        except Exception as e:
            logger.warning("Score quality failed in compare endpoint: %s", e)
            return {}

    def log_and_wrap(strategy, result):
        quality = score_quality(result)
        log_id = None
        try:
            log = QueryLog(query=req.query, strategy=strategy, filters=req.filters.model_dump(),
                            answer=result["answer"], trace=result["pipeline"], chunks=result["chunks"],
                            input_tokens=result["input_tokens"], output_tokens=result["output_tokens"],
                            latency_ms=result["latency_ms"], user_id=current_user.id)
            db.add(log)
            db.commit()
            db.refresh(log)
            log_id = log.id
        except Exception as e:
            logger.error("Failed to commit QueryLog to database in compare endpoint: %s", e, exc_info=True)
            db.rollback()
            import uuid
            log_id = f"failed-log-{uuid.uuid4().hex[:8]}"

        return QueryResponse(
            query_id=log_id, query=req.query, strategy=strategy, answer=result["answer"],
            chunks=_to_chunk_scores(result["chunks"]),
            pipeline=[PipelineStep(name=s["name"], detail=s["detail"], duration_ms=s["duration_ms"])
                      for s in result["pipeline"]],
            input_tokens=result["input_tokens"], output_tokens=result["output_tokens"],
            latency_ms=result["latency_ms"], estimated_cost_usd=result.get("estimated_cost_usd", 0.0),
            **quality,
        )

    resp_a = log_and_wrap(req.strategy_a, result_a)
    resp_b = log_and_wrap(req.strategy_b, result_b)

    ids_a = {c.chunk_id for c in resp_a.chunks}
    ids_b = {c.chunk_id for c in resp_b.chunks}
    overlap = list(ids_a & ids_b)

    logger.info("[%s] Compare completed: overlap_chunks=%d", current_user.email, len(overlap))
    return CompareResponse(query=req.query, result_a=resp_a, result_b=resp_b, overlap_chunk_ids=overlap)


@router.get("/query/{query_id}/pipeline")
def get_pipeline(query_id: str, db: Session = Depends(get_db), current_user: User = Depends(_get_current_user)):
    log = db.query(QueryLog).filter(QueryLog.id == query_id, QueryLog.user_id == current_user.id).first()
    if not log:
        raise HTTPException(404, "Query not found")
    logger.info("[%s] Get pipeline: query_id=%s steps=%d", current_user.email, query_id, len(log.trace or []))
    return {"query_id": log.id, "pipeline": log.trace}


@router.get("/query/{query_id}/chunks")
def get_query_chunks(query_id: str, db: Session = Depends(get_db), current_user: User = Depends(_get_current_user)):
    log = db.query(QueryLog).filter(QueryLog.id == query_id, QueryLog.user_id == current_user.id).first()
    if not log:
        raise HTTPException(404, "Query not found")
    logger.info("[%s] Get query chunks: query_id=%s chunks=%d", current_user.email, query_id, len(log.chunks or []))
    return {"query_id": log.id, "chunks": _to_chunk_scores(log.chunks or [])}


@router.get("/chunks/search")
def debug_chunk_search(q: str, strategy: str = "recursive", top_k: int = 10,
                       current_user: User = Depends(_get_current_user)):
    """Direct chunk search debug endpoint (no LLM call, just retrieval)."""
    logger.info("[%s] Debug chunk search: q=%r strategy=%s top_k=%d", current_user.email, q, strategy, top_k)
    return {"results": basic_vector_search(q, top_k=top_k, strategy_tag=strategy)}


@router.get("/query/history", response_model=list[QueryLogOut])
def get_query_history(limit: int = 20, db: Session = Depends(get_db), current_user: User = Depends(_get_current_user)):
    """Return the most recent query logs (latest first), including chunks and
    pipeline traces so the frontend can restore full result state from history."""
    logs = db.query(QueryLog).filter(QueryLog.user_id == current_user.id).order_by(QueryLog.created_at.desc()).limit(limit).all()
    logger.info("[%s] Get history: limit=%d results=%d", current_user.email, limit, len(logs))
    return [QueryLogOut.model_validate(log) for log in logs]
