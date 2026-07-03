from typing import Dict, Any, Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.core.logging import get_logger
from app.services.rag_chain import execute_rag_pipeline, execute_rag_pipeline_stream

logger = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])

class QueryRequest(BaseModel):
    query: str = Field(..., example="What is attention in Transformer models?")
    strategy: str = Field(default="hybrid_rerank", description="One of: dense, sparse, hybrid, hybrid_rerank, multiquery_hybrid_rerank, hyde_hybrid_rerank, decomposed_hybrid_rerank, step_back_hybrid_rerank")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters (e.g. {'doc_id': '...', 'tags': '...', 'chunk_strategy': '...'})")
    stream: bool = Field(default=False, description="Set to true to stream the response via Server-Sent Events (SSE)")

class CompareRequest(BaseModel):
    query: str = Field(..., example="Explain self-attention")
    strategy_a: str = Field(..., description="First retrieval/RAG strategy")
    strategy_b: str = Field(..., description="Second retrieval/RAG strategy")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")

@router.post("")
async def query_rag(request: Request, payload: QueryRequest):
    """
    Executes a single RAG pipeline run with the specified retrieval strategy and filters.
    If stream=True is passed, returns a Server-Sent Events (SSE) streaming token flow.
    Otherwise, returns the final answer and a detailed execution trace for frontend transparency.
    """
    logger.info(f"Query request received: '{payload.query}' (Strategy: {payload.strategy}, Stream: {payload.stream})")
    
    valid_strategies = [
        "dense", "sparse", "hybrid", "hybrid_rerank",
        "multiquery_hybrid_rerank", "hyde_hybrid_rerank", "decomposed_hybrid_rerank",
        "step_back_hybrid_rerank"
    ]
    if payload.strategy not in valid_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy '{payload.strategy}'. Valid: {', '.join(valid_strategies)}"
        )

    try:
        filters = payload.filters or {}
        
        if payload.stream:
            logger.info("Initializing SSE streaming flow for query...")
            event_generator = execute_rag_pipeline_stream(
                query=payload.query,
                strategy=payload.strategy,
                filters=filters,
                app=request.app
            )
            return EventSourceResponse(event_generator)
            
        result = await execute_rag_pipeline(
            query=payload.query,
            strategy=payload.strategy,
            filters=filters,
            app=request.app
        )
        return result
    except Exception as e:
        logger.error(f"Error during query execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare")
async def compare_rag(request: Request, payload: CompareRequest):
    """
    Executes two different RAG pipelines side-by-side for comparison.
    Returns trace and generation results for both Strategy A and Strategy B.
    """
    logger.info(f"Compare request received: '{payload.query}' ({payload.strategy_a} vs {payload.strategy_b})")
    
    valid_strategies = [
        "dense", "sparse", "hybrid", "hybrid_rerank",
        "multiquery_hybrid_rerank", "hyde_hybrid_rerank", "decomposed_hybrid_rerank",
        "step_back_hybrid_rerank"
    ]
    if payload.strategy_a not in valid_strategies or payload.strategy_b not in valid_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategies. Valid: {', '.join(valid_strategies)}"
        )

    try:
        filters = payload.filters or {}
        
        # Execute Strategy A
        result_a = await execute_rag_pipeline(
            query=payload.query,
            strategy=payload.strategy_a,
            filters=filters,
            app=request.app
        )
        
        # Execute Strategy B
        result_b = await execute_rag_pipeline(
            query=payload.query,
            strategy=payload.strategy_b,
            filters=filters,
            app=request.app
        )
        
        return {
            "query": payload.query,
            "strategy_a": {
                "strategy": payload.strategy_a,
                "answer": result_a["answer"],
                "trace": result_a["trace"]
            },
            "strategy_b": {
                "strategy": payload.strategy_b,
                "answer": result_b["answer"],
                "trace": result_b["trace"]
            }
        }
    except Exception as e:
        logger.error(f"Error during comparison query execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{query_id}/pipeline")
async def get_query_pipeline(query_id: str):
    logger.info(f"Fetching pipeline trace for query: {query_id}")
    from app.database.database import get_pipeline_trace_steps
    steps = get_pipeline_trace_steps(query_id)
    if not steps:
        raise HTTPException(status_code=404, detail=f"No pipeline trace found for query ID '{query_id}'")
    return steps

@router.get("/{query_id}/chunks")
async def get_query_chunks(query_id: str):
    logger.info(f"Fetching retrieved chunks for query: {query_id}")
    from app.database.database import get_pipeline_trace_steps
    steps = get_pipeline_trace_steps(query_id)
    if not steps:
        raise HTTPException(status_code=404, detail=f"No query history or chunks found for query ID '{query_id}'")
    
    # Extract initial_retrieval or re_ranked_retrieval chunks
    retrieved_chunks = []
    for step in steps:
        if step["step_name"] in ["re_ranked_retrieval", "initial_retrieval"]:
            retrieved_chunks = step["step_data"]
            # If we have re-ranked, it's more relevant, so prefer it
            if step["step_name"] == "re_ranked_retrieval":
                break
    return retrieved_chunks

@router.get("/history")
async def get_history_queries():
    logger.info("Retrieving past queries history.")
    from app.database.database import get_query_history
    return get_query_history()



# Common Root API Router for prefixless endpoints like /api/strategies, /api/stats, /api/chunks/search
from fastapi import Query
from app.services.rag_chain import retrieve_dense, retrieve_sparse
from app.services.retrieval import rrf_merge, filter_documents_by_metadata

common_router = APIRouter(tags=["Common"])

@common_router.get("/strategies")
async def get_strategies():
    logger.info("Fetching list of retrieval/RAG strategies")
    return [
        {"id": "dense", "name": "Dense Similarity Search Only"},
        {"id": "sparse", "name": "Sparse Keyword (BM25) Search Only"},
        {"id": "hybrid", "name": "Hybrid (Dense + Sparse with RRF)"},
        {"id": "hybrid_rerank", "name": "Hybrid + Re-ranking (Cross-Encoder)"},
        {"id": "multiquery_hybrid_rerank", "name": "Multi-Query Expansion + Hybrid + Re-ranking"},
        {"id": "hyde_hybrid_rerank", "name": "HyDE (Hypothetical Doc Embeddings) + Hybrid + Re-ranking"},
        {"id": "decomposed_hybrid_rerank", "name": "Query Decomposition + Hybrid + Re-ranking"},
        {"id": "step_back_hybrid_rerank", "name": "Step-Back Prompting + Hybrid + Re-ranking"}
    ]

@common_router.get("/stats")
async def get_stats():
    logger.info("Calculating platform statistics summary")
    from app.database.database import get_stats_summary
    return get_stats_summary()

@common_router.get("/chunks/search")
async def search_chunks(
    request: Request,
    query: str = Query(..., description="The query string to search for"),
    strategy: str = Query("dense", description="Retrieval strategy: dense, sparse, hybrid"),
    k: int = Query(5, description="Number of chunks to return"),
    doc_id: Optional[str] = Query(None, description="Filter by Document ID"),
    chunk_strategy: Optional[str] = Query(None, description="Filter by chunk strategy (e.g. recursive, parent_child)"),
    tags: Optional[str] = Query(None, description="Filter by tag (comma-separated)")
):
    logger.info(f"Direct chunk search query: '{query}' (Strategy: {strategy}, k: {k})")
    valid_strategies = ["dense", "sparse", "hybrid"]
    if strategy not in valid_strategies:
        raise HTTPException(status_code=400, detail=f"Invalid strategy. Valid: {', '.join(valid_strategies)}")
    
    filters = {}
    if doc_id:
        filters["doc_id"] = doc_id
    if chunk_strategy:
        filters["chunk_strategy"] = chunk_strategy
    if tags:
        filters["tags"] = tags
        
    try:
        if strategy == "dense":
            docs = await retrieve_dense(query, request.app, k=k)
            docs = filter_documents_by_metadata(docs, filters)
        elif strategy == "sparse":
            docs = await retrieve_sparse(query, request.app, k=k)
            docs = filter_documents_by_metadata(docs, filters)
        else: # hybrid
            dense = await retrieve_dense(query, request.app, k=k)
            sparse = await retrieve_sparse(query, request.app, k=k)
            dense = filter_documents_by_metadata(dense, filters)
            sparse = filter_documents_by_metadata(sparse, filters)
            docs = rrf_merge(dense, sparse, top_n=k)
            
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": doc.metadata.get("similarity_score") or doc.metadata.get("bm25_score") or doc.metadata.get("rrf_score") or 0.0
            }
            for doc in docs
        ]
    except Exception as e:
        logger.error(f"Error in direct chunk search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

