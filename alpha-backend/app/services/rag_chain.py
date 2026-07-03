import time
import uuid
import asyncio
import json
from typing import List, Dict, Any, Tuple, AsyncGenerator
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.core.logging import get_logger
from app.vectorstore.chroma import get_chroma_db
from app.prompts import RAG_SYSTEM_PROMPT
from app.services.retrieval import get_bm25_retriever, rrf_merge, filter_documents_by_metadata
from app.services.reranker import rerank_documents
from app.services.query_transform import (
    generate_multi_queries,
    generate_hyde_answer,
    decompose_query,
    generate_step_back_query
)
from app.database.database import save_query_log, save_pipeline_trace

logger = get_logger(__name__)

def swap_parent_child_context(docs: List[Document]) -> List[Document]:
    """
    If retrieved chunks are child chunks with parent metadata,
    de-duplicates them by parent_id and swaps child content with parent_text from SQLite.
    """
    from app.database.database import get_parent_document_text
    seen_parents = set()
    swapped_docs = []
    for doc in docs:
        parent_id = doc.metadata.get("parent_id")
        if parent_id:
            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                parent_text = get_parent_document_text(parent_id)
                if parent_text:
                    # Create a new document with parent content but preserving original metadata
                    new_doc = Document(page_content=parent_text, metadata=doc.metadata)
                    swapped_docs.append(new_doc)
                else:
                    swapped_docs.append(doc)
        else:
            swapped_docs.append(doc)
    return swapped_docs


async def retrieve_dense(query: str, app, k: int = 20) -> List[Document]:
    """Retrieve documents using dense vector similarity search."""
    logger.info(f"Retrieving dense vector matches for query: {query}")
    try:
        db = get_chroma_db(app.state.embeddings)
        results_with_scores = db.similarity_search_with_relevance_scores(query, k=k)
        
        docs = []
        for doc, score in results_with_scores:
            doc.metadata["similarity_score"] = float(score)
            docs.append(doc)
        return docs
    except Exception as e:
        logger.error(f"Dense retrieval failed: {e}", exc_info=True)
        return []

async def retrieve_sparse(query: str, app, k: int = 20) -> List[Document]:
    """Retrieve documents using sparse BM25 keyword search."""
    logger.info(f"Retrieving sparse BM25 matches for query: {query}")
    bm25 = get_bm25_retriever(app)
    if bm25 is None:
        return []
    try:
        # Offload synchronous BM25 keyword matching to a worker thread
        bm25.k = k
        docs = await asyncio.to_thread(bm25.invoke, query)
        for i, doc in enumerate(docs):
            doc.metadata["bm25_score"] = float(len(docs) - i) / len(docs) # Pseudo score
        return docs
    except Exception as e:
        logger.error(f"Sparse retrieval failed: {e}", exc_info=True)
        return []

async def _retrieve_and_build_prompt(
    query: str,
    strategy: str,
    filters: Dict[str, Any],
    app,
    trace: Dict[str, Any]
) -> Tuple[List[Document], str]:
    """
    Private helper to run the retrieval, re-ranking, metadata filtering,
    and prompt construction stages of the RAG pipeline.
    """
    # 1. Fetch LLM from app state
    llm_manager = getattr(app.state, "llm_manager", None)
    if llm_manager is None:
        raise RuntimeError("LLM Manager is not loaded.")
    llm = llm_manager.get_llm()

    initial_docs = []
    final_docs = []
    
    # 2. Strategy-based Retrieval and Query Transformation
    if strategy == "dense":
        initial_docs = await retrieve_dense(query, app, k=settings.RETRIEVAL_TOP_K)
        initial_docs = filter_documents_by_metadata(initial_docs, filters)
        trace["initial_retrieval"] = [
            {"content": doc.page_content, "metadata": doc.metadata, "score": doc.metadata.get("similarity_score", 0.0)}
            for doc in initial_docs
        ]
        final_docs = initial_docs[:settings.FINAL_TOP_K]

    elif strategy == "sparse":
        initial_docs = await retrieve_sparse(query, app, k=settings.RETRIEVAL_TOP_K)
        initial_docs = filter_documents_by_metadata(initial_docs, filters)
        trace["initial_retrieval"] = [
            {"content": doc.page_content, "metadata": doc.metadata, "score": doc.metadata.get("bm25_score", 0.0)}
            for doc in initial_docs
        ]
        final_docs = initial_docs[:settings.FINAL_TOP_K]

    elif strategy == "hybrid":
        dense = await retrieve_dense(query, app, k=settings.RETRIEVAL_TOP_K)
        sparse = await retrieve_sparse(query, app, k=settings.RETRIEVAL_TOP_K)
        dense = filter_documents_by_metadata(dense, filters)
        sparse = filter_documents_by_metadata(sparse, filters)
        
        initial_docs = rrf_merge(dense, sparse, top_n=settings.RETRIEVAL_TOP_K)
        trace["initial_retrieval"] = [
            {"content": doc.page_content, "metadata": doc.metadata, "score": doc.metadata.get("rrf_score", 0.0)}
            for doc in initial_docs
        ]
        final_docs = initial_docs[:settings.FINAL_TOP_K]

    elif strategy == "hybrid_rerank":
        dense = await retrieve_dense(query, app, k=settings.RETRIEVAL_TOP_K)
        sparse = await retrieve_sparse(query, app, k=settings.RETRIEVAL_TOP_K)
        dense = filter_documents_by_metadata(dense, filters)
        sparse = filter_documents_by_metadata(sparse, filters)
        
        initial_docs = rrf_merge(dense, sparse, top_n=settings.RETRIEVAL_TOP_K)
        trace["initial_retrieval"] = [
            {"content": doc.page_content, "metadata": doc.metadata, "score": doc.metadata.get("rrf_score", 0.0)}
            for doc in initial_docs
        ]
        final_docs = await rerank_documents(query, initial_docs, app, top_n=settings.FINAL_TOP_K)
        trace["re_ranked_retrieval"] = [
            {"content": doc.page_content, "metadata": doc.metadata, "score": doc.metadata.get("rerank_score", 0.0)}
            for doc in final_docs
        ]

    elif strategy == "multiquery_hybrid_rerank":
        queries = await generate_multi_queries(query, llm)
        trace["transformations"]["multi_queries"] = queries
        
        all_dense = []
        all_sparse = []
        for q in queries:
            all_dense.extend(await retrieve_dense(q, app, k=10))
            all_sparse.extend(await retrieve_sparse(q, app, k=10))
            
        all_dense = filter_documents_by_metadata(all_dense, filters)
        all_sparse = filter_documents_by_metadata(all_sparse, filters)
        
        initial_docs = rrf_merge(all_dense, all_sparse, top_n=settings.RETRIEVAL_TOP_K)
        trace["initial_retrieval"] = [
            {"content": doc.page_content, "metadata": doc.metadata, "score": doc.metadata.get("rrf_score", 0.0)}
            for doc in initial_docs
        ]
        final_docs = await rerank_documents(query, initial_docs, app, top_n=settings.FINAL_TOP_K)
        trace["re_ranked_retrieval"] = [
            {"content": doc.page_content, "metadata": doc.metadata, "score": doc.metadata.get("rerank_score", 0.0)}
            for doc in final_docs
        ]

    elif strategy == "hyde_hybrid_rerank":
        hyde_answer = await generate_hyde_answer(query, llm)
        trace["transformations"]["hyde_answer"] = hyde_answer
        
        dense = await retrieve_dense(hyde_answer, app, k=settings.RETRIEVAL_TOP_K)
        sparse = await retrieve_sparse(hyde_answer, app, k=settings.RETRIEVAL_TOP_K)
        dense = filter_documents_by_metadata(dense, filters)
        sparse = filter_documents_by_metadata(sparse, filters)
        
        initial_docs = rrf_merge(dense, sparse, top_n=settings.RETRIEVAL_TOP_K)
        trace["initial_retrieval"] = [
            {"content": doc.page_content, "metadata": doc.metadata, "score": doc.metadata.get("rrf_score", 0.0)}
            for doc in initial_docs
        ]
        final_docs = await rerank_documents(query, initial_docs, app, top_n=settings.FINAL_TOP_K)
        trace["re_ranked_retrieval"] = [
            {"content": doc.page_content, "metadata": doc.metadata, "score": doc.metadata.get("rerank_score", 0.0)}
            for doc in final_docs
        ]

    elif strategy == "decomposed_hybrid_rerank":
        sub_queries = await decompose_query(query, llm)
        trace["transformations"]["decomposed_queries"] = sub_queries
        
        all_dense = []
        all_sparse = []
        for q in sub_queries:
            all_dense.extend(await retrieve_dense(q, app, k=10))
            all_sparse.extend(await retrieve_sparse(q, app, k=10))
            
        all_dense = filter_documents_by_metadata(all_dense, filters)
        all_sparse = filter_documents_by_metadata(all_sparse, filters)
        
        initial_docs = rrf_merge(all_dense, all_sparse, top_n=settings.RETRIEVAL_TOP_K)
        trace["initial_retrieval"] = [
            {"content": doc.page_content, "metadata": doc.metadata, "score": doc.metadata.get("rrf_score", 0.0)}
            for doc in initial_docs
        ]
        final_docs = await rerank_documents(query, initial_docs, app, top_n=settings.FINAL_TOP_K)
        trace["re_ranked_retrieval"] = [
            {"content": doc.page_content, "metadata": doc.metadata, "score": doc.metadata.get("rerank_score", 0.0)}
            for doc in final_docs
        ]

    elif strategy == "step_back_hybrid_rerank":
        step_back_query = await generate_step_back_query(query, llm)
        trace["transformations"]["step_back_query"] = step_back_query
        
        dense_orig = await retrieve_dense(query, app, k=settings.RETRIEVAL_TOP_K)
        sparse_orig = await retrieve_sparse(query, app, k=settings.RETRIEVAL_TOP_K)
        dense_step = await retrieve_dense(step_back_query, app, k=settings.RETRIEVAL_TOP_K)
        sparse_step = await retrieve_sparse(step_back_query, app, k=settings.RETRIEVAL_TOP_K)
        
        dense_orig = filter_documents_by_metadata(dense_orig, filters)
        sparse_orig = filter_documents_by_metadata(sparse_orig, filters)
        dense_step = filter_documents_by_metadata(dense_step, filters)
        sparse_step = filter_documents_by_metadata(sparse_step, filters)
        
        initial_docs = rrf_merge(dense_orig + dense_step, sparse_orig + sparse_step, top_n=settings.RETRIEVAL_TOP_K)
        trace["initial_retrieval"] = [
            {"content": doc.page_content, "metadata": doc.metadata, "score": doc.metadata.get("rrf_score", 0.0)}
            for doc in initial_docs
        ]
        final_docs = await rerank_documents(query, initial_docs, app, top_n=settings.FINAL_TOP_K)
        trace["re_ranked_retrieval"] = [
            {"content": doc.page_content, "metadata": doc.metadata, "score": doc.metadata.get("rerank_score", 0.0)}
            for doc in final_docs
        ]
        
    else:
        raise ValueError(f"Unknown retrieval strategy: {strategy}")

    # 3. Swap child chunks with parent content if using parent-child chunking
    final_docs = swap_parent_child_context(final_docs)

    # 4. Assemble Context
    context_str = "\n\n".join([
        f"--- Source: {doc.metadata.get('filename')} (Page {doc.metadata.get('page', 'N/A')}) ---\n{doc.page_content}"
        for doc in final_docs
    ])
    trace["context_assembled"] = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in final_docs
    ]

    # 5. Construct Final Prompt
    prompt_template = ChatPromptTemplate.from_template(RAG_SYSTEM_PROMPT)
    formatted_prompt = prompt_template.format(context=context_str, question=query)
    trace["prompt"] = formatted_prompt

    return final_docs, context_str

async def execute_rag_pipeline(
    query: str,
    strategy: str,
    filters: Dict[str, Any],
    app
) -> Dict[str, Any]:
    """
    Executes the full RAG pipeline based on the requested strategy.
    Tracks all stages, execution times, and outputs to build a transparent execution trace.
    """
    start_time = time.time()
    trace = {
        "original_query": query,
        "strategy": strategy,
        "filters": filters,
        "transformations": {},
        "initial_retrieval": [],
        "re_ranked_retrieval": [],
        "context_assembled": [],
        "prompt": "",
        "token_usage": {"input": 0, "output": 0, "total": 0},
        "latency_ms": 0
    }

    # Retrieve documents and build context prompt
    final_docs, context_str = await _retrieve_and_build_prompt(query, strategy, filters, app, trace)

    # LLM generation setup
    llm = app.state.llm_manager.get_llm()
    prompt_template = ChatPromptTemplate.from_template(RAG_SYSTEM_PROMPT)

    # LLM Call (wrapped with Semaphore to protect API limits)
    async with app.state.llm_semaphore:
        logger.info("Executing final answer generation...")
        chain = prompt_template | llm | StrOutputParser()
        answer = await chain.ainvoke({"context": context_str, "question": query})

    # Estimate tokens
    input_tokens = len(trace["prompt"].split())
    output_tokens = len(answer.split())
    trace["token_usage"] = {
        "input": input_tokens,
        "output": output_tokens,
        "total": input_tokens + output_tokens
    }

    trace["latency_ms"] = int((time.time() - start_time) * 1000)
    logger.info(f"RAG Pipeline execution completed in {trace['latency_ms']} ms.")
    
    # Save query log to database
    query_id = str(uuid.uuid4())
    save_query_log(
        query_id=query_id,
        query_text=query,
        strategy=strategy,
        filters=filters,
        answer=answer,
        latency_ms=trace["latency_ms"],
        token_count=trace["token_usage"]["total"]
    )
    
    # Save trace steps
    save_pipeline_trace(str(uuid.uuid4()), query_id, "original_query", query, 0)
    if trace["transformations"]:
        save_pipeline_trace(str(uuid.uuid4()), query_id, "transformations", trace["transformations"], 0)
    if trace["initial_retrieval"]:
        save_pipeline_trace(str(uuid.uuid4()), query_id, "initial_retrieval", trace["initial_retrieval"], 0)
    if trace["re_ranked_retrieval"]:
        save_pipeline_trace(str(uuid.uuid4()), query_id, "re_ranked_retrieval", trace["re_ranked_retrieval"], 0)
    save_pipeline_trace(str(uuid.uuid4()), query_id, "context_assembled", trace["context_assembled"], 0)
    save_pipeline_trace(str(uuid.uuid4()), query_id, "prompt", trace["prompt"], 0)

    return {
        "answer": answer,
        "trace": trace
    }

async def execute_rag_pipeline_stream(
    query: str,
    strategy: str,
    filters: Dict[str, Any],
    app
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Executes the RAG pipeline and yields Server-Sent Events (SSE) compatible dicts.
    Yields:
      1. 'trace': Detailed execution trace (metadata, chunks, transformations, prompt).
      2. 'token': Incremental LLM response tokens.
      3. 'done': Done indicator with final database logged query_id and execution stats.
    """
    start_time = time.time()
    trace = {
        "original_query": query,
        "strategy": strategy,
        "filters": filters,
        "transformations": {},
        "initial_retrieval": [],
        "re_ranked_retrieval": [],
        "context_assembled": [],
        "prompt": "",
        "token_usage": {"input": 0, "output": 0, "total": 0},
        "latency_ms": 0
    }

    try:
        # Retrieve documents and build context prompt
        final_docs, context_str = await _retrieve_and_build_prompt(query, strategy, filters, app, trace)

        # Yield the trace dictionary immediately so the frontend knows what was retrieved
        yield {
            "event": "trace",
            "data": json.dumps(trace)
        }

        # LLM generation setup
        llm = app.state.llm_manager.get_llm()
        prompt_template = ChatPromptTemplate.from_template(RAG_SYSTEM_PROMPT)
        chain = prompt_template | llm | StrOutputParser()

        full_answer = []
        
        # Stream the LLM Call (wrapped with Semaphore to protect API limits)
        async with app.state.llm_semaphore:
            logger.info("Executing streaming answer generation...")
            async for token in chain.astream({"context": context_str, "question": query}):
                full_answer.append(token)
                yield {
                    "event": "token",
                    "data": token
                }

        answer = "".join(full_answer)

        # Estimate tokens and latency
        input_tokens = len(trace["prompt"].split())
        output_tokens = len(answer.split())
        trace["token_usage"] = {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens
        }

        trace["latency_ms"] = int((time.time() - start_time) * 1000)
        logger.info(f"RAG streaming pipeline completed in {trace['latency_ms']} ms.")

        # Save query log to database
        query_id = str(uuid.uuid4())
        save_query_log(
            query_id=query_id,
            query_text=query,
            strategy=strategy,
            filters=filters,
            answer=answer,
            latency_ms=trace["latency_ms"],
            token_count=trace["token_usage"]["total"]
        )
        
        # Save trace steps
        save_pipeline_trace(str(uuid.uuid4()), query_id, "original_query", query, 0)
        if trace["transformations"]:
            save_pipeline_trace(str(uuid.uuid4()), query_id, "transformations", trace["transformations"], 0)
        if trace["initial_retrieval"]:
            save_pipeline_trace(str(uuid.uuid4()), query_id, "initial_retrieval", trace["initial_retrieval"], 0)
        if trace["re_ranked_retrieval"]:
            save_pipeline_trace(str(uuid.uuid4()), query_id, "re_ranked_retrieval", trace["re_ranked_retrieval"], 0)
        save_pipeline_trace(str(uuid.uuid4()), query_id, "context_assembled", trace["context_assembled"], 0)
        save_pipeline_trace(str(uuid.uuid4()), query_id, "prompt", trace["prompt"], 0)

        # Yield completion data
        yield {
            "event": "done",
            "data": json.dumps({
                "query_id": query_id,
                "latency_ms": trace["latency_ms"],
                "token_usage": trace["token_usage"]
            })
        }

    except Exception as e:
        logger.error(f"Error in RAG streaming pipeline: {e}", exc_info=True)
        yield {
            "event": "error",
            "data": str(e)
        }
