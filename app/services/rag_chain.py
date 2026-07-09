"""
The heart of the system: given a strategy name, runs the full retrieval ->
(optional rerank) -> context assembly -> LCEL generation pipeline, and
returns everything needed for both the answer AND full pipeline
transparency (chunks w/ scores, timings, assembled prompt, token counts).

LCEL is used throughout for the generation step:
    prompt | llm | StrOutputParser()
Retrieval strategy selection happens *before* the LCEL chain (since it
determines which chunks become the `context` variable), which keeps the
generation chain itself simple, declarative and reusable across strategies.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
import re
import time

import tiktoken
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document as LCDocument

logger = logging.getLogger("rag_chain_service")
from app.services.llm_factory import get_llm, get_embeddings, estimate_cost_usd
from app.services.retrieval import (
    basic_vector_search, hybrid_search, parent_child_search,
    multi_vector_search, section_search,
)
from app.services.reranker import rerank
from app.services.query_transform import hyde_generate, decompose_query, step_back_query
from app.services.pipeline_tracer import PipelineTracer

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise, factual assistant. Answer the user's question using "
     "ONLY the information in the provided context. If the context does not "
     "contain enough information to answer, say so explicitly instead of "
     "guessing. Cite the source document name in parentheses after claims "
     "where relevant.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text or ""))


def assemble_context(chunks: List[Dict[str, Any]], max_chunks: int = 5) -> str:
    parts = []
    for i, c in enumerate(chunks[:max_chunks], start=1):
        parts.append(f"[Source {i}: {c['source']} | page {c.get('page')}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def context_token_breakdown(chunks: List[Dict[str, Any]], max_chunks: int = 5) -> List[Dict[str, Any]]:
    """Per-source-chunk token counts, so the UI can show exactly how the
    context-window budget is spent across sources (assignment explicitly
    asks for this, previously missing entirely)."""
    breakdown = []
    for i, c in enumerate(chunks[:max_chunks], start=1):
        breakdown.append({
            "source": c["source"],
            "page": c.get("page"),
            "chunk_id": c.get("chunk_id"),
            "tokens": count_tokens(c["text"]),
        })
    return breakdown


def _dedupe(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for c in chunks:
        if c["chunk_id"] in seen:
            continue
        seen.add(c["chunk_id"])
        out.append(c)
    return out


def compress_chunks(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Bonus: contextual compression. Uses LangChain's real `LLMChainExtractor`
    to extract only the sentences of each retrieved chunk that are actually
    relevant to the query, shrinking what gets sent to the generation LLM."""
    if not chunks:
        return chunks
    try:
        from langchain.retrievers.document_compressors import LLMChainExtractor
    except ImportError:
        try:
            from langchain_community.retrievers.document_compressors import LLMChainExtractor
        except ImportError:
            logger.warning("LLMChainExtractor not available; skipping compression.")
            return chunks

    docs = [LCDocument(page_content=c["text"], metadata={"chunk_id": c["chunk_id"]}) for c in chunks]
    compressor = LLMChainExtractor.from_llm(get_llm())
    try:
        compressed = compressor.compress_documents(docs, query)
    except Exception:
        return chunks

    compressed_by_id = {d.metadata.get("chunk_id"): d.page_content for d in compressed if d.page_content.strip()}
    out = []
    for c in chunks:
        new_text = compressed_by_id.get(c["chunk_id"])
        if new_text:
            merged = dict(c)
            merged["text"] = new_text
            merged["compressed"] = True
            out.append(merged)
    return out or chunks  # never fully empty the context if the compressor drops everything


def _classify_query_for_auto(query: str) -> str:
    """Bonus: lightweight auto-strategy heuristic. Not an ML classifier --
    just query-shape signals that are cheap to check before spending an
    LLM/retrieval call."""
    q = query.strip()
    if re.search(r"\b[A-Z]{1,4}-?\d{2,6}\b", q) or re.search(r"\d{3,}", q):
        return "hybrid_rerank"  # exact codes/numbers -> lexical matters, keep BM25 in the mix
    if re.search(r"\b(compare|vs\.?|versus|difference between)\b", q, re.IGNORECASE):
        return "decomposition"
    if re.search(r"\b(what was|revenue|q[1-4]\s*20\d{2}|specific)\b", q, re.IGNORECASE):
        return "step_back"
    if len(q.split()) <= 6:
        return "hyde"  # short/abstract queries benefit from HyDE's embedding trick
    return "hybrid_rerank"


def retrieve(strategy: str, query: str, filters, top_k_initial: int, top_k_final: int,
             semantic_weight: float, bm25_weight: float,
             tracer: PipelineTracer, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Executes retrieval (+ optional rerank/query-transform) for the given
    strategy and returns the final list of scored chunk records."""

    if strategy == "auto":
        with tracer.step("auto_strategy_selection") as rec:
            chosen = _classify_query_for_auto(query)
            rec["detail"]["chosen_strategy"] = chosen
        return retrieve(chosen, query, filters, top_k_initial, top_k_final,
                         semantic_weight, bm25_weight, tracer, user_id=user_id)

    if strategy == "basic_vector":
        with tracer.step("query_transformation") as rec:
            rec["detail"]["type"] = "none"
        with tracer.step("retrieval") as rec:
            chunks = basic_vector_search(query, top_k=top_k_final, filters=filters, user_id=user_id)
            rec["detail"]["strategy"] = strategy
            rec["detail"]["chunks_found"] = len(chunks)
        return chunks

    if strategy == "hybrid":
        with tracer.step("query_transformation") as rec:
            rec["detail"]["type"] = "none"
        with tracer.step("retrieval") as rec:
            chunks = hybrid_search(query, top_k=top_k_final, filters=filters,
                                    semantic_weight=semantic_weight, bm25_weight=bm25_weight, user_id=user_id)
            rec["detail"]["strategy"] = strategy
            rec["detail"]["fusion"] = "RRF"
            rec["detail"]["k"] = 60
            rec["detail"]["chunks_found"] = len(chunks)
        return chunks

    if strategy == "hybrid_rerank":
        with tracer.step("query_transformation") as rec:
            rec["detail"]["type"] = "none"
        with tracer.step("retrieval") as rec:
            initial = hybrid_search(query, top_k=top_k_initial, filters=filters,
                                     semantic_weight=semantic_weight, bm25_weight=bm25_weight, user_id=user_id)
            rec["detail"]["strategy"] = "hybrid"
            rec["detail"]["fusion"] = "RRF"
            rec["detail"]["k"] = 60
            rec["detail"]["chunks_found"] = len(initial)
        with tracer.step("reranking") as rec:
            reranked = rerank(query, initial, top_k=top_k_final)
            rec["detail"]["model"] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            rec["detail"]["before"] = len(initial)
            rec["detail"]["after"] = len(reranked)
        return reranked

    if strategy == "parent_child":
        with tracer.step("query_transformation") as rec:
            rec["detail"]["type"] = "none"
        with tracer.step("retrieval") as rec:
            chunks = parent_child_search(query, top_k=top_k_final, filters=filters, user_id=user_id)
            rec["detail"]["strategy"] = strategy
            rec["detail"]["note"] = "Searches small child chunks, returns parent chunk text for LLM context"
            rec["detail"]["chunks_found"] = len(chunks)
        return chunks

    if strategy == "multi_query":
        from app.services.query_transform import multi_query_expand
        with tracer.step("query_transformation") as rec:
            variants = multi_query_expand(query)
            rec["detail"]["type"] = "multi_query"
            rec["detail"]["variants"] = variants
            rec["detail"]["variant_count"] = len(variants)
        with tracer.step("retrieval") as rec:
            all_results = []
            with ThreadPoolExecutor(max_workers=min(len(variants), 4)) as executor:
                futures = {
                    executor.submit(
                        hybrid_search, var, top_k_initial, filters, "recursive",
                        semantic_weight, bm25_weight, user_id
                    ): var for var in variants
                }
                for future in as_completed(futures):
                    try:
                        all_results.extend(future.result())
                    except Exception as e:
                        logger.warning("[uid:%s] Multi-query search failed for variant %r: %s", user_id or "-", futures[future], e)
            merged = _dedupe(sorted(all_results, key=lambda c: c.get("rrf_score", 0), reverse=True))
            rec["detail"]["strategy"] = "hybrid_per_variant"
            rec["detail"]["total_results"] = len(all_results)
            rec["detail"]["chunks_found"] = len(merged)
        with tracer.step("reranking") as rec:
            reranked = rerank(query, merged, top_k=top_k_final)
            rec["detail"]["model"] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            rec["detail"]["before"] = len(merged)
            rec["detail"]["after"] = len(reranked)
        return reranked

    if strategy == "hyde":
        with tracer.step("query_transformation") as rec:
            hypothetical = hyde_generate(query)
            rec["detail"]["type"] = "hyde"
            rec["detail"]["hypothetical_answer"] = hypothetical
        with tracer.step("retrieval") as rec:
            chunks = hybrid_search(hypothetical, top_k=top_k_initial, filters=filters,
                                    semantic_weight=semantic_weight, bm25_weight=bm25_weight, user_id=user_id)
            rec["detail"]["strategy"] = "hybrid_on_hypothetical"
            rec["detail"]["chunks_found"] = len(chunks)
        with tracer.step("reranking") as rec:
            reranked = rerank(query, chunks, top_k=top_k_final)
            rec["detail"]["model"] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            rec["detail"]["before"] = len(chunks)
            rec["detail"]["after"] = len(reranked)
        return reranked

    if strategy == "decomposition":
        with tracer.step("query_transformation") as rec:
            sub_qs = decompose_query(query)
            rec["detail"]["type"] = "decomposition"
            rec["detail"]["sub_questions"] = sub_qs
            rec["detail"]["sub_question_count"] = len(sub_qs)
        with tracer.step("retrieval") as rec:
            per_sub_k = max(3, top_k_initial // len(sub_qs))
            all_results: List[Dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=min(len(sub_qs), 4)) as executor:
                futures = {
                    executor.submit(
                        hybrid_search, sq, per_sub_k, filters, "recursive",
                        semantic_weight, bm25_weight, user_id
                    ): sq for sq in sub_qs
                }
                for future in as_completed(futures):
                    try:
                        all_results.extend(future.result())
                    except Exception as e:
                        logger.warning("[uid:%s] Sub-query retrieval failed for %r: %s", user_id or "-", futures[future], e)
            merged = _dedupe(sorted(all_results, key=lambda c: c.get("rrf_score", 0), reverse=True))
            rec["detail"]["strategy"] = "hybrid_per_subquestion"
            rec["detail"]["total_results"] = len(all_results)
            rec["detail"]["chunks_found"] = len(merged)
        with tracer.step("reranking") as rec:
            reranked = rerank(query, merged, top_k=top_k_final)
            rec["detail"]["model"] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            rec["detail"]["before"] = len(merged)
            rec["detail"]["after"] = len(reranked)
        return reranked

    if strategy == "step_back":
        with tracer.step("query_transformation") as rec:
            step_back_q = step_back_query(query)
            rec["detail"]["type"] = "step_back"
            rec["detail"]["original_query"] = query
            rec["detail"]["step_back_question"] = step_back_q
        with tracer.step("retrieval") as rec:
            half_k = max(3, top_k_initial // 2)
            with ThreadPoolExecutor(max_workers=2) as executor:
                specific_future = executor.submit(
                    hybrid_search, query, half_k, filters, "recursive",
                    semantic_weight, bm25_weight, user_id
                )
                broad_future = executor.submit(
                    hybrid_search, step_back_q, half_k, filters, "recursive",
                    semantic_weight, bm25_weight, user_id
                )
                specific = specific_future.result()
                broad = broad_future.result()
            merged = _dedupe(sorted(specific + broad, key=lambda c: c.get("rrf_score", 0), reverse=True))
            rec["detail"]["strategy"] = "hybrid_original_and_stepback"
            rec["detail"]["chunks_from_original"] = len(specific)
            rec["detail"]["chunks_from_stepback"] = len(broad)
            rec["detail"]["chunks_found"] = len(merged)
        with tracer.step("reranking") as rec:
            reranked = rerank(query, merged, top_k=top_k_final)
            rec["detail"]["model"] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            rec["detail"]["before"] = len(merged)
            rec["detail"]["after"] = len(reranked)
        return reranked

    if strategy == "multi_vector":
        with tracer.step("query_transformation") as rec:
            rec["detail"]["type"] = "none"
        with tracer.step("retrieval") as rec:
            chunks = multi_vector_search(query, top_k=top_k_final, filters=filters, user_id=user_id)
            rec["detail"]["strategy"] = strategy
            rec["detail"]["note"] = (
                "Searched summaries + hypothetical-questions embeddings; "
                "returning original chunk text to LLM."
            )
            rec["detail"]["chunks_found"] = len(chunks)
        return chunks

    if strategy == "section_search":
        with tracer.step("query_transformation") as rec:
            rec["detail"]["type"] = "none"
        with tracer.step("retrieval") as rec:
            chunks = section_search(query, top_k=top_k_final, filters=filters, user_id=user_id)
            rec["detail"]["strategy"] = strategy
            rec["detail"]["chunks_found"] = len(chunks)
        return chunks

    raise ValueError(f"Unknown strategy: {strategy}")


def run_rag_query(query: str, strategy: str, filters=None, top_k_initial: int = 20,
                   top_k_final: int = 5, semantic_weight: float = 0.7,
                   bm25_weight: float = 0.3, compress_context: bool = False,
                   user_id: Optional[str] = None) -> Dict[str, Any]:
    """Full end-to-end pipeline for one query. Returns everything the API /
    frontend needs: answer, chunks, pipeline trace, token counts, latency."""
    logger.info("[uid:%s] RAG pipeline started: strategy=%s query=%r", user_id or "-", strategy, query[:80])
    tracer = PipelineTracer()
    start = time.perf_counter()

    chunks = retrieve(strategy, query, filters, top_k_initial, top_k_final,
                       semantic_weight, bm25_weight, tracer, user_id=user_id)

    if not chunks:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.warning("[uid:%s] RAG pipeline: 0 chunks retrieved in %.2fms. strategy=%s", user_id or "-", latency_ms, strategy)
        return {
            "answer": "No relevant document chunks found matching your query/filters. Please clear metadata filters or upload documents.",
            "chunks": [],
            "pipeline": tracer.as_list(),
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": latency_ms,
            "estimated_cost_usd": 0.0,
            "context": "",
        }

    logger.info("[uid:%s] RAG retrieved %d chunks. Proceeding with generation.", user_id or "-", len(chunks))

    if compress_context:
        with tracer.step("contextual_compression") as rec:
            before = len(chunks)
            chunks = compress_chunks(query, chunks)
            rec["detail"]["chunks_before"] = before
            rec["detail"]["chunks_after"] = len(chunks)
            logger.info("[uid:%s] Context compression: %d -> %d chunks", user_id or "-", before, len(chunks))

    with tracer.step("context_assembly") as rec:
        context = assemble_context(chunks, max_chunks=top_k_final)
        rec["detail"]["context_tokens"] = count_tokens(context)
        rec["detail"]["per_source_tokens"] = context_token_breakdown(chunks, max_chunks=top_k_final)

    logger.info("[uid:%s] Context assembled: %d tokens. Invoking LLM.", user_id or "-", count_tokens(context))

    with tracer.step("llm_generation") as rec:
        chain = RAG_PROMPT | get_llm() | StrOutputParser()
        prompt_value = RAG_PROMPT.format(context=context, question=query)
        answer = chain.invoke({"context": context, "question": query})
        rec["detail"]["prompt_preview"] = prompt_value[:4000]
        rec["detail"]["output_tokens"] = count_tokens(answer)

    input_tokens = count_tokens(context) + count_tokens(query)
    output_tokens = count_tokens(answer)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    estimated_cost_usd = estimate_cost_usd(input_tokens, output_tokens)

    logger.info("[uid:%s] RAG pipeline completed: latency=%.2fms answer_len=%d in_tokens=%d out_tokens=%d",
                user_id or "-", latency_ms, len(answer), input_tokens, output_tokens)

    return {
        "answer": answer,
        "chunks": chunks,
        "pipeline": tracer.as_list(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "estimated_cost_usd": estimated_cost_usd,
        "context": context,
    }


async def stream_rag_answer(query: str, context: str, provider: str = None,
                             tracer: 'PipelineTracer' = None):
    """Async generator yielding answer tokens for SSE streaming, used after
    retrieval has already produced `context` (kept separate so retrieval
    tracing above stays synchronous and simple).

    When a PipelineTracer is passed, records context_assembly and
    llm_generation steps so the pipeline visualizer shows the complete
    pipeline (not just retrieval) for streamed queries."""
    if tracer:
        with tracer.step("context_assembly") as rec:
            rec["detail"]["context_tokens"] = count_tokens(context)
            rec["detail"]["context_length_chars"] = len(context)

    chain = RAG_PROMPT | get_llm(provider=provider) | StrOutputParser()

    if tracer:
        with tracer.step("llm_generation") as rec:
            prompt_value = RAG_PROMPT.format(context=context, question=query)
            rec["detail"]["prompt_preview"] = prompt_value[:4000]
            full_answer = []
            async for token in chain.astream({"context": context, "question": query}):
                full_answer.append(token)
                yield token
            rec["detail"]["output_tokens"] = count_tokens("".join(full_answer))
    else:
        async for token in chain.astream({"context": context, "question": query}):
            yield token
