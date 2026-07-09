"""
Implements every chunking strategy required by the assignment. All
strategies consume the same list of LangChain `Document`s (already
metadata-enriched by ingestion.py) and return chunks tagged with
`metadata["strategy"]` so they can be filtered / A-B compared later.

Strategies:
  - recursive     -> RecursiveCharacterTextSplitter (500 tok / 50 overlap)
  - semantic      -> embedding-similarity based splitting
  - parent_child  -> small (200 tok) chunks that reference a parent (1000 tok) chunk
  - section       -> one chunk per detected H1/H2 section
"""
import logging
import uuid
from typing import List, Dict, Tuple

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.llm_factory import get_embeddings

logger = logging.getLogger("chunking_service")


def _new_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------- recursive
def recursive_chunks(docs: List[LCDocument], chunk_size: int = 500 * 4,
                      chunk_overlap: int = 50 * 4) -> List[LCDocument]:
    """chunk_size/overlap given in characters (~4 chars/token heuristic)."""
    uid = docs[0].metadata.get("user_id", "-") if docs else "-"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    out = []
    for d in docs:
        for chunk in splitter.split_documents([d]):
            chunk.metadata = {**chunk.metadata, "strategy": "recursive", "chunk_id": _new_id()}
            out.append(chunk)
    logger.info("[uid:%s] Recursive chunking: %d docs -> %d chunks", uid, len(docs), len(out))
    return out


# ----------------------------------------------------------------- semantic
def semantic_chunks(docs: List[LCDocument], breakpoint_threshold_amount: float = 72.0,
                     max_chunk_chars: int = 2500) -> List[LCDocument]:
    """
    Uses LangChain's real `SemanticChunker` (langchain_experimental) instead
    of a hand-rolled sentence-similarity loop: it embeds each sentence and
    starts a new chunk wherever the semantic distance between consecutive
    sentences crosses a percentile-based breakpoint, producing semantically
    coherent chunks without us reimplementing that logic ourselves.
    """
    uid = docs[0].metadata.get("user_id", "-") if docs else "-"
    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError:
        logger.warning("[uid:%s] SemanticChunker unavailable, falling back to recursive", uid)
        return recursive_chunks(docs)

    embeddings = get_embeddings()
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )

    out = []
    valid_docs = [d for d in docs if d.page_content.strip()]
    if not valid_docs:
        return []

    # Cap to first 40 pages to avoid O(n) embedding calls on large docs.
    # Semantic chunking embeds every sentence — beyond 40 pages it becomes
    # prohibitively slow on CPU. The remaining pages still get indexed via
    # the recursive strategy which runs on all pages.
    if len(valid_docs) > 40:
        logger.warning(
            "[uid:%s] Semantic chunking: capping from %d to 40 pages for speed.",
            uid, len(valid_docs),
        )
        valid_docs = valid_docs[:40]

    texts = [d.page_content for d in valid_docs]
    metadatas = [dict(d.metadata) for d in valid_docs]

    try:
        sub_docs = splitter.create_documents(texts, metadatas=metadatas)
    except Exception:
        # Fall back if the chunker fails on batch input
        sub_docs = []
        for d in valid_docs:
            try:
                sub_docs.extend(splitter.create_documents([d.page_content], metadatas=[d.metadata]))
            except Exception:
                sub_docs.append(LCDocument(page_content=d.page_content, metadata=dict(d.metadata)))

    for sub in sub_docs:
        # Keep track of the original page's metadata structure
        orig_metadata = dict(sub.metadata)
        if len(sub.page_content) > max_chunk_chars:
            sub_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_chunk_chars, chunk_overlap=0
            )
            for piece in sub_splitter.split_text(sub.page_content):
                out.append(LCDocument(
                    page_content=piece,
                    metadata={**orig_metadata, "strategy": "semantic", "chunk_id": _new_id()},
                ))
        else:
            out.append(LCDocument(
                page_content=sub.page_content,
                metadata={**orig_metadata, "strategy": "semantic", "chunk_id": _new_id()},
            ))
    logger.info("[uid:%s] Semantic chunking: %d docs -> %d chunks", uid, len(valid_docs), len(out))
    return out


# ------------------------------------------------------------- parent-child
def parent_child_chunks(docs: List[LCDocument], parent_size: int = 1000 * 4,
                         child_size: int = 200 * 4) -> Tuple[List[LCDocument], Dict[str, LCDocument]]:
    """
    Returns (child_chunks_for_indexing, parent_lookup)
    Child chunks carry metadata['parent_id'] pointing into `parent_lookup`,
    which maps parent_id -> full parent LCDocument (larger context window).
    Retrieval matches against *children*; the parent is what gets sent to
    the LLM, per ParentDocumentRetriever semantics.
    """
    uid = docs[0].metadata.get("user_id", "-") if docs else "-"
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=0)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_size, chunk_overlap=40)

    children_out: List[LCDocument] = []
    parent_lookup: Dict[str, LCDocument] = {}

    for d in docs:
        parents = parent_splitter.split_documents([d])
        for parent in parents:
            parent_id = _new_id()
            parent.metadata = {**parent.metadata, "strategy": "parent_child", "parent_id": parent_id}
            parent_lookup[parent_id] = parent

            children = child_splitter.split_documents([parent])
            for child in children:
                child.metadata = {
                    **child.metadata,
                    "strategy": "parent_child",
                    "parent_id": parent_id,
                    "chunk_id": _new_id(),
                }
                children_out.append(child)

    logger.info("[uid:%s] Parent-child chunking: %d docs -> %d children, %d parents", uid, len(docs), len(children_out), len(parent_lookup))
    return children_out, parent_lookup


# ------------------------------------------------------------------ section
def section_chunks(docs: List[LCDocument]) -> List[LCDocument]:
    """
    Groups consecutive source Documents/pages that share the same detected
    `section` metadata (set during ingestion) into a single chunk per
    section. Best for structured docs (reports, manuals, legal).
    """
    uid = docs[0].metadata.get("user_id", "-") if docs else "-"
    out = []
    buckets: Dict[str, List[LCDocument]] = {}
    order: List[str] = []

    for d in docs:
        section = d.metadata.get("section", "Untitled Section")
        key = f"{d.metadata.get('doc_id')}::{section}"
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(d)

    for key in order:
        parts = buckets[key]
        combined_text = "\n\n".join(p.page_content for p in parts)
        base_meta = dict(parts[0].metadata)
        base_meta.update({
            "strategy": "section",
            "chunk_id": _new_id(),
            "page": parts[0].metadata.get("page"),
        })
        out.append(LCDocument(page_content=combined_text, metadata=base_meta))

    logger.info("[uid:%s] Section chunking: %d docs -> %d section chunks", uid, len(docs), len(out))
    return out


# ---------------------------------------------------------- multi-vector
def multi_vector_docs(docs: List[LCDocument]) -> List[LCDocument]:
    """
    Multi-vector retrieval per the assignment spec.
    Generates summary + hypothetical questions per chunk so retrieval is
    driven by richer semantic representations than raw chunk text.
    Skips immediately (raises RuntimeError) when LLM is unavailable/rate-limited
    so ingestion is never blocked.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from app.services.llm_factory import get_llm

    uid = docs[0].metadata.get("user_id", "-") if docs else "-"

    # ── Fast LLM availability probe ──────────────────────────────────────────
    # Try a tiny call before spawning threads. If ALL providers are rate-limited
    # this fails in ~2-3 s instead of blocking for 30 s per chunk.
    try:
        get_llm().invoke("ping")
    except Exception as probe_err:
        logger.warning(
            "[uid:%s] multi_vector: LLM unavailable (%s) — skipping strategy.",
            uid, str(probe_err)[:120],
        )
        raise RuntimeError(f"multi_vector: LLM probe failed: {probe_err}")

    logger.info("[uid:%s] Multi-vector chunking: starting for %d docs", uid, len(docs))

    SUMMARY_PROMPT = ChatPromptTemplate.from_template(
        "Write a concise 1-2 sentence summary of this passage, capturing its "
        "key facts and topics. Be specific -- include numbers, names, and "
        "technical terms if present.\n\nPassage:\n{text}"
    )
    QUESTIONS_PROMPT = ChatPromptTemplate.from_template(
        "Generate exactly 3 distinct questions that this passage would "
        "directly answer. Return ONE question per line with no numbering, "
        "bullets, or extra commentary.\n\nPassage:\n{text}"
    )

    # Build two LCEL chains -- prompt | llm | parser, no legacy classes
    summary_chain = SUMMARY_PROMPT | get_llm() | StrOutputParser()
    questions_chain = QUESTIONS_PROMPT | get_llm() | StrOutputParser()

    base_chunks = recursive_chunks(docs)  # start from clean, sized units
    out: List[LCDocument] = []

    from concurrent.futures import ThreadPoolExecutor

    def process_chunk(chunk: LCDocument) -> List[LCDocument]:
        original_text = chunk.page_content
        chunk_id = chunk.metadata.get("chunk_id", _new_id())
        truncated = original_text[:1500]  # stay within cheap-model context

        # --- generate summary ---
        try:
            summary = summary_chain.invoke({"text": truncated}).strip()
        except Exception:
            summary = original_text[:300]  # fallback: first 300 chars

        # --- generate hypothetical questions ---
        try:
            raw_qs = questions_chain.invoke({"text": truncated})
            questions = [
                q.strip("-•* \t")
                for q in raw_qs.strip().split("\n")
                if q.strip() and len(q.strip()) > 10
            ][:3]
        except Exception:
            questions = []

        base_meta = {
            **chunk.metadata,
            "strategy": "multi_vector",
            "parent_text": original_text,
        }

        chunk_docs = []
        # One document for the summary
        chunk_docs.append(LCDocument(
            page_content=summary,
            metadata={
                **base_meta,
                "multi_vector_type": "summary",
                "chunk_id": f"mv_sum_{chunk_id}",
            },
        ))

        # One document per generated question
        for i, q in enumerate(questions):
            chunk_docs.append(LCDocument(
                page_content=q,
                metadata={
                    **base_meta,
                    "multi_vector_type": "question",
                    "chunk_id": f"mv_q{i}_{chunk_id}",
                },
            ))
        return chunk_docs

    import concurrent.futures
    with ThreadPoolExecutor(max_workers=4) as executor:
        try:
            chunk_results = list(executor.map(process_chunk, base_chunks, timeout=60))
        except concurrent.futures.TimeoutError:
            logger.warning("[uid:%s] multi_vector: timed out after 60s — skipping", uid)
            raise RuntimeError("multi_vector timed out")

    for item in chunk_results:
        out.extend(item)

    logger.info("[uid:%s] Multi-vector chunking complete: %d docs -> %d multi-vector chunks", uid, len(docs), len(out))
    return out


def run_all_strategies(docs: List[LCDocument]) -> Dict[str, List[LCDocument]]:
    """Runs every chunking strategy on the same source docs simultaneously."""
    from app.services.parent_child_retriever import persist_parents

    uid = docs[0].metadata.get("user_id", "-") if docs else "-"
    logger.info("[uid:%s] Running all chunking strategies on %d docs", uid, len(docs))

    children, parents = parent_child_chunks(docs)
    persist_parents(parents)

    results: Dict[str, List[LCDocument]] = {
        "recursive": recursive_chunks(docs),
        "semantic": semantic_chunks(docs),
        "parent_child": children,
        "section": section_chunks(docs),
    }

    # multi_vector requires LLM calls (summaries + questions) — skip gracefully
    # if all LLM providers are rate-limited or unavailable so ingestion completes.
    try:
        mv = multi_vector_docs(docs)
        results["multi_vector"] = mv
    except Exception as e:
        logger.warning(
            "[uid:%s] multi_vector strategy skipped (LLM unavailable: %s). "
            "The document is still fully indexed via all other strategies.",
            uid, e,
        )
        results["multi_vector"] = []

    logger.info("[uid:%s] All strategies complete: %s", uid, {k: len(v) for k, v in results.items()})
    return results

