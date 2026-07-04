from langchain_core.documents import Document
from app.core.logging import get_logger
from app.services.retrieval import retrieval_service
from app.services.bm25_retriever import get_bm25_retriever
from app.services.hybrid_retriever import get_hybrid_retriever
from app.services.reranker import get_reranker
from app.services.hyde import build_hyde_chain
from app.services.step_back import build_step_back_chain
from app.services.decomposition import build_decomposition_chain, parse_sub_queries
from app.llm import get_llm_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import asyncio

logger = get_logger(__name__)

MULTI_QUERY_PROMPT_TEMPLATE = """You are an AI assistant tasked with generating three different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.

Original question: {question}

Provide these alternative questions separated by newlines. Do not output anything else.
Alternative questions:"""


def build_multi_query_chain():
    prompt = ChatPromptTemplate.from_template(MULTI_QUERY_PROMPT_TEMPLATE)
    return prompt | get_llm_chain() | StrOutputParser()


def deduplicate_documents(docs: list[Document]) -> list[Document]:
    seen = set()
    deduped = []
    for doc in docs:
        # Deduplicate based on page_content
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            deduped.append(doc)
    return deduped


def _normalize_strategy(strategy: str) -> str:
    """Normalize strategy names: accept both hyphens and underscores."""
    return strategy.replace("_", "-")


def classify_query_strategy(query: str) -> str:
    query_lower = query.lower()
    if "compare" in query_lower or "versus" in query_lower or "vs" in query_lower or "difference" in query_lower:
        return "hybrid-rerank"
    if "why" in query_lower or "how" in query_lower or "explain" in query_lower:
        return "decomposition"
    if "concept" in query_lower or "define" in query_lower or "what is" in query_lower:
        return "hyde"
    return "hybrid"


def classify_query_strategy_llm(query: str) -> str:
    try:
        prompt = ChatPromptTemplate.from_template(
            "You are a query router for a RAG system. Classify the user query into one of these strategies:\n"
            "1. 'hybrid-rerank' (for complex, technical, or comparison questions)\n"
            "2. 'decomposition' (for multi-part or multi-step questions)\n"
            "3. 'hyde' (for definitions, concepts, or broad background questions)\n"
            "4. 'hybrid' (for simple search or fact lookup)\n\n"
            "Query: {query}\n\n"
            "Return ONLY the exact strategy name as a lowercase string (e.g. 'hybrid-rerank')."
        )
        chain = prompt | get_llm_chain() | StrOutputParser()
        result = chain.invoke({"query": query}).strip().lower()
        if result in ("hybrid-rerank", "decomposition", "hyde", "hybrid"):
            return result
    except Exception as e:
        logger.error(f"LLM strategy routing failed: {e}")
    return classify_query_strategy(query)


def compress_document_chunk(query: str, doc_content: str) -> str:
    try:
        from app.llm import get_llm_chain
        prompt = ChatPromptTemplate.from_template(
            "You are a contextual compressor. Extract ONLY the sentences or fragments from the context that are directly relevant to answering the query. Do not rewrite, synthesize, or explain. Keep the extracted text verbatim.\n"
            "If no part of the context is relevant, return an empty string.\n\n"
            "Query: {query}\n"
            "Context: {context}\n\n"
            "Extracted relevant text:"
        )
        chain = prompt | get_llm_chain() | StrOutputParser()
        result = chain.invoke({"query": query, "context": doc_content}).strip()
        if result:
            return result
    except Exception as e:
        logger.error(f"Contextual compression failed for chunk: {e}")
    return doc_content


def search_documents(
    query: str,
    k: int = 5,
    strategy: str = "vector",
    filters: dict = None,
    rerank: bool = False,
    rerank_top_k: int = 3,
    vectorstore = None,
    compress: bool = False,
) -> dict:
    strategy = _normalize_strategy(strategy)
    trace = {
        "original_query": query,
        "strategy": strategy,
        "transformed_queries": [],
        "retrieved_chunks": [],
        "reranked_chunks": [],
    }

    if strategy == "auto":
        actual_strategy = classify_query_strategy_llm(query)
        logger.info(f"Auto-routed strategy to: {actual_strategy}")
        trace["auto_selected_strategy"] = actual_strategy
        strategy = actual_strategy

    docs = []

    if strategy == "vector":
        docs = retrieval_service.invoke({"query": query, "k": k, "filters": filters, "vectorstore": vectorstore})

    elif strategy == "bm25":
        bm25 = get_bm25_retriever()
        docs = bm25.search(query, k=k)

    elif strategy == "hybrid":
        hybrid = get_hybrid_retriever(vectorstore)
        docs = hybrid.search(query, k=k)

    elif strategy == "hybrid-rerank":
        hybrid = get_hybrid_retriever(vectorstore)
        docs = hybrid.search(query, k=max(k, 20))
        rerank = True

    elif strategy == "parent-child":
        # Force filter to parent-child strategy
        pc_filters = dict(filters or {})
        pc_filters["strategy"] = "parent-child"
        docs = retrieval_service.invoke({"query": query, "k": k, "filters": pc_filters, "vectorstore": vectorstore})

    elif strategy in ("multi-query", "multi-query-expansion"):
        chain = build_multi_query_chain()
        output = chain.invoke({"question": query})
        phrasings = [q.strip() for q in output.strip().split("\n") if q.strip()]
        trace["transformed_queries"] = phrasings
        
        all_docs = []
        # Retrieve for original and alternative queries
        for q in [query] + phrasings:
            all_docs.extend(retrieval_service.invoke({"query": q, "k": k, "filters": filters, "vectorstore": vectorstore}))
        docs = deduplicate_documents(all_docs)[:k]

    elif strategy == "hyde":
        chain = build_hyde_chain()
        hypothetical_doc = chain.invoke({"question": query})
        trace["transformed_queries"] = [hypothetical_doc]
        
        docs = retrieval_service.invoke({"query": hypothetical_doc, "k": k, "filters": filters, "vectorstore": vectorstore})

    elif strategy in ("step-back", "step-back-prompting"):
        chain = build_step_back_chain()
        step_back_query = chain.invoke({"question": query})
        trace["transformed_queries"] = [step_back_query]
        
        docs_specific = retrieval_service.invoke({"query": query, "k": k, "filters": filters, "vectorstore": vectorstore})
        docs_broad = retrieval_service.invoke({"query": step_back_query, "k": k, "filters": filters, "vectorstore": vectorstore})
        docs = deduplicate_documents(docs_specific + docs_broad)[:k]

    elif strategy == "decomposition":
        chain = build_decomposition_chain()
        output = chain.invoke({"question": query})
        sub_queries = parse_sub_queries(output)
        if not sub_queries:
            sub_queries = [query]
        trace["transformed_queries"] = sub_queries
        
        all_docs = []
        for sq in sub_queries:
            all_docs.extend(retrieval_service.invoke({"query": sq, "k": k, "filters": filters, "vectorstore": vectorstore}))
        docs = deduplicate_documents(all_docs)[:k]

    else:
        logger.warning(f"Unknown strategy {strategy}, falling back to vector")
        docs = retrieval_service.invoke({"query": query, "k": k, "filters": filters, "vectorstore": vectorstore})

    # Prepare retrieved chunks list for trace
    trace["retrieved_chunks"] = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]

    # Re-ranking
    if rerank and docs:
        reranker = get_reranker()
        reranked = reranker.rerank(query, docs, top_k=rerank_top_k)
        trace["reranked_chunks"] = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in reranked
        ]
        docs = reranked

    # Contextual Compression
    if compress and docs:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=5) as executor:
            compressed_contents = list(executor.map(lambda d: compress_document_chunk(query, d.page_content), docs))
        
        compressed_docs = []
        for doc, comp_content in zip(docs, compressed_contents):
            if comp_content.strip():
                # Update content with compressed content
                from langchain_core.documents import Document as LCDocument
                new_doc = LCDocument(page_content=comp_content.strip(), metadata=doc.metadata)
                compressed_docs.append(new_doc)
            else:
                # If compression removed it completely, fall back to original chunk rather than skipping completely
                compressed_docs.append(doc)
        
        if compressed_docs:
            docs = compressed_docs
        
        trace["compressed_chunks"] = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in docs
        ]

    return {
        "documents": docs,
        "trace": trace,
    }


from langchain_core.runnables import RunnableLambda
search_service = RunnableLambda(
    lambda x: search_documents(
        query=x["query"],
        k=x.get("k", 5),
        strategy=x.get("strategy", "vector"),
        filters=x.get("filters"),
        rerank=x.get("rerank", False),
        rerank_top_k=x.get("rerank_top_k", 3),
        vectorstore=x.get("vectorstore"),
        compress=x.get("compress", False)
    )
)
