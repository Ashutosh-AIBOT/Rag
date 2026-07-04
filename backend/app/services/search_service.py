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


def search_documents(
    query: str,
    k: int = 5,
    strategy: str = "vector",
    filters: dict = None,
    rerank: bool = False,
    rerank_top_k: int = 3,
    vectorstore = None,
) -> dict:
    strategy = _normalize_strategy(strategy)
    trace = {
        "original_query": query,
        "strategy": strategy,
        "transformed_queries": [],
        "retrieved_chunks": [],
        "reranked_chunks": [],
    }

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

    return {
        "documents": docs,
        "trace": trace,
    }
