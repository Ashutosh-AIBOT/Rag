"""Hybrid retriever using EnsembleRetriever pattern (vector + BM25)
with Reciprocal Rank Fusion via configurable weights.

Note: In langchain >= 1.0, EnsembleRetriever moved out of the main langchain package.
We implement the identical algorithm (weighted RRF) directly using langchain_core primitives,
following the same API as langchain.retrievers.EnsembleRetriever.
"""
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from app.services.bm25_retriever import get_bm25_retriever, BM25Retriever
from app.services.rrf import reciprocal_rank_fusion
from app.core.logging import get_logger
import asyncio

logger = get_logger(__name__)


class HybridRetriever:
    """EnsembleRetriever-compatible hybrid retriever (vector + BM25).

    Combines semantic vector search with BM25 keyword search using
    weighted Reciprocal Rank Fusion, exactly matching the algorithm
    from langchain.retrievers.EnsembleRetriever.
    """

    def __init__(self, vectorstore=None, bm25_retriever=None, weights: list[float] = None):
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        self.weights = weights or [0.5, 0.5]

    def _get_vector_results(self, query: str, k: int, filters: dict = None) -> list[Document]:
        if self.vectorstore is None:
            return []
        from app.services.retrieval import retrieval_service
        return retrieval_service.invoke({"query": query, "k": k, "filters": filters, "vectorstore": self.vectorstore})

    def _get_bm25_results(self, query: str, k: int, filters: dict = None) -> list[Document]:
        if self.bm25_retriever is None or self.bm25_retriever.bm25 is None:
            return []
        return self.bm25_retriever.search(query, k=k, filters=filters)

    def _weighted_rrf(
        self, result_lists: list[list[Document]], weights: list[float], k: int = 60
    ) -> list[Document]:
        """Weighted Reciprocal Rank Fusion — identical to EnsembleRetriever algorithm.
        
        score(doc) = sum_over_retrievers( weight_i / (k + rank_i) )
        """
        fused_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for results, weight in zip(result_lists, weights):
            for rank, doc in enumerate(results):
                doc_key = doc.page_content[:200]
                if doc_key not in fused_scores:
                    fused_scores[doc_key] = 0.0
                    doc_map[doc_key] = doc
                fused_scores[doc_key] += weight / (k + rank + 1)

        sorted_keys = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        result = []
        for doc_key in sorted_keys:
            doc = doc_map[doc_key]
            doc.metadata["rrf_score"] = fused_scores[doc_key]
            result.append(doc)

        return result

    def search(self, query: str, k: int = 5, filters: dict = None) -> list[Document]:
        """Execute hybrid search with weighted RRF fusion."""
        result_lists = []
        active_weights = []

        vector_results = self._get_vector_results(query, k, filters)
        if vector_results:
            result_lists.append(vector_results)
            active_weights.append(self.weights[0] if len(self.weights) > 0 else 0.5)

        bm25_results = self._get_bm25_results(query, k, filters)
        if bm25_results:
            result_lists.append(bm25_results)
            active_weights.append(self.weights[1] if len(self.weights) > 1 else 0.5)

        if not result_lists:
            logger.warning("No retrievers available for hybrid search")
            return []

        if len(result_lists) == 1:
            return result_lists[0][:k]

        # Weighted RRF (same algorithm as LangChain EnsembleRetriever)
        fused = self._weighted_rrf(result_lists, active_weights, k=60)
        logger.info(f"Hybrid search (EnsembleRetriever pattern) returned {len(fused)} results")
        return fused[:k]


_hybrid_instance = None


def get_hybrid_retriever(vectorstore=None):
    global _hybrid_instance
    if _hybrid_instance is None:
        bm25 = get_bm25_retriever()
        _hybrid_instance = HybridRetriever(vectorstore=vectorstore, bm25_retriever=bm25)
    return _hybrid_instance


def build_hybrid_retriever(vectorstore, documents: list[Document] = None, weights: list[float] = None):
    global _hybrid_instance
    from app.services.bm25_retriever import build_bm25_index
    if documents:
        bm25_retriever = build_bm25_index(documents)
    else:
        bm25_retriever = get_bm25_retriever()

    _hybrid_instance = HybridRetriever(vectorstore=vectorstore, bm25_retriever=bm25_retriever, weights=weights)
    return _hybrid_instance


async def hybrid_search_async(query: str, k: int = 5, filters: dict = None) -> list[Document]:
    retriever = get_hybrid_retriever()
    return await asyncio.to_thread(retriever.search, query, k, filters)


hybrid_retrieval_chain = RunnableLambda(lambda x: hybrid_search_async(x["query"], x.get("k", 5), x.get("filters")))
