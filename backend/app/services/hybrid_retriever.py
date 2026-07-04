from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from app.services.bm25_retriever import get_bm25_retriever, build_bm25_index
from app.services.rrf import reciprocal_rank_fusion
from app.core.logging import get_logger
import asyncio

logger = get_logger(__name__)


class HybridRetriever:
    def __init__(self, vectorstore=None, bm25_retriever=None, weights: list[float] = None):
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        self.weights = weights or [0.5, 0.5]

    def search(self, query: str, k: int = 5) -> list[Document]:
        all_results = []

        if self.vectorstore:
            vector_results = self.vectorstore.similarity_search(query, k=k)
            all_results.append(vector_results)

        if self.bm25_retriever and self.bm25_retriever.bm25:
            bm25_results = self.bm25_retriever.search(query, k=k)
            all_results.append(bm25_results)

        if not all_results:
            logger.warning("No retrievers available")
            return []

        if len(all_results) == 1:
            return all_results[0][:k]

        fused = reciprocal_rank_fusion(all_results, k=60)
        logger.info(f"Hybrid search returned {len(fused)} results")
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
    if documents:
        bm25_retriever = build_bm25_index(documents)
    else:
        bm25_retriever = get_bm25_retriever()

    _hybrid_instance = HybridRetriever(vectorstore=vectorstore, bm25_retriever=bm25_retriever, weights=weights)
    return _hybrid_instance


async def hybrid_search_async(query: str, k: int = 5) -> list[Document]:
    retriever = get_hybrid_retriever()
    return await asyncio.to_thread(retriever.search, query, k)


hybrid_retrieval_chain = RunnableLambda(lambda x: hybrid_search_async(x["query"], x.get("k", 5)))
