from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from app.core.logging import get_logger
import asyncio

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    results_list: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    fused_scores = {}
    doc_map = {}

    for results in results_list:
        for rank, doc in enumerate(results):
            doc_id = doc.metadata.get("id", doc.page_content[:100])
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                doc_map[doc_id] = doc
            fused_scores[doc_id] += 1 / (k + rank + 1)

    sorted_doc_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    result = []
    for doc_id in sorted_doc_ids:
        doc = doc_map[doc_id]
        doc.metadata["rrf_score"] = fused_scores[doc_id]
        result.append(doc)

    logger.info(f"RRF returned {len(result)} results")
    return result


class RRFRetriever:
    def __init__(self, retrievers: list, k: int = 60):
        self.retrievers = retrievers
        self.k = k

    def search(self, query: str, n: int = 5) -> list[Document]:
        all_results = []
        for retriever in self.retrievers:
            if hasattr(retriever, 'search'):
                results = retriever.search(query, k=n)
            else:
                results = retriever.invoke(query)
            all_results.append(results)

        fused = reciprocal_rank_fusion(all_results, self.k)
        return fused[:n]


_rrf_instance = None


def get_rrf_retriever(retrievers: list, k: int = 60):
    global _rrf_instance
    if _rrf_instance is None:
        _rrf_instance = RRFRetriever(retrievers, k)
    return _rrf_instance


async def rrf_search_async(query: str, n: int = 5) -> list[Document]:
    retriever = get_rrf_retriever([])
    return await asyncio.to_thread(retriever.search, query, n)


rrf_retrieval_chain = RunnableLambda(lambda x: rrf_search_async(x["query"], x.get("n", 5)))
