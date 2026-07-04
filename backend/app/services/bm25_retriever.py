from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from app.core.logging import get_logger
import asyncio

logger = get_logger(__name__)


class BM25Retriever:
    def __init__(self, documents: list[Document] = None):
        self.documents = documents or []
        self.bm25 = None
        self.tokenized_corpus = []
        if documents:
            self._build_index(documents)

    def _build_index(self, documents: list[Document]):
        self.documents = documents
        self.tokenized_corpus = [doc.page_content.split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"BM25 index built with {len(documents)} documents")

    def search(self, query: str, k: int = 5) -> list[Document]:
        if not self.bm25:
            logger.warning("BM25 index not built")
            return []

        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_k_indices = ranked_indices[:k]

        results = []
        for idx in top_k_indices:
            doc = self.documents[idx]
            doc.metadata["bm25_score"] = float(scores[idx])
            results.append(doc)

        logger.info(f"BM25 search returned {len(results)} results")
        return results


_bm25_instance = None


def get_bm25_retriever():
    global _bm25_instance
    if _bm25_instance is None:
        _bm25_instance = BM25Retriever()
    return _bm25_instance


def build_bm25_index(documents: list[Document]):
    global _bm25_instance
    _bm25_instance = BM25Retriever(documents)
    return _bm25_instance


async def bm25_search_async(query: str, k: int = 5) -> list[Document]:
    retriever = get_bm25_retriever()
    return await asyncio.to_thread(retriever.search, query, k)


bm25_retrieval_chain = RunnableLambda(lambda x: bm25_search_async(x["query"], x.get("k", 5)))
