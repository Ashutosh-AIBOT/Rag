import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from app.core.logging import get_logger
import asyncio

logger = get_logger(__name__)

INDEX_FILE = Path("uploads/bm25_index.pkl")


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
        self.save()

    def search(self, query: str, k: int = 5, filters: dict = None) -> list[Document]:
        if not self.bm25:
            logger.warning("BM25 index not built")
            return []

        # Apply metadata filters to locate eligible document indices
        filtered_indices = []
        for idx, doc in enumerate(self.documents):
            match = True
            if filters:
                for key, val in filters.items():
                    if val is not None and val != "":
                        doc_val = doc.metadata.get(key)
                        if key == "source":
                            if isinstance(val, list):
                                if doc_val not in val:
                                    match = False
                            elif isinstance(val, str) and "," in val:
                                parts = [p.strip() for p in val.split(",") if p.strip()]
                                if doc_val not in parts:
                                    match = False
                            elif doc_val != val:
                                match = False
                        elif key == "page":
                            try:
                                if int(doc_val) != int(val):
                                    match = False
                            except (ValueError, TypeError):
                                match = False
                        elif key == "strategy":
                            if doc_val != val:
                                match = False
            if match:
                filtered_indices.append(idx)

        if not filtered_indices:
            logger.info("BM25 search matched 0 documents with filters")
            return []

        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        # Sort filtered documents by their score
        ranked_filtered = sorted(filtered_indices, key=lambda idx: scores[idx], reverse=True)
        top_k_indices = ranked_filtered[:k]

        results = []
        for idx in top_k_indices:
            doc = self.documents[idx]
            doc.metadata["bm25_score"] = float(scores[idx])
            results.append(doc)

        logger.info(f"BM25 search returned {len(results)} results after applying filters")
        return results

    def save(self):
        try:
            INDEX_FILE.parent.mkdir(exist_ok=True)
            with open(INDEX_FILE, "wb") as f:
                pickle.dump((self.documents, self.tokenized_corpus), f)
            logger.info("BM25 index saved to file")
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")

    def load(self) -> bool:
        try:
            if INDEX_FILE.exists():
                with open(INDEX_FILE, "rb") as f:
                    self.documents, self.tokenized_corpus = pickle.load(f)
                self.bm25 = BM25Okapi(self.tokenized_corpus)
                logger.info(f"BM25 index loaded from file with {len(self.documents)} documents")
                return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
        return False

    def delete_document(self, doc_id: str):
        if not self.bm25:
            return
        new_docs = [doc for doc in self.documents if doc.metadata.get("doc_id") != doc_id]
        if len(new_docs) == len(self.documents):
            return
        if not new_docs:
            self.clear()
        else:
            self._build_index(new_docs)

    def clear(self):
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None
        if INDEX_FILE.exists():
            INDEX_FILE.unlink()
        logger.info("BM25 index cleared")


_bm25_instance = None


def get_bm25_retriever():
    global _bm25_instance
    if _bm25_instance is None:
        _bm25_instance = BM25Retriever()
        _bm25_instance.load()
    return _bm25_instance


def build_bm25_index(documents: list[Document]):
    global _bm25_instance
    _bm25_instance = BM25Retriever(documents)
    return _bm25_instance


async def bm25_search_async(query: str, k: int = 5, filters: dict = None) -> list[Document]:
    retriever = get_bm25_retriever()
    return await asyncio.to_thread(retriever.search, query, k, filters)


bm25_retrieval_chain = RunnableLambda(lambda x: bm25_search_async(x["query"], x.get("k", 5), x.get("filters")))
