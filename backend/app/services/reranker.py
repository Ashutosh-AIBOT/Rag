from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from app.core.logging import get_logger
import asyncio

logger = get_logger(__name__)

_cross_encoder_instance = None


def get_cross_encoder():
    global _cross_encoder_instance
    if _cross_encoder_instance is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder model...")
        _cross_encoder_instance = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Cross-encoder model loaded")
    return _cross_encoder_instance


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder loaded")

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        if not documents:
            return []

        self._load_model()

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        result = []
        for rank, idx in enumerate(ranked_indices[:top_k]):
            doc = documents[idx]
            doc.metadata["original_rank"] = idx + 1
            doc.metadata["reranked_position"] = rank + 1
            doc.metadata["relevance_score"] = float(scores[idx])
            result.append(doc)

        logger.info(f"Reranked {len(documents)} docs to {len(result)} docs")
        return result


_reranker_instance = None


def get_reranker():
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance


async def rerank_async(query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
    reranker = get_reranker()
    return await asyncio.to_thread(reranker.rerank, query, documents, top_k)


rerank_chain = RunnableLambda(lambda x: rerank_async(x["query"], x["documents"], x.get("top_k", 5)))
