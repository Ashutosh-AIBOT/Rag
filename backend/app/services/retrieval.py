from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from app.core.logging import get_logger

logger = get_logger(__name__)

_vectorstore = None


def set_vectorstore(vs):
    global _vectorstore
    _vectorstore = vs


def _search(input_data: dict) -> list[Document]:
    query = input_data["query"]
    k = input_data.get("k", 5)
    vs = input_data.get("vectorstore", _vectorstore)
    if vs is None:
        logger.error("Vectorstore not available")
        return []
    return vs.similarity_search(query, k=k)


retrieval_service = RunnableLambda(_search)
