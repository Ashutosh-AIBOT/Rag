from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from app.vectorstore.chroma import get_vectorstore
from app.core.logging import get_logger

logger = get_logger(__name__)


def _search(query: str, k: int = 5) -> list[Document]:
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search(query, k=k)


retrieval_service = RunnableLambda(lambda x: _search(x["query"], x.get("k", 5)))
