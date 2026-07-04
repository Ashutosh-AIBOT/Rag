from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from app.core.logging import get_logger
from app.database.database import get_parent_document

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
    docs = vs.similarity_search(query, k=k)
    
    swapped_docs = []
    for doc in docs:
        strategy = doc.metadata.get("strategy")
        parent_id = doc.metadata.get("parent_id")
        if strategy == "parent-child" and parent_id:
            parent_content = get_parent_document(parent_id)
            if parent_content:
                swapped_doc = Document(
                    page_content=parent_content,
                    metadata=doc.metadata
                )
                swapped_docs.append(swapped_doc)
                continue
        swapped_docs.append(doc)
    return swapped_docs


retrieval_service = RunnableLambda(_search)
