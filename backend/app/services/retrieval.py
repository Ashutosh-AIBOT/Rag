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
    filters = input_data.get("filters")
    if vs is None:
        logger.error("Vectorstore not available")
        return []
    
    chroma_filter = None
    if filters:
        conditions = []
        for key, value in filters.items():
            if value is not None and value != "":
                if key == "page":
                    try:
                        conditions.append({"page": int(value)})
                    except ValueError:
                        pass
                elif key == "page_start":
                    try:
                        conditions.append({"page": {"$gte": int(value)}})
                    except ValueError:
                        pass
                elif key == "page_end":
                    try:
                        conditions.append({"page": {"$lte": int(value)}})
                    except ValueError:
                        pass
                elif key in ["source", "strategy", "section", "doc_id", "tags", "upload_date", "date"]:
                    if isinstance(value, list):
                        if len(value) == 1:
                            conditions.append({key: str(value[0])})
                        elif len(value) > 1:
                            conditions.append({key: {"$in": [str(v) for v in value]}})
                    elif isinstance(value, str) and "," in value:
                        parts = [v.strip() for v in value.split(",") if v.strip()]
                        if len(parts) == 1:
                            conditions.append({key: parts[0]})
                        elif len(parts) > 1:
                            conditions.append({key: {"$in": parts}})
                    else:
                        conditions.append({key: str(value)})

        if len(conditions) > 1:
            chroma_filter = {"$and": conditions}
        elif len(conditions) == 1:
            chroma_filter = conditions[0]

    if chroma_filter:
        logger.info(f"Retrieving with filter: {chroma_filter}")
        docs = vs.similarity_search(query, k=k, filter=chroma_filter)
    else:
        docs = vs.similarity_search(query, k=k)
    
    swapped_docs = []
    for doc in docs:
        parent_chunk_id = doc.metadata.get("parent_chunk_id")
        if parent_chunk_id:
            try:
                parent_data = vs._collection.get(ids=[parent_chunk_id])
                if parent_data and parent_data.get("documents"):
                    from langchain_core.documents import Document as LCDocument
                    swapped_doc = LCDocument(
                        page_content=parent_data["documents"][0],
                        metadata=parent_data["metadatas"][0] if parent_data.get("metadatas") else doc.metadata
                    )
                    swapped_docs.append(swapped_doc)
                    continue
            except Exception as ex:
                logger.error(f"Failed to fetch parent chunk {parent_chunk_id}: {ex}")

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
