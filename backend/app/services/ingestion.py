from pathlib import Path
from langchain_core.runnables import RunnableLambda
from app.core.logging import get_logger
from app.services.loaders import load_chain
from app.services.chunking_strategies import all_strategies_chain
from app.database.database import get_document_by_filename, update_document_chunk_count, update_document_status, insert_parent_document, update_document_pages
from app.vectorstore.chroma import add_documents_to_chroma
from app.models.schemas import IngestionResult

logger = get_logger(__name__)


def _validate(data: dict) -> dict:
    file_path = data["file_path"]
    doc_id = data["doc_id"]
    existing = get_document_by_filename(Path(file_path).name)
    if existing:
        data["doc_id"] = existing["id"]
        data["duplicate"] = True
        return data
    data["duplicate"] = False
    return data


def _store(data: dict) -> IngestionResult:
    chunks_result = data["chunks"]
    doc_id = data["doc_id"]
    file_path = data["file_path"]
    filename = Path(file_path).name

    all_chunks = []
    for strategy_name, chunks in chunks_result.items():
        if strategy_name == "parent_mapping":
            continue
        if isinstance(chunks, dict):
            if "child_chunks" in chunks:
                all_chunks.extend(chunks["child_chunks"])
            continue
        all_chunks.extend(chunks)

    texts = [chunk.page_content for chunk in all_chunks]
    metadatas = [
        {
            "source": filename,
            "page": chunk.metadata.get("page", 0),
            "strategy": chunk.metadata.get("strategy", "recursive"),
            "doc_id": doc_id,
            "section": chunk.metadata.get("section", ""),
            "parent_id": chunk.metadata.get("parent_id", ""),
        }
        for chunk in all_chunks
    ]
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(all_chunks))]

    add_documents_to_chroma(data["chroma_store"], texts, metadatas, ids)

    parent_mapping = []
    if "parent_child" in chunks_result and isinstance(chunks_result["parent_child"], dict):
        parent_mapping = chunks_result["parent_child"].get("parent_mapping", [])

    inserted_parent_ids = set()
    for mapping in parent_mapping:
        p_id = mapping["parent_id"]
        if p_id in inserted_parent_ids:
            continue
        inserted_parent_ids.add(p_id)
        insert_parent_document(
            parent_id=p_id,
            document_id=doc_id,
            parent_content=mapping["parent_content"],
            chunk_index=0,
        )

    update_document_chunk_count(doc_id, len(all_chunks))
    update_document_pages(doc_id, data.get("total_pages", 0))
    update_document_status(doc_id, "completed")

    try:
        from app.services.bm25_retriever import get_bm25_retriever
        bm25_retriever = get_bm25_retriever()
        current_docs = list(bm25_retriever.documents)
        current_docs.extend(all_chunks)
        bm25_retriever._build_index(current_docs)
    except Exception as e:
        logger.error(f"Failed to update BM25 index during ingestion: {e}")

    logger.info(f"Ingestion complete: {filename} - {len(all_chunks)} chunks")
    return IngestionResult(
        doc_id=doc_id,
        filename=filename,
        chunks=len(all_chunks),
        duplicate=data["duplicate"],
    )


def _load_and_split(data: dict) -> dict:
    documents = load_chain.invoke(data["file_path"])
    chunks = all_strategies_chain.invoke(documents)
    data["chunks"] = chunks
    data["total_pages"] = len(documents)
    return data


def get_ingestion_chain(chroma_store):
    def _set_chroma(data: dict) -> dict:
        data["chroma_store"] = chroma_store
        return data

    return (
        RunnableLambda(_validate)
        | RunnableLambda(_set_chroma)
        | RunnableLambda(_load_and_split)
        | RunnableLambda(_store)
    )
