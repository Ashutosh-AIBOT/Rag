from pathlib import Path

from langchain_core.runnables import RunnableLambda

from app.core.logging import get_logger
from app.services.validators import validate_all
from app.services.loaders import load_chain
from app.services.chunking import split_chain
from app.database.database import get_document_by_filename, update_document_chunk_count, update_document_status
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
    chunks = data["chunks"]
    doc_id = data["doc_id"]
    file_path = data["file_path"]
    filename = Path(file_path).name

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [
        {
            "source": filename,
            "page": chunk.metadata.get("page", 0),
            "strategy": "recursive",
            "doc_id": doc_id,
        }
        for chunk in chunks
    ]
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

    add_documents_to_chroma(data["chroma_store"], texts, metadatas, ids)

    update_document_chunk_count(doc_id, len(chunks))
    update_document_status(doc_id, "completed")

    logger.info(f"Ingestion complete: {filename}")
    return IngestionResult(doc_id=doc_id, filename=filename, chunks=len(chunks), duplicate=data["duplicate"])


def _load_and_split(data: dict) -> dict:
    documents = load_chain.invoke(data["file_path"])
    chunks = split_chain.invoke(documents)
    data["chunks"] = chunks
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
