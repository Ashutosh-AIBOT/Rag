import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from app.core.logging import get_logger
from app.services.validators import validate_all
from app.services.loaders import load_chain
from app.services.chunking import split_chain
from app.database.database import insert_document, update_document_chunk_count
from app.vectorstore.chroma import add_documents_to_chroma
from app.models.schemas import IngestionResult

logger = get_logger(__name__)


def _validate(file_path: str) -> dict:
    existing_id = validate_all(file_path)
    if existing_id:
        return {"file_path": file_path, "doc_id": existing_id, "duplicate": True}
    doc_id = str(uuid.uuid4())
    return {"file_path": file_path, "doc_id": doc_id, "duplicate": False}


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

    file_size = Path(file_path).stat().st_size
    insert_document(
        doc_id=doc_id,
        filename=filename,
        file_type=Path(file_path).suffix.lower(),
        file_size=file_size,
        total_pages=len(chunks),
    )
    update_document_chunk_count(doc_id, len(chunks))

    print(f"[stage01 | ingestion | 011-A] OK: Ingestion complete - {filename}")
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
