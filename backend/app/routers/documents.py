import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, Request, BackgroundTasks

from app.core.logging import get_logger
from app.core.dependencies import get_chroma_store
from app.database.database import list_documents, delete_document, get_document
from app.database.schemas import DocumentListResponse, DocumentResponse
from app.services.ingestion import ingest_document
from app.core.exceptions import DocumentNotFoundException

logger = get_logger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])


def process_upload(file_path: str, filename: str, chroma_store):
    try:
        result = ingest_document(file_path, chroma_store, filename)
        logger.info(f"Background ingestion complete: {result}")
    except Exception as e:
        logger.error(f"Background ingestion failed: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@router.post("/upload", status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
):
    upload_dir = tempfile.mkdtemp()
    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    chroma_store = get_chroma_store(request)
    background_tasks.add_task(process_upload, file_path, file.filename, chroma_store)

    return {"message": "Document upload accepted", "filename": file.filename}


@router.get("", response_model=DocumentListResponse)
async def get_documents():
    docs = list_documents()
    return DocumentListResponse(
        documents=[DocumentResponse(**doc) for doc in docs],
        total=len(docs),
    )


@router.delete("/{doc_id}")
async def remove_document(doc_id: str, request: Request):
    doc = get_document(doc_id)
    if not doc:
        raise DocumentNotFoundException(doc_id)

    chroma_store = get_chroma_store(request)
    from app.vectorstore.chroma import delete_from_chroma
    delete_from_chroma(chroma_store, {"doc_id": doc_id})

    delete_document(doc_id)
    return {"message": f"Document {doc_id} deleted"}
