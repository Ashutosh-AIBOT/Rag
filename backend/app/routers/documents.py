import asyncio
import shutil
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Request
from app.config import settings
from app.services.validators import validate_all
from app.services.ingestion import get_ingestion_chain
from app.database.database import insert_document, list_documents, get_document, delete_document, update_document_status, get_document_by_filename
from app.models.schemas import DocumentUploadResponse, DocumentListResponse, DocumentResponse
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def process_document(file_path: str, doc_id: str, chroma_store):
    try:
        chain = get_ingestion_chain(chroma_store)
        result = chain.invoke({"file_path": file_path, "doc_id": doc_id})
        logger.info("Document processed")
    except Exception as e:
        update_document_status(doc_id, "failed")
        logger.error(f"Processing failed: {e}")


@router.post("/upload", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    request: Request = None,
):
    try:
        doc_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{doc_id}_{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        validate_all(str(file_path))

        existing = get_document_by_filename(file.filename)
        if existing:
            if existing["status"] == "completed":
                file_path.unlink()
                return DocumentUploadResponse(
                    doc_id=existing["id"], filename=file.filename, message="Document already processed"
                )
            elif existing["status"] == "processing":
                file_path.unlink()
                return DocumentUploadResponse(
                    doc_id=existing["id"], filename=file.filename, message="Document is being processed"
                )
            else:
                update_document_status(existing["id"], "processing")
                chroma_store = request.app.state.vectorstore
                background_tasks.add_task(process_document, str(file_path), existing["id"], chroma_store)
                return DocumentUploadResponse(
                    doc_id=existing["id"], filename=file.filename, message="Processing document"
                )

        insert_document(
            doc_id=doc_id,
            filename=file.filename,
            file_type=file.content_type,
            file_size=file.size,
        )

        chroma_store = request.app.state.vectorstore
        background_tasks.add_task(process_document, str(file_path), doc_id, chroma_store)

        logger.info(f"Upload accepted: {file.filename}")
        return DocumentUploadResponse(
            doc_id=doc_id, filename=file.filename, message="Document uploaded"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")


@router.get("", response_model=DocumentListResponse)
async def get_documents():
    try:
        documents = list_documents()
        logger.info(f"Listed {len(documents)} documents")
        return DocumentListResponse(
            documents=[DocumentResponse(**doc) for doc in documents]
        )
    except Exception as e:
        logger.error(f"List failed: {e}")
        raise HTTPException(status_code=500, detail="List failed")


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_single_document(doc_id: str):
    try:
        document = get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        logger.info(f"Document found: {doc_id}")
        return DocumentResponse(**document)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get failed: {e}")
        raise HTTPException(status_code=500, detail="Get failed")


@router.get("/{doc_id}/status")
async def get_document_status(doc_id: str):
    try:
        document = get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"status": document["status"], "chunk_count": document["chunk_count"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Status check failed")


@router.delete("/{doc_id}")
async def remove_document(doc_id: str):
    try:
        document = get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        delete_document(doc_id)

        file_path = UPLOAD_DIR / f"{doc_id}_{document['filename']}"
        if file_path.exists():
            file_path.unlink()

        logger.info(f"Document deleted: {doc_id}")
        return {"message": "Document deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail="Delete failed")


@router.delete("")
async def remove_all_documents():
    try:
        documents = list_documents()
        for doc in documents:
            delete_document(doc["id"])
            file_path = UPLOAD_DIR / f"{doc['id']}_{doc['filename']}"
            if file_path.exists():
                file_path.unlink()
        logger.info(f"Deleted {len(documents)} documents")
        return {"message": f"Deleted {len(documents)} documents"}
    except Exception as e:
        logger.error(f"Delete all failed: {e}")
        raise HTTPException(status_code=500, detail="Delete all failed")
