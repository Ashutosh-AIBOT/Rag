import os
import uuid
import shutil
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, BackgroundTasks, Request, HTTPException, Form
from app.config import settings
from app.core.logging import get_logger
from app.services.ingestion import process_ingestion, delete_document_from_platform
from app.database.database import get_all_documents

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload", status_code=202)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tags: Optional[str] = Form(None)
):
    """
    Accepts a document file and returns a 202 Accepted status with a unique document ID.
    The document loading, splitting, and embedding logic runs in the background.
    """
    logger.info(f"Received upload request for file: {file.filename}")

    # Validate file format
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        logger.warning(f"File extension {ext} not permitted.")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )

    # Parse optional tags
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    # Save to a temporary location for the background thread to read
    temp_dir = os.path.join(settings.DATA_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    doc_id = str(uuid.uuid4())
    temp_file_path = os.path.join(temp_dir, f"{doc_id}{ext}")

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Stored temporary upload file at: {temp_file_path}")
    except Exception as e:
        logger.error(f"Failed to write uploaded file to disk: {e}")
        raise HTTPException(status_code=500, detail="Failed to store uploaded file.")

    # Retrieve preloaded embeddings from global app state
    embeddings = getattr(request.app.state, "embeddings", None)
    if embeddings is None:
        logger.warning("Embedding model is not loaded in app state. Vectors will not be generated.")

    # Dispatch full processing pipeline to background thread
    background_tasks.add_task(
        process_ingestion,
        temp_file_path,
        file.filename,
        tag_list,
        embeddings,
        doc_id,
        request.app
    )

    return {
        "status": "accepted",
        "document_id": doc_id,
        "message": "File accepted. Processing is running in the background."
    }

@router.get("")
async def list_documents():
    """
    Lists metadata of all currently ingested documents in the system.
    """
    logger.info("Retrieving all documents from SQLite database.")
    docs = get_all_documents()
    return docs

@router.delete("/{doc_id}")
async def delete_document(request: Request, doc_id: str):
    """
    Deletes the document's chunks from ChromaDB and metadata from SQLite.
    """
    logger.info(f"Deleting document with ID: {doc_id}")
    embeddings = getattr(request.app.state, "embeddings", None)
    delete_document_from_platform(doc_id, embeddings)
    
    # Invalidate BM25 index on deletion
    request.app.state.bm25_index = None
    logger.info("BM25 index cache cleared in app state on deletion.")
    
    return {
        "status": "deleted",
        "document_id": doc_id,
        "message": f"Document '{doc_id}' and all its vector chunks have been deleted."
    }
