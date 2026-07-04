import shutil
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Request
from app.config import settings
from app.services.validators import validate_all
from app.services.ingestion import get_ingestion_chain
from app.database.database import insert_document, list_documents, get_document, delete_document
from app.models.schemas import DocumentUploadResponse, DocumentListResponse, DocumentResponse
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def process_document(file_path: str, chroma_store):
    try:
        chain = get_ingestion_chain(chroma_store)
        chain.invoke(file_path)
        print(f"[stage01 | documents | 013-A] OK: Document processed")
    except Exception as e:
        print(f"[stage01 | documents | 013-A] FAIL: Processing failed - {e}")


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

        insert_document(
            doc_id=doc_id,
            filename=file.filename,
            file_type=file.content_type,
            file_size=file.size,
        )

        chroma_store = request.app.state.vectorstore
        background_tasks.add_task(process_document, str(file_path), chroma_store)

        print(f"[stage01 | documents | 013-B] OK: Upload accepted - {file.filename}")
        return DocumentUploadResponse(
            doc_id=doc_id, filename=file.filename, message="Document uploaded"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[stage01 | documents | 013-B] FAIL: Upload failed - {e}")
        raise HTTPException(status_code=500, detail="Upload failed")


@router.get("", response_model=DocumentListResponse)
async def get_documents():
    try:
        documents = list_documents()
        print(f"[stage01 | documents | 013-C] OK: Listed {len(documents)} documents")
        return DocumentListResponse(
            documents=[DocumentResponse(**doc) for doc in documents]
        )
    except Exception as e:
        print(f"[stage01 | documents | 013-C] FAIL: List failed - {e}")
        raise HTTPException(status_code=500, detail="List failed")


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_single_document(doc_id: str):
    try:
        document = get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        print(f"[stage01 | documents | 013-D] OK: Document found - {doc_id}")
        return DocumentResponse(**document)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[stage01 | documents | 013-D] FAIL: Get failed - {e}")
        raise HTTPException(status_code=500, detail="Get failed")


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

        print(f"[stage01 | documents | 013-E] OK: Document deleted - {doc_id}")
        return {"message": "Document deleted"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"[stage01 | documents | 013-E] FAIL: Delete failed - {e}")
        raise HTTPException(status_code=500, detail="Delete failed")
