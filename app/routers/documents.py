"""
Document upload & management.

Ingestion (loading -> multi-strategy chunking -> embedding -> vector store
insert) can take anywhere from a couple seconds (small .txt) to a minute or
more (large PDFs run through 4 chunking strategies + local embeddings). To
keep the frontend responsive and support several people uploading at once,
`/upload` only does the fast disk write + DB row synchronously, then hands
the slow part to a bounded background worker pool (see job_manager.py) and
returns immediately with a job_id the client polls.
"""
import os
import shutil
import logging
import datetime as dt
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models.database import get_db, Document, IngestionJob, SessionLocal, User
from app.routers.auth import _get_current_user
from app.models.schemas import (
    DocumentOut, IngestionJobOut, UploadAccepted, DocumentChunksOut, DocumentChunkOut,
)
from app.services.ingestion import load_document, extract_total_pages
from app.services.chunking_strategies import run_all_strategies
from app.services.vector_store import add_chunks, delete_document_chunks, get_chunks_for_document
from app.services.job_manager import submit_ingestion
from app.services.retrieval import clear_bm25_cache
from app.services.cache import clear_pattern

logger = logging.getLogger("documents_router")
router = APIRouter(prefix="/api/documents", tags=["documents"])
settings = get_settings()

SUPPORTED_TYPES = {"pdf", "txt", "docx", "md", "markdown"}


def _run_ingestion_job(job_id: str, doc_id: str, dest_path: str, ext: str,
                        filename: str, tag_list: List[str]) -> None:
    """Runs on a background worker thread (job_manager.ingestion_executor).
    Opens its OWN DB session since SQLite sessions aren't thread-safe to
    share with the request thread that returned already."""
    db = SessionLocal()
    try:
        job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not job or not doc:
            logger.warning("[uid:%s] Ingestion Job: Job %s or Document %s not found in DB.", "-", job_id, doc_id)
            return

        uid = doc.user_id or "-"
        logger.info("[uid:%s] Ingestion started: job_id=%s file=%s ext=%s tags=%s", uid, job_id, filename, ext, tag_list)

        job.status = "running"
        job.progress = 5
        job.message = "Loading document with LangChain loader..."
        job.updated_at = dt.datetime.now(dt.timezone.utc)
        db.commit()

        raw_docs = load_document(dest_path, ext, doc_id, filename, tag_list, user_id=doc.user_id)
        logger.info("[uid:%s] Ingestion loaded: job_id=%s pages=%d", uid, job_id, len(raw_docs))

        job.progress = 30
        job.message = "Running all chunking strategies (recursive, semantic, parent-child, section)..."
        db.commit()

        chunk_sets = run_all_strategies(raw_docs)
        logger.info("[uid:%s] Ingestion chunked: job_id=%s strategies=%s", uid, job_id, list(chunk_sets.keys()))

        job.progress = 60
        job.message = "Embedding & storing chunks in ChromaDB..."
        db.commit()

        chunk_counts = {}
        all_chunks = []
        for strategy_name, chunks in chunk_sets.items():
            all_chunks.extend(chunks)
            chunk_counts[strategy_name] = len(chunks)

        add_chunks(all_chunks)
        logger.info("[uid:%s] Ingestion embedded: job_id=%s total_chunks=%d counts=%s", uid, job_id, len(all_chunks), chunk_counts)

        doc.total_pages = extract_total_pages(raw_docs)
        doc.file_size = os.path.getsize(dest_path)
        doc.chunk_counts = chunk_counts
        doc.status = "ready"
        db.commit()
        clear_bm25_cache()
        clear_pattern("query:result:*")

        job.status = "done"
        job.progress = 100
        job.message = "Ingestion complete."
        job.result = chunk_counts
        job.updated_at = dt.datetime.now(dt.timezone.utc)
        db.commit()
        logger.info("[uid:%s] Ingestion completed: job_id=%s doc_id=%s stats=%s", uid, job_id, doc_id, chunk_counts)
    except Exception as e:  # noqa: BLE001
        uid = doc.user_id if doc else "-"
        logger.error("[uid:%s] Ingestion failed: job_id=%s error=%s", uid, job_id, e, exc_info=True)
        db.rollback()
        doc = db.query(Document).filter(Document.id == doc_id).first()
        job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
        if doc:
            doc.status = "failed"
        if job:
            job.status = "failed"
            job.error = str(e)
            job.message = "Ingestion failed."
        db.commit()
    finally:
        db.close()


@router.post("/upload", response_model=UploadAccepted, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    tags: Optional[str] = Form(""),  # comma-separated
    db: Session = Depends(get_db),
    current_user=Depends(_get_current_user),
):
    safe_filename = os.path.basename(file.filename).replace("/", "").replace("\\", "")
    ext = safe_filename.split(".")[-1].lower()
    if ext not in SUPPORTED_TYPES:
        raise HTTPException(400, f"Unsupported file type: {ext}. Supported: {SUPPORTED_TYPES}")

    MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
    contents = await file.read()
    file_size = len(contents)
    if file_size == 0:
        raise HTTPException(400, "File is empty. Please upload a non-empty file.")
    if file_size > MAX_UPLOAD_SIZE:
        raise HTTPException(413, f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024*1024)}MB.")
    await file.seek(0)

    logger.info("[%s] Upload started: file=%s size=%d type=%s", current_user.email, safe_filename, file_size, ext)

    # Check for existing document with same filename, file_size, and status "ready"
    existing = db.query(Document).filter(
        Document.filename == file.filename,
        Document.file_size == file_size,
        Document.status == "ready",
        Document.user_id == current_user.id,
    ).first()

    if existing:
        # Create an instantly done IngestionJob
        job = IngestionJob(
            doc_id=existing.id,
            filename=safe_filename,
            status="done",
            progress=100,
            message="Ingestion complete (reused existing index).",
            result=existing.chunk_counts
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        logger.info("[%s] Upload reused existing: doc_id=%s job_id=%s", current_user.email, existing.id, job.id)
        return UploadAccepted(document=DocumentOut.model_validate(existing), job_id=job.id)

    os.makedirs(settings.upload_dir, exist_ok=True)
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    doc = Document(filename=safe_filename, file_type=ext, status="processing", tags=tag_list, file_size=file_size, user_id=current_user.id)
    db.add(doc)
    db.commit()
    db.refresh(doc)

    dest_path = os.path.join(settings.upload_dir, f"{doc.id}_{safe_filename}")
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    doc.file_path = dest_path
    db.commit()

    job = IngestionJob(doc_id=doc.id, filename=safe_filename, status="queued", message="Queued for ingestion...", user_id=current_user.id)
    db.add(job)
    db.commit()
    db.refresh(job)

    # Hand off to the bounded background worker pool -- returns instantly.
    submit_ingestion(_run_ingestion_job, job.id, doc.id, dest_path, ext, safe_filename, tag_list)

    logger.info("[%s] Upload accepted: doc_id=%s job_id=%s file=%s", current_user.email, doc.id, job.id, safe_filename)
    return UploadAccepted(document=DocumentOut.model_validate(doc), job_id=job.id)


@router.get("/jobs/{job_id}", response_model=IngestionJobOut)
def get_ingestion_job(job_id: str, db: Session = Depends(get_db), current_user: User = Depends(_get_current_user)):
    job = db.query(IngestionJob).filter(IngestionJob.id == job_id, IngestionJob.user_id == current_user.id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    logger.info("[%s] Get job: job_id=%s status=%s progress=%d", current_user.email, job_id, job.status, job.progress)
    return job


@router.get("", response_model=List[DocumentOut])
def list_documents(db: Session = Depends(get_db), current_user: User = Depends(_get_current_user)):
    docs = db.query(Document).filter(Document.user_id == current_user.id).order_by(Document.upload_date.desc()).all()
    logger.info("[%s] List documents: count=%d", current_user.email, len(docs))
    return docs


@router.get("/{doc_id}/chunks", response_model=DocumentChunksOut)
def get_document_chunks(doc_id: str, db: Session = Depends(get_db), current_user: User = Depends(_get_current_user)):
    """Powers Screen 5's 'click document to see all its chunks, grouped by
    strategy, with chunk text previews' requirement -- previously the only
    chunk endpoint required a free-text search query, so there was no way
    to just list everything a given document was split into."""
    doc = db.query(Document).filter(Document.id == doc_id, Document.user_id == current_user.id).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    grouped = get_chunks_for_document(doc_id)
    total_chunks = sum(len(chunks) for chunks in grouped.values())
    logger.info("[%s] Get chunks: doc_id=%s strategies=%d total_chunks=%d", current_user.email, doc_id, len(grouped), total_chunks)
    return DocumentChunksOut(
        doc_id=doc_id,
        filename=doc.filename,
        by_strategy={
            strategy: [DocumentChunkOut(**c) for c in chunks]
            for strategy, chunks in grouped.items()
        },
    )


@router.delete("/{doc_id}")
def delete_document(doc_id: str, db: Session = Depends(get_db), current_user: User = Depends(_get_current_user)):
    doc = db.query(Document).filter(Document.id == doc_id, Document.user_id == current_user.id).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    logger.info("[%s] Delete started: doc_id=%s file=%s", current_user.email, doc_id, doc.filename)
    delete_document_chunks(doc_id, user_id=current_user.id)

    # Remove the uploaded file from disk too -- previously only the DB row
    # and vectors were deleted, leaking one file per upload forever.
    file_path = doc.file_path or os.path.join(settings.upload_dir, f"{doc.id}_{doc.filename}")
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except OSError:
        pass  # best-effort; don't fail the delete over a filesystem hiccup

    db.delete(doc)
    db.commit()
    clear_bm25_cache()
    clear_pattern("query:result:*")
    logger.info("[%s] Delete completed: doc_id=%s", current_user.email, doc_id)
    return {"deleted": doc_id}


@router.get("/{doc_id}/overlap")
def get_chunk_overlap(doc_id: str, db: Session = Depends(get_db), current_user: User = Depends(_get_current_user)):
    """
    Good-to-Have: Chunk overlap visualization.
    Returns per-strategy chunk span metadata (text length, section, page,
    chunk_id) so the frontend can render a heatmap showing how different
    chunking strategies carve the document into different shapes and sizes.

    Since ChromaDB does not store character offsets, we use `text_length`
    as a proxy for span width -- the frontend renders proportional bars
    that show how each strategy's chunks divide the document space, and
    highlights which sections are covered by multiple strategies.
    """
    doc = db.query(Document).filter(Document.id == doc_id, Document.user_id == current_user.id).first()
    if not doc:
        raise HTTPException(404, "Document not found")

    grouped = get_chunks_for_document(doc_id)

    strategies_data = {}
    total_chars_by_strategy: dict = {}

    for strategy, chunks in grouped.items():
        strategy_chunks = []
        total = 0
        for chunk in chunks:
            text_len = len(chunk.get("text", ""))
            total += text_len
            strategy_chunks.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "text_preview": chunk.get("text", "")[:120].strip(),
                "text_length": text_len,
                "page": chunk.get("page"),
                "section": chunk.get("section", ""),
                "source": chunk.get("source", ""),
            })
        strategies_data[strategy] = strategy_chunks
        total_chars_by_strategy[strategy] = total

    logger.info("[%s] Get overlap: doc_id=%s strategies=%d", current_user.email, doc_id, len(grouped))
    return {
        "doc_id": doc_id,
        "filename": doc.filename,
        "total_chars_by_strategy": total_chars_by_strategy,
        "strategies": strategies_data,
    }

