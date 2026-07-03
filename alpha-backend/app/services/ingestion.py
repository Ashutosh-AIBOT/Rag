import os
import uuid
import datetime
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

from app.core.logging import get_logger
from app.config import settings
from app.vectorstore.chroma import get_chroma_db
from app.services.chunking_strategies import split_recursive, split_parent_child, split_section_based, split_semantic
from app.database.database import save_document, delete_document, get_all_documents

logger = get_logger(__name__)

# =============================================================================
# INGESTION PIPELINE
# =============================================================================

def load_document(file_path: str) -> List[Document]:
    """Loads document using the correct loader depending on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    logger.info(f"Loading file: {file_path} with extension {ext}")
    
    docs = []
    if ext == ".pdf":
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        except Exception as e:
            logger.warning(f"PyPDFLoader failed: {e}. Falling back to UnstructuredPDFLoader.")
            try:
                from langchain_community.document_loaders import UnstructuredPDFLoader
                loader = UnstructuredPDFLoader(file_path)
                docs = loader.load()
            except Exception as ex:
                logger.error(f"UnstructuredPDFLoader also failed: {ex}")
                raise ex
        
        # Basic PDF section/header extraction per page
        for doc in docs:
            lines = [l.strip() for l in doc.page_content.split("\n") if l.strip()]
            header = lines[0] if lines else "General"
            doc.metadata["section_title"] = header[:80]
        return docs
    elif ext in [".docx", ".doc"]:
        try:
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
        except Exception as e:
            logger.warning(f"Docx2txtLoader failed: {e}. Falling back to UnstructuredWordDocumentLoader.")
            try:
                from langchain_community.document_loaders import UnstructuredWordDocumentLoader
                loader = UnstructuredWordDocumentLoader(file_path)
                docs = loader.load()
            except Exception as ex:
                logger.error(f"UnstructuredWordDocumentLoader failed: {ex}")
                raise ex
    elif ext == ".md":
        try:
            from langchain_community.document_loaders import UnstructuredMarkdownLoader
            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()
        except Exception as e:
            logger.warning(f"UnstructuredMarkdownLoader failed: {e}. Falling back to TextLoader.")
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
    else:
        logger.warning(f"Unsupported file format {ext}. Defaulting to TextLoader.")
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        
    return docs


def enrich_metadata(docs: List[Document], filename: str, doc_id: str, file_size: int, tags: List[str]) -> List[Document]:
    """Enriches pages metadata with filename, type, size, upload date, and custom tags."""
    upload_date = datetime.datetime.utcnow().isoformat()
    file_type = os.path.splitext(filename)[1].lower()
    tags_str = ",".join(tags) if tags else ""
    
    for doc in docs:
        doc.metadata["doc_id"] = doc_id
        doc.metadata["filename"] = filename
        doc.metadata["file_type"] = file_type
        doc.metadata["file_size"] = file_size
        doc.metadata["upload_date"] = upload_date
        doc.metadata["tags"] = tags_str
        # Standardize default section title if not set
        if "section_title" not in doc.metadata:
            doc.metadata["section_title"] = "General"
            
    return docs

def process_ingestion(file_path: str, filename: str, tags: List[str], embeddings, doc_id: str, app=None) -> None:
    """
    Full document ingestion pipeline meant to run inside a BackgroundTask.
    Loads, enriches, chunks using all strategies, and stores results in ChromaDB.
    """
    logger.info(f"Starting background ingestion for doc_id: {doc_id} (file: {filename})")
    try:
        # 1. Load document page-by-page
        raw_docs = load_document(file_path)
        file_size = os.path.getsize(file_path)
        total_pages = len(raw_docs) if os.path.splitext(filename)[1].lower() == ".pdf" else 1
        
        # 2. Enrich metadata
        enriched_docs = enrich_metadata(raw_docs, filename, doc_id, file_size, tags)
        
        # 3. Record document metadata in SQLite database first to satisfy foreign key constraints
        save_document(doc_id, filename, os.path.splitext(filename)[1].lower(), file_size, tags, total_pages)
        
        # 4. Split using all strategies
        recursive_chunks = split_recursive(enriched_docs, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        parent_child_chunks = split_parent_child(enriched_docs, settings.CHUNK_SIZE, settings.CHUNK_SIZE // 5)
        section_chunks = split_section_based(enriched_docs)
        semantic_chunks = split_semantic(enriched_docs, embeddings)
        
        # Save parent documents to SQLite and strip parent_text from ChromaDB metadata to optimize storage size
        from app.database.database import save_parent_document
        seen_parent_ids = set()
        for chunk in parent_child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            parent_text = chunk.metadata.pop("parent_text", None)
            if parent_id and parent_text and parent_id not in seen_parent_ids:
                save_parent_document(parent_id, doc_id, parent_text)
                seen_parent_ids.add(parent_id)
        
        all_chunks = recursive_chunks + parent_child_chunks + section_chunks + semantic_chunks
        
        # 5. Insert chunks to ChromaDB
        db = get_chroma_db(embeddings)
        db.add_documents(all_chunks)
        logger.info(f"Successfully added {len(all_chunks)} chunks to ChromaDB.")
        logger.info(f"Ingestion successful for document ID: {doc_id}")
        
        # 6. Invalidate BM25 index to trigger rebuild on next retrieval
        if app is not None:
            app.state.bm25_index = None
            logger.info("BM25 index cache cleared in app state.")
            
    except Exception as e:
        logger.error(f"Ingestion failed for doc_id: {doc_id}. Error: {e}", exc_info=True)
    finally:
        # Delete temporary uploaded file to clean space
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Temporary file removed: {file_path}")

def delete_document_from_platform(doc_id: str, embeddings) -> None:
    """Removes a document metadata from SQLite and all associated chunks from ChromaDB."""
    logger.info(f"Request to delete document ID: {doc_id}")
    
    # 1. Delete chunks from ChromaDB
    try:
        db = get_chroma_db(embeddings)
        db._collection.delete(where={"doc_id": doc_id})
        logger.info("ChromaDB chunks deleted.")
    except Exception as e:
        logger.error(f"Failed to delete chunks from ChromaDB for ID {doc_id}: {e}")
        
    # 2. Delete metadata from SQLite
    delete_document(doc_id)
    logger.info(f"Document ID: {doc_id} successfully deleted.")

