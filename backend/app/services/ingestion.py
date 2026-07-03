import uuid
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.core.logging import get_logger
from app.database.database import insert_document, update_document_chunk_count
from app.vectorstore.chroma import add_documents_to_chroma

logger = get_logger(__name__)

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".md": UnstructuredMarkdownLoader,
}

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def get_loader(file_path: str):
    ext = Path(file_path).suffix.lower()
    loader_cls = LOADER_MAP.get(ext)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader_cls(file_path)


def load_and_split(file_path: str) -> list:
    loader = get_loader(file_path)
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} pages from {file_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks


def ingest_document(file_path: str, chroma_store, filename: str) -> dict:
    doc_id = str(uuid.uuid4())

    chunks = load_and_split(file_path)

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

    add_documents_to_chroma(chroma_store, texts, metadatas, ids)

    file_size = Path(file_path).stat().st_size
    insert_document(
        doc_id=doc_id,
        filename=filename,
        file_type=Path(file_path).suffix.lower(),
        file_size=file_size,
        total_pages=len(chunks),
    )
    update_document_chunk_count(doc_id, len(chunks))

    logger.info(f"Ingestion complete: {doc_id} ({filename}) - {len(chunks)} chunks")
    return {"doc_id": doc_id, "filename": filename, "chunks": len(chunks)}
