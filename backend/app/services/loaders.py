from pathlib import Path
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)

from app.core.logging import get_logger

logger = get_logger(__name__)

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".md": UnstructuredMarkdownLoader,
}


def _load(file_path: str) -> list[Document]:
    ext = Path(file_path).suffix.lower()
    loader_cls = LOADER_MAP.get(ext)
    if loader_cls is None:
        raise ValueError(f"No loader for: {ext}")
    loader = loader_cls(file_path)
    documents = loader.load()
    print(f"[stage01 | loaders | 009-A] OK: Loaded {len(documents)} pages")
    return documents


load_chain = RunnableLambda(_load)
