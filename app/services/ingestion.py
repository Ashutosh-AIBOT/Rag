"""
Loads raw files (PDF / TXT / DOCX / Markdown) into LangChain Documents
using the appropriate loader, and enriches every page/doc with metadata
(file name, type, page number, upload date, tags, detected section headers).

This module is deliberately provider-agnostic: it just returns a list of
`langchain_core.documents.Document` with rich `.metadata`, ready to be
handed to `chunking_strategies.py`.
"""
import os
import re
import logging
import datetime as dt
from typing import List, Optional

from langchain_core.documents import Document as LCDocument

logger = logging.getLogger("ingestion_service")


HEADER_RE = re.compile(r"^(#{1,2})\s+(.*)$", re.MULTILINE)  # markdown-style headers
CAPS_HEADER_RE = re.compile(r"^([A-Z][A-Za-z0-9 \-/&]{3,60})\n", re.MULTILINE)  # heuristic for PDFs/txt

# Heuristic "document date" extraction: looks for common date formats
# anywhere in the first couple pages (e.g. "March 2025", "2025-03-14",
# "Q3 2025", "FY2025"). Best-effort only -- used for the date metadata
# filter, not for anything safety-critical.
DATE_PATTERNS = [
    re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b"),
    re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(20\d{2})\b"),
    re.compile(r"\b(Q[1-4]\s*20\d{2})\b"),
    re.compile(r"\b(FY\s?20\d{2})\b"),
]


def _extract_doc_date(text: str) -> Optional[str]:
    snippet = text[:4000]
    for pattern in DATE_PATTERNS:
        m = pattern.search(snippet)
        if m:
            return m.group(0)
    return None


def _detect_current_section(text: str, position: int, sections: List[tuple]) -> str:
    """Given a list of (offset, header) tuples sorted ascending, find which
    section a given character offset in the full document falls under."""
    current = "Untitled Section"
    for offset, header in sections:
        if offset <= position:
            current = header
        else:
            break
    return current


def _find_sections(full_text: str) -> List[tuple]:
    sections = []
    for m in HEADER_RE.finditer(full_text):
        sections.append((m.start(), m.group(2).strip()))
    if not sections:
        for m in CAPS_HEADER_RE.finditer(full_text):
            sections.append((m.start(), m.group(1).strip()))
    return sorted(sections, key=lambda x: x[0])


def load_document(file_path: str, file_type: str, doc_id: str,
                   filename: str, tags: List[str], user_id: Optional[str] = None) -> List[LCDocument]:
    """Load a file with the correct LangChain loader and enrich metadata."""
    file_type = file_type.lower()
    logger.info("[uid:%s] Loading document: file=%s type=%s doc_id=%s", user_id or "-", filename, file_type, doc_id)

    if file_type == "pdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        docs = loader.load()  # one Document per page, metadata['page'] already set
    elif file_type == "txt":
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
    elif file_type == "docx":
        try:
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader
            loader = UnstructuredWordDocumentLoader(file_path)
            docs = loader.load()
        except Exception as e:
            logger.warning("[uid:%s] UnstructuredWordDocumentLoader failed for %s, falling back to Docx2txtLoader: %s", user_id or "-", filename, e)
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
    elif file_type in ("md", "markdown"):
        try:
            from langchain_community.document_loaders import UnstructuredMarkdownLoader
            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()
        except Exception as e:
            logger.warning("[uid:%s] UnstructuredMarkdownLoader failed for %s, falling back to TextLoader: %s",
                           user_id or "-", filename, e)
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
    else:
        raise ValueError(f"Unsupported file_type: {file_type}")

    full_text = "\n".join(d.page_content for d in docs)
    sections = _find_sections(full_text)
    doc_date = _extract_doc_date(full_text)

    file_size = os.path.getsize(file_path)
    upload_dt = dt.datetime.now(dt.timezone.utc)
    upload_date = upload_dt.isoformat()
    upload_ts = upload_dt.replace(tzinfo=dt.timezone.utc).timestamp()
    total_pages = extract_total_pages(docs)

    running_offset = 0
    enriched: List[LCDocument] = []
    for i, d in enumerate(docs):
        section = _detect_current_section(full_text, running_offset, sections)
        running_offset += len(d.page_content) + 1

        d.metadata.update({
            "doc_id": doc_id,
            "source": filename,
            "file_type": file_type,
            "page": d.metadata.get("page", i) + 1 if "page" in d.metadata else i + 1,
            "total_pages": total_pages,
            "section": section,
            "upload_date": upload_date,
            "upload_ts": upload_ts,
            "doc_date": doc_date,
            "file_size": file_size,
            "tags": tags,
            "user_id": user_id or "",
        })
        enriched.append(d)

    logger.info("[uid:%s] Document loaded: file=%s pages=%d sections=%d date=%s", user_id or "-", filename, len(enriched), len(sections), doc_date)
    return enriched


def extract_doc_date(docs: List[LCDocument]) -> Optional[str]:
    for d in docs:
        if d.metadata.get("doc_date"):
            return d.metadata["doc_date"]
    return None


def extract_total_pages(docs: List[LCDocument]) -> int:
    pages = {d.metadata.get("page") for d in docs if d.metadata.get("page") is not None}
    return len(pages) if pages else len(docs)
