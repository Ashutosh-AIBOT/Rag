import sqlite3
from pathlib import Path

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

DB_PATH = Path(settings.DATABASE_URL.replace("sqlite+aiosqlite:///", ""))


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def insert_document(doc_id: str, filename: str, file_type: str, file_size: int, total_pages: int = 0, tags: str = "") -> None:
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO documents (id, filename, file_type, file_size, total_pages, tags) VALUES (?, ?, ?, ?, ?, ?)",
        (doc_id, filename, file_type, file_size, total_pages, tags),
    )
    conn.commit()
    conn.close()
    logger.info(f"Document inserted: {doc_id} ({filename})")


def get_document(doc_id: str) -> dict | None:
    conn = get_db_connection()
    cursor = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def list_documents() -> list[dict]:
    conn = get_db_connection()
    cursor = conn.execute("SELECT * FROM documents ORDER BY upload_date DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def delete_document(doc_id: str) -> bool:
    conn = get_db_connection()
    cursor = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.execute("DELETE FROM parent_documents WHERE document_id = ?", (doc_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def update_document_chunk_count(doc_id: str, chunk_count: int) -> None:
    conn = get_db_connection()
    conn.execute("UPDATE documents SET chunk_count = ? WHERE id = ?", (chunk_count, doc_id))
    conn.commit()
    conn.close()


def insert_parent_document(parent_id: str, document_id: str, parent_content: str, chunk_index: int) -> None:
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO parent_documents (id, document_id, parent_content, chunk_index) VALUES (?, ?, ?, ?)",
        (parent_id, document_id, parent_content, chunk_index),
    )
    conn.commit()
    conn.close()


def get_parent_document(parent_id: str) -> str | None:
    conn = get_db_connection()
    cursor = conn.execute("SELECT parent_content FROM parent_documents WHERE id = ?", (parent_id,))
    row = cursor.fetchone()
    conn.close()
    return row["parent_content"] if row else None
