import sqlite3
from pathlib import Path
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def get_db():
    try:
        conn = sqlite3.connect(settings.DATABASE_URL, timeout=30)
        conn.row_factory = sqlite3.Row
        logger.info("DB connection created")
        return conn
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        raise


def insert_document(doc_id, filename, file_type, file_size, total_pages=0, tags=""):
    if not doc_id or not filename or not file_type:
        raise ValueError("doc_id, filename, file_type are required")

    if file_size < 0:
        raise ValueError("file_size cannot be negative")

    conn = None
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?)",
            (doc_id, filename, file_type, file_size, total_pages, tags, 0, "processing"),
        )
        conn.commit()
        logger.info(f"Document inserted: {filename}")
    except Exception as e:
        logger.error(f"Insert failed: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def get_document(doc_id):
    if not doc_id:
        print("[stage01 | database | 012-C] FAIL: Missing doc_id")
        raise ValueError("doc_id is required")

    try:
        conn = get_db()
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        conn.close()
        if row:
            print(f"[stage01 | database | 012-C] OK: Document found - {doc_id}")
            return dict(row)
        print(f"[stage01 | database | 012-C] WARN: Document not found - {doc_id}")
        return None
    except Exception as e:
        print(f"[stage01 | database | 012-C] FAIL: Get failed - {e}")
        raise


def get_document_by_filename(filename):
    if not filename:
        print("[stage01 | database | 012-D] FAIL: Missing filename")
        raise ValueError("filename is required")

    try:
        conn = get_db()
        row = conn.execute("SELECT * FROM documents WHERE filename = ?", (filename,)).fetchone()
        conn.close()
        if row:
            print(f"[stage01 | database | 012-D] OK: Duplicate found - {filename}")
            return dict(row)
        print(f"[stage01 | database | 012-D] OK: No duplicate - {filename}")
        return None
    except Exception as e:
        print(f"[stage01 | database | 012-D] FAIL: Check failed - {e}")
        raise


def list_documents():
    try:
        conn = get_db()
        rows = conn.execute("SELECT * FROM documents ORDER BY upload_date DESC").fetchall()
        conn.close()
        print(f"[stage01 | database | 012-E] OK: Listed {len(rows)} documents")
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"[stage01 | database | 012-E] FAIL: List failed - {e}")
        raise


def delete_document(doc_id):
    if not doc_id:
        print("[stage01 | database | 012-F] FAIL: Missing doc_id")
        raise ValueError("doc_id is required")

    conn = None
    try:
        conn = get_db()
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.execute("DELETE FROM parent_documents WHERE document_id = ?", (doc_id,))
        conn.commit()
        print(f"[stage01 | database | 012-F] OK: Document deleted - {doc_id}")
    except Exception as e:
        print(f"[stage01 | database | 012-F] FAIL: Delete failed - {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def update_document_chunk_count(doc_id, chunk_count):
    if not doc_id:
        print("[stage01 | database | 012-G] FAIL: Missing doc_id")
        raise ValueError("doc_id is required")

    if chunk_count < 0:
        print("[stage01 | database | 012-G] FAIL: Invalid chunk_count")
        raise ValueError("chunk_count cannot be negative")

    conn = None
    try:
        conn = get_db()
        conn.execute("UPDATE documents SET chunk_count = ? WHERE id = ?", (chunk_count, doc_id))
        conn.commit()
        print(f"[stage01 | database | 012-G] OK: Chunk count updated - {chunk_count}")
    except Exception as e:
        print(f"[stage01 | database | 012-G] FAIL: Update failed - {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def insert_parent_document(parent_id, document_id, parent_content, chunk_index):
    if not parent_id or not document_id or not parent_content:
        print("[stage01 | database | 012-H] FAIL: Missing required fields")
        raise ValueError("parent_id, document_id, parent_content are required")

    conn = None
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO parent_documents VALUES (?, ?, ?, ?)",
            (parent_id, document_id, parent_content, chunk_index),
        )
        conn.commit()
        print(f"[stage01 | database | 012-H] OK: Parent document inserted")
    except Exception as e:
        print(f"[stage01 | database | 012-H] FAIL: Insert failed - {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def get_parent_document(parent_id):
    if not parent_id:
        print("[stage01 | database | 012-I] FAIL: Missing parent_id")
        raise ValueError("parent_id is required")

    try:
        conn = get_db()
        row = conn.execute("SELECT parent_content FROM parent_documents WHERE id = ?", (parent_id,)).fetchone()
        conn.close()
        if row:
            print(f"[stage01 | database | 012-I] OK: Parent document found")
            return row["parent_content"]
        print(f"[stage01 | database | 012-I] WARN: Parent document not found")
        return None
    except Exception as e:
        print(f"[stage01 | database | 012-I] FAIL: Get failed - {e}")
        raise