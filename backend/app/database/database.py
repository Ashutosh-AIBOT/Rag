import sqlite3
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
        raise ValueError("doc_id is required")

    try:
        conn = get_db()
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        conn.close()
        if row:
            return dict(row)
        return None
    except Exception as e:
        logger.error(f"Get failed: {e}")
        raise


def get_document_by_filename(filename):
    if not filename:
        raise ValueError("filename is required")

    try:
        conn = get_db()
        row = conn.execute("SELECT * FROM documents WHERE filename = ?", (filename,)).fetchone()
        conn.close()
        if row:
            return dict(row)
        return None
    except Exception as e:
        logger.error(f"Check failed: {e}")
        raise


def list_documents():
    try:
        conn = get_db()
        rows = conn.execute("SELECT * FROM documents ORDER BY upload_date DESC").fetchall()
        conn.close()
        logger.info(f"Listed {len(rows)} documents")
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"List failed: {e}")
        raise


def delete_document(doc_id):
    if not doc_id:
        raise ValueError("doc_id is required")

    conn = None
    try:
        conn = get_db()
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.execute("DELETE FROM parent_documents WHERE document_id = ?", (doc_id,))
        conn.commit()
        logger.info(f"Document deleted: {doc_id}")
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def update_document_chunk_count(doc_id, chunk_count):
    if not doc_id:
        raise ValueError("doc_id is required")

    if chunk_count < 0:
        raise ValueError("chunk_count cannot be negative")

    conn = None
    try:
        conn = get_db()
        conn.execute("UPDATE documents SET chunk_count = ? WHERE id = ?", (chunk_count, doc_id))
        conn.commit()
        logger.info(f"Chunk count updated: {chunk_count}")
    except Exception as e:
        logger.error(f"Update failed: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def update_document_status(doc_id, status):
    if not doc_id:
        raise ValueError("doc_id is required")

    if status not in ("processing", "completed", "failed", "pending"):
        raise ValueError("status must be processing, completed, failed, or pending")

    conn = None
    try:
        conn = get_db()
        conn.execute("UPDATE documents SET status = ? WHERE id = ?", (status, doc_id))
        conn.commit()
        logger.info(f"Status updated: {status}")
    except Exception as e:
        logger.error(f"Status update failed: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def insert_parent_document(parent_id, document_id, parent_content, chunk_index):
    if not parent_id or not document_id or not parent_content:
        raise ValueError("parent_id, document_id, parent_content are required")

    conn = None
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO parent_documents VALUES (?, ?, ?, ?)",
            (parent_id, document_id, parent_content, chunk_index),
        )
        conn.commit()
        logger.info("Parent document inserted")
    except Exception as e:
        logger.error(f"Insert failed: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def get_parent_document(parent_id):
    if not parent_id:
        raise ValueError("parent_id is required")

    try:
        conn = get_db()
        row = conn.execute("SELECT parent_content FROM parent_documents WHERE id = ?", (parent_id,)).fetchone()
        conn.close()
        if row:
            return row["parent_content"]
        return None
    except Exception as e:
        logger.error(f"Get failed: {e}")
        raise
