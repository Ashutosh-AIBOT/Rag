import sqlite3
import uuid
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


def update_document_pages(doc_id, total_pages):
    if not doc_id:
        raise ValueError("doc_id is required")

    conn = None
    try:
        conn = get_db()
        conn.execute("UPDATE documents SET total_pages = ? WHERE id = ?", (total_pages, doc_id))
        conn.commit()
        logger.info(f"Total pages updated: {total_pages}")
    except Exception as e:
        logger.error(f"Pages update failed: {e}")
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


def insert_query_history(query_id, question, answer, strategy, latency_ms):
    conn = None
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO query_history VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            (query_id, question, answer, strategy, latency_ms),
        )
        conn.commit()
        logger.info(f"Query history inserted: {query_id}")
    except Exception as e:
        logger.error(f"Query history insert failed: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def get_query_history(query_id):
    try:
        conn = get_db()
        row = conn.execute("SELECT * FROM query_history WHERE id = ?", (query_id,)).fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        logger.error(f"Query history get failed: {e}")
        raise


def insert_pipeline_trace(trace_id, query_id, steps):
    conn = None
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO pipeline_traces VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            (trace_id, query_id, str(steps)),
        )
        conn.commit()
        logger.info(f"Pipeline trace inserted: {trace_id}")
    except Exception as e:
        logger.error(f"Pipeline trace insert failed: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def get_pipeline_trace(trace_id):
    try:
        conn = get_db()
        row = conn.execute("SELECT * FROM pipeline_traces WHERE id = ?", (trace_id,)).fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        logger.error(f"Pipeline trace get failed: {e}")
        raise


def get_pipeline_trace_by_query_id(query_id):
    try:
        conn = get_db()
        row = conn.execute("SELECT * FROM pipeline_traces WHERE query_id = ?", (query_id,)).fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        logger.error(f"Pipeline trace get by query_id failed: {e}")
        raise


def insert_eval_result(eval_id, query_id, faithfulness, relevancy, precision, recall):
    conn = None
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO eval_results VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            (eval_id, query_id, faithfulness, relevancy, precision, recall),
        )
        conn.commit()
        logger.info(f"Eval result inserted: {eval_id}")
    except Exception as e:
        logger.error(f"Eval result insert failed: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def list_eval_results():
    try:
        conn = get_db()
        rows = conn.execute("SELECT * FROM eval_results ORDER BY created_at DESC").fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Eval results list failed: {e}")
        raise


def get_retrieval_stats():
    try:
        conn = get_db()
        total_queries = conn.execute("SELECT COUNT(*) FROM query_history").fetchone()[0]
        avg_latency = conn.execute("SELECT AVG(latency_ms) FROM query_history").fetchone()[0] or 0.0
        strategies = conn.execute("SELECT strategy, COUNT(*) FROM query_history GROUP BY strategy").fetchall()
        strategy_counts = {row["strategy"]: row[1] for row in strategies}
        total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        total_chunks = conn.execute("SELECT SUM(chunk_count) FROM documents").fetchone()[0] or 0
        
        traces = conn.execute("SELECT steps FROM pipeline_traces").fetchall()
        total_input_tokens = 0
        total_output_tokens = 0
        for row in traces:
            try:
                import json
                steps = json.loads(row["steps"])
                q_len = len(steps.get("original_query", ""))
                c_len = sum(len(c.get("content", "")) for c in steps.get("retrieved_chunks", []))
                a_len = len(steps.get("answer", ""))
                total_input_tokens += (q_len + c_len) // 4
                total_output_tokens += a_len // 4
            except Exception:
                pass
        
        conn.close()
        return {
            "total_queries": total_queries,
            "avg_latency_ms": round(avg_latency, 2),
            "strategy_counts": strategy_counts,
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "estimated_input_tokens": total_input_tokens,
            "estimated_output_tokens": total_output_tokens,
            "estimated_total_tokens": total_input_tokens + total_output_tokens,
        }
    except Exception as e:
        logger.error(f"Get stats failed: {e}")
        raise


def list_recent_queries(limit=10):
    try:
        conn = get_db()
        rows = conn.execute(
            "SELECT * FROM query_history ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"List recent queries failed: {e}")
        raise
