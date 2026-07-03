import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

def get_db_connection() -> sqlite3.Connection:
    """Creates a connection to SQLite database and enables dict-like row access."""
    conn = sqlite3.connect(settings.SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db() -> None:
    """Initializes the database schema and enables Write-Ahead Logging (WAL) mode."""
    logger.info(f"Initializing SQLite database at: {settings.SQLITE_DB_PATH} in WAL mode.")
    conn = get_db_connection()
    try:
        # Enable WAL mode
        conn.execute("PRAGMA journal_mode=WAL;")
        
        # 1. Documents Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                tags TEXT,
                total_pages INTEGER DEFAULT 0,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # 2. Query History Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                id TEXT PRIMARY KEY,
                query_text TEXT NOT NULL,
                strategy TEXT NOT NULL,
                filters_applied TEXT,
                answer TEXT NOT NULL,
                latency_ms INTEGER NOT NULL,
                token_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # 3. Pipeline Traces Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_traces (
                id TEXT PRIMARY KEY,
                query_id TEXT NOT NULL,
                step_name TEXT NOT NULL,
                step_data TEXT, -- JSON string representing step metadata
                timing_ms INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (query_id) REFERENCES query_history(id) ON DELETE CASCADE
            );
        """)
        
        # 4. Evaluation Runs Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                avg_faithfulness REAL NOT NULL,
                avg_relevancy REAL NOT NULL,
                avg_precision REAL NOT NULL,
                avg_recall REAL NOT NULL,
                run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 5. Evaluation Run Details Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                category TEXT,
                answer TEXT NOT NULL,
                reference TEXT NOT NULL,
                faithfulness REAL NOT NULL,
                relevancy REAL NOT NULL,
                precision_score REAL NOT NULL,
                recall_score REAL NOT NULL,
                latency_ms INTEGER NOT NULL,
                FOREIGN KEY (run_id) REFERENCES evaluation_runs(id) ON DELETE CASCADE
            );
        """)

        # 6. Parent-Child Documents Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS parent_documents (
                parent_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                parent_text TEXT NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
            );
        """)
        
        # Add index optimizations
        conn.execute("CREATE INDEX IF NOT EXISTS idx_query_id ON pipeline_traces(query_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_run_id ON evaluation_details(run_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_parent_doc_id ON parent_documents(doc_id);")
        
        conn.commit()
        logger.info("Database schemas created and optimized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
        raise e
    finally:
        conn.close()

# =============================================================================
# DOCUMENTS HELPERS
# =============================================================================

def save_parent_document(parent_id: str, doc_id: str, parent_text: str) -> None:
    """Saves a parent document text to the database."""
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO parent_documents (parent_id, doc_id, parent_text) VALUES (?, ?, ?);",
            (parent_id, doc_id, parent_text)
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to save parent document: {e}")
    finally:
        conn.close()

def get_parent_document_text(parent_id: str) -> Optional[str]:
    """Retrieves a parent document text by parent_id from database."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT parent_text FROM parent_documents WHERE parent_id = ?;", (parent_id,))
        row = cursor.fetchone()
        return row["parent_text"] if row else None
    except Exception as e:
        logger.error(f"Failed to fetch parent document text for {parent_id}: {e}")
        return None
    finally:
        conn.close()


def save_document(doc_id: str, filename: str, file_type: str, file_size: int, tags: List[str], total_pages: int) -> None:
    conn = get_db_connection()
    try:
        tags_str = ",".join(tags) if tags else ""
        conn.execute(
            "INSERT OR REPLACE INTO documents (id, filename, file_type, file_size, tags, total_pages) VALUES (?, ?, ?, ?, ?, ?);",
            (doc_id, filename, file_type, file_size, tags_str, total_pages)
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to save document metadata: {e}")
    finally:
        conn.close()

def delete_document(doc_id: str) -> None:
    conn = get_db_connection()
    try:
        conn.execute("DELETE FROM documents WHERE id = ?;", (doc_id,))
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to delete document metadata: {e}")
    finally:
        conn.close()

def get_all_documents() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, filename, file_type, file_size, tags, total_pages, uploaded_at FROM documents ORDER BY uploaded_at DESC;")
        rows = cursor.fetchall()
        docs = []
        for r in rows:
            docs.append({
                "id": r["id"],
                "filename": r["filename"],
                "file_type": r["file_type"],
                "file_size": r["file_size"],
                "tags": r["tags"].split(",") if r["tags"] else [],
                "total_pages": r["total_pages"],
                "uploaded_at": r["uploaded_at"]
            })
        return docs
    except Exception as e:
        logger.error(f"Failed to retrieve documents list: {e}")
        return []
    finally:
        conn.close()

# =============================================================================
# QUERY HISTORY & TRACES HELPERS
# =============================================================================

def save_query_log(query_id: str, query_text: str, strategy: str, filters: Dict[str, Any], answer: str, latency_ms: int, token_count: int) -> None:
    conn = get_db_connection()
    try:
        filters_str = json.dumps(filters) if filters else "{}"
        conn.execute(
            "INSERT INTO query_history (id, query_text, strategy, filters_applied, answer, latency_ms, token_count) VALUES (?, ?, ?, ?, ?, ?, ?);",
            (query_id, query_text, strategy, filters_str, answer, latency_ms, token_count)
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to log query history: {e}")
    finally:
        conn.close()

def save_pipeline_trace(trace_id: str, query_id: str, step_name: str, step_data: Any, timing_ms: int) -> None:
    conn = get_db_connection()
    try:
        data_str = json.dumps(step_data) if step_data is not None else None
        conn.execute(
            "INSERT INTO pipeline_traces (id, query_id, step_name, step_data, timing_ms) VALUES (?, ?, ?, ?, ?);",
            (trace_id, query_id, step_name, data_str, timing_ms)
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to log pipeline trace step: {e}")
    finally:
        conn.close()

def get_query_history() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, query_text, strategy, filters_applied, answer, latency_ms, token_count, created_at FROM query_history ORDER BY created_at DESC;")
        rows = cursor.fetchall()
        history = []
        for r in rows:
            history.append({
                "id": r["id"],
                "query_text": r["query_text"],
                "strategy": r["strategy"],
                "filters_applied": json.loads(r["filters_applied"]) if r["filters_applied"] else {},
                "answer": r["answer"],
                "latency_ms": r["latency_ms"],
                "token_count": r["token_count"],
                "created_at": r["created_at"]
            })
        return history
    except Exception as e:
        logger.error(f"Failed to fetch query history: {e}")
        return []
    finally:
        conn.close()

# =============================================================================
# EVALUATION HELPERS
# =============================================================================

def save_evaluation_run(
    strategy: str,
    latency_ms: float,
    avg_faithfulness: float,
    avg_relevancy: float,
    avg_precision: float,
    avg_recall: float,
    details: Dict[int, Dict[str, Any]]
) -> int:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO evaluation_runs (strategy, latency_ms, avg_faithfulness, avg_relevancy, avg_precision, avg_recall) VALUES (?, ?, ?, ?, ?, ?);",
            (strategy, latency_ms, avg_faithfulness, avg_relevancy, avg_precision, avg_recall)
        )
        run_id = cursor.lastrowid
        
        # Save details
        for q_id, info in details.items():
            if "error" in info:
                continue
            conn.execute(
                """INSERT INTO evaluation_details (
                    run_id, question, category, answer, reference, faithfulness, relevancy, precision_score, recall_score, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                (
                    run_id,
                    info["question"],
                    info.get("category", "General"),
                    info["answer"],
                    info["reference"],
                    info["faithfulness"],
                    info["relevancy"],
                    info["precision"],
                    info["recall"],
                    info["latency_ms"]
                )
            )
        conn.commit()
        return run_id
    except Exception as e:
        logger.error(f"Failed to save evaluation run: {e}")
        return -1
    finally:
        conn.close()

def get_evaluation_runs() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, strategy, latency_ms, avg_faithfulness, avg_relevancy, avg_precision, avg_recall, run_date FROM evaluation_runs ORDER BY run_date DESC;")
        rows = cursor.fetchall()
        runs = []
        for r in rows:
            runs.append({
                "id": r["id"],
                "strategy": r["strategy"],
                "latency_ms": r["latency_ms"],
                "avg_faithfulness": r["avg_faithfulness"],
                "avg_relevancy": r["avg_relevancy"],
                "avg_precision": r["avg_precision"],
                "avg_recall": r["avg_recall"],
                "run_date": r["run_date"]
            })
        return runs
    except Exception as e:
        logger.error(f"Failed to fetch evaluation runs: {e}")
        return []
    finally:
        conn.close()

def get_pipeline_trace_steps(query_id: str) -> List[Dict[str, Any]]:
    """Retrieves step-by-step pipeline traces for a given query ID."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, step_name, step_data, timing_ms, created_at FROM pipeline_traces WHERE query_id = ? ORDER BY created_at ASC;",
            (query_id,)
        )
        rows = cursor.fetchall()
        steps = []
        for r in rows:
            steps.append({
                "id": r["id"],
                "step_name": r["step_name"],
                "step_data": json.loads(r["step_data"]) if r["step_data"] else None,
                "timing_ms": r["timing_ms"],
                "created_at": r["created_at"]
            })
        return steps
    except Exception as e:
        logger.error(f"Failed to fetch pipeline trace steps for query {query_id}: {e}")
        return []
    finally:
        conn.close()

def get_stats_summary() -> Dict[str, Any]:
    """Calculates aggregate metrics for the platform."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total_queries, AVG(latency_ms) as avg_latency, SUM(token_count) as total_tokens FROM query_history;")
        row = cursor.fetchone()
        total_queries = row["total_queries"] or 0
        avg_latency = row["avg_latency"] or 0.0
        total_tokens = row["total_tokens"] or 0
        
        # Estimate API costs (average of $0.000002 per token for input/output models)
        estimated_cost = total_tokens * 0.000002
        
        return {
            "total_queries": total_queries,
            "avg_latency_ms": round(avg_latency, 2),
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(estimated_cost, 6)
        }
    except Exception as e:
        logger.error(f"Failed to calculate stats summary: {e}")
        return {
            "total_queries": 0,
            "avg_latency_ms": 0.0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0
        }
    finally:
        conn.close()

