import sqlite3
from pathlib import Path
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def init_sqlite_wal():
    try:
        conn = sqlite3.connect(settings.DATABASE_URL, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                total_pages INTEGER DEFAULT 0,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tags TEXT DEFAULT '',
                chunk_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'processing'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS parent_documents (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                parent_content TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                strategy TEXT DEFAULT 'vector',
                latency_ms INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_traces (
                id TEXT PRIMARY KEY,
                query_id TEXT NOT NULL,
                steps TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (query_id) REFERENCES query_history(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_results (
                id TEXT PRIMARY KEY,
                query_id TEXT NOT NULL,
                faithfulness REAL DEFAULT 0,
                relevancy REAL DEFAULT 0,
                precision REAL DEFAULT 0,
                recall REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (query_id) REFERENCES query_history(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS background_jobs (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                progress REAL DEFAULT 0.0,
                result TEXT,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        logger.info("SQLite WAL initialized")
    except Exception as e:
        logger.error(f"SQLite init failed: {e}")
        raise


def check_sqlite_health():
    try:
        conn = sqlite3.connect(settings.DATABASE_URL, timeout=30)
        conn.execute("SELECT 1")
        conn.close()
        logger.info("SQLite health check passed")
        return True
    except Exception as e:
        logger.error(f"SQLite health check failed: {e}")
        return False
