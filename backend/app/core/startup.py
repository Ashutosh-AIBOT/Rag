import sqlite3
from pathlib import Path
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def init_sqlite_wal():
    try:
        conn = sqlite3.connect(settings.DATABASE_URL, timeout=10)
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
                chunk_count INTEGER DEFAULT 0
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
        conn.commit()
        conn.close()
        print("[stage00 | startup | 003-A] OK: SQLite WAL initialized")
    except Exception as e:
        print(f"[stage00 | startup | 003-A] FAIL: SQLite init failed - {e}")
        raise


def check_sqlite_health():
    try:
        conn = sqlite3.connect(settings.DATABASE_URL, timeout=10)
        conn.execute("SELECT 1")
        conn.close()
        print("[stage00 | startup | 003-B] OK: SQLite health check passed")
        return True
    except Exception as e:
        print(f"[stage00 | startup | 003-B] FAIL: SQLite health check failed - {e}")
        return False