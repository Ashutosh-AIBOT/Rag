import sqlite3
from pathlib import Path

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

DB_PATH = Path(settings.DATABASE_URL.replace("sqlite+aiosqlite:///", ""))


def init_sqlite_wal() -> None:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """
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
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS parent_documents (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            parent_content TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        )
        """
    )
    conn.commit()
    conn.close()
    logger.info(f"SQLite WAL mode initialized at: {DB_PATH}")


def check_sqlite_health() -> bool:
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.execute("SELECT 1")
        result = cursor.fetchone()
        conn.close()
        return result is not None
    except Exception as e:
        logger.error(f"SQLite health check failed: {e}")
        return False
