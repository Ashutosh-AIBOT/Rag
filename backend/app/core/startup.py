import sqlite3
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)
DB_PATH = Path("rag_database.db")


def init_sqlite_wal() -> None:
    try:
        conn = sqlite3.connect(str(DB_PATH))
    except Exception as e:
        print(f"[stage00 | startup | 007] FAIL: SQLite connect - {e}")
        raise

    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        print("[stage00 | startup | 008] OK: WAL mode enabled")
    except Exception as e:
        print(f"[stage00 | startup | 008] FAIL: WAL pragma - {e}")
        conn.close()
        raise

    try:
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
        conn.commit()
        print("[stage00 | startup | 009] OK: Documents table created")
    except Exception as e:
        print(f"[stage00 | startup | 009] FAIL: Table create - {e}")
        conn.close()
        raise

    conn.close()
    print("[stage00 | startup | 010] OK: SQLite ready")


def check_sqlite_health() -> bool:
    try:
        conn = sqlite3.connect(str(DB_PATH))
        result = conn.execute("SELECT 1").fetchone()
        conn.close()
        print("[stage00 | startup | 011] OK: SQLite health passed")
        return result is not None
    except Exception as e:
        print(f"[stage00 | startup | 011] FAIL: Health check - {e}")
        return False