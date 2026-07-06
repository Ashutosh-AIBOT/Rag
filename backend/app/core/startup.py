import sqlite3
from pathlib import Path
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


from app.database.database import get_db

def init_sqlite_wal():
    try:
        conn = get_db()
        if not conn.is_postgres:
            conn.conn.execute("PRAGMA journal_mode=WAL")
            conn.conn.execute("PRAGMA synchronous=NORMAL")
            conn.conn.execute("PRAGMA foreign_keys=ON")
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id VARCHAR(255) PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                file_type VARCHAR(100) NOT NULL,
                file_size INTEGER NOT NULL,
                total_pages INTEGER DEFAULT 0,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tags VARCHAR(255) DEFAULT '',
                chunk_count INTEGER DEFAULT 0,
                status VARCHAR(50) DEFAULT 'processing'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS parent_documents (
                id VARCHAR(255) PRIMARY KEY,
                document_id VARCHAR(255) NOT NULL,
                parent_content TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                id VARCHAR(255) PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                strategy VARCHAR(100) DEFAULT 'vector',
                latency_ms INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_traces (
                id VARCHAR(255) PRIMARY KEY,
                query_id VARCHAR(255) NOT NULL,
                steps TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (query_id) REFERENCES query_history(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_results (
                id VARCHAR(255) PRIMARY KEY,
                query_id VARCHAR(255) NOT NULL,
                faithfulness REAL DEFAULT 0,
                relevancy REAL DEFAULT 0,
                precision REAL DEFAULT 0,
                recall REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (query_id) REFERENCES query_history(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS background_jobs (
                id VARCHAR(255) PRIMARY KEY,
                type VARCHAR(100) NOT NULL,
                status VARCHAR(50) NOT NULL,
                progress REAL DEFAULT 0.0,
                result TEXT,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id VARCHAR(255) PRIMARY KEY,
                username VARCHAR(100) NOT NULL,
                daily_token_limit INTEGER DEFAULT 50000,
                tokens_consumed INTEGER DEFAULT 0,
                last_reset_date DATE DEFAULT CURRENT_DATE
            )
        """)
        
        # Insert a default seed user if not exists
        try:
            conn.execute("INSERT INTO users (id, username, daily_token_limit, tokens_consumed) VALUES ('default_user', 'Enterprise Admin', 100000, 0) ON CONFLICT (id) DO NOTHING")
        except Exception:
            # SQLite does not support ON CONFLICT in some older versions or ON CONFLICT (id) syntax
            try:
                # Fallback for older sqlite versions
                existing = conn.execute("SELECT id FROM users WHERE id = 'default_user'").fetchone()
                if not existing:
                    conn.execute("INSERT INTO users (id, username, daily_token_limit, tokens_consumed) VALUES ('default_user', 'Enterprise Admin', 100000, 0)")
            except Exception:
                pass

        conn.commit()
        conn.close()
        logger.info("Database schemas initialized successfully")
    except Exception as e:
        logger.error(f"Database init failed: {e}")
        raise


def check_sqlite_health():
    try:
        conn = get_db()
        conn.execute("SELECT 1")
        conn.close()
        logger.info("Database health check passed")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
