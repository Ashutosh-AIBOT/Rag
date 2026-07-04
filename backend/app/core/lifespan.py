from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.logging import get_logger
from app.core.startup import init_sqlite_wal, check_sqlite_health

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[stage00 | lifespan | 012] OK: Startup starting...")
    init_sqlite_wal()

    if not check_sqlite_health():
        raise RuntimeError("SQLite health check failed")

    print("[stage00 | lifespan | 013] OK: App ready")
    yield
    print("[stage00 | lifespan | 014] OK: App stopped")