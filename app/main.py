import logging
import logging.config
import threading
import time
import uuid
from contextvars import ContextVar

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings
from app.models.database import init_db
from app.routers import documents, query, evaluation, misc, auth
from app.services import job_manager
from app.services.cache import init_cache

_cid_var: ContextVar[str] = ContextVar("correlation_id", default="-")

class CustomGZipMiddleware(GZipMiddleware):
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            matched = "/query/stream" in path or "/logs" in path
            logger.info("CustomGZipMiddleware: path=%s matched=%s", path, matched)
            if matched:
                await self.app(scope, receive, send)
                return
        await super().__call__(scope, receive, send)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": "INFO",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}

logging.config.dictConfig(LOGGING_CONFIG)


class CorrelationIdFilter(logging.Filter):
    """Inject correlation_id into every log record using ContextVar (thread-safe)."""
    def filter(self, record):
        record.correlation_id = _cid_var.get("-")
        return True


_correlation_filter = CorrelationIdFilter()
logging.getLogger().addFilter(_correlation_filter)

logger = logging.getLogger("main")

settings = get_settings()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        init_cache("redis://localhost:6379/0")
    except Exception:
        logger.warning("Redis not available, using in-memory cache")
    logger.info(
        "Startup complete. max_concurrent_queries=%s ingestion_workers=%s eval_workers=%s "
        "llm_provider=%s fallback_order=%s langsmith_tracing=%s",
        settings.max_concurrent_queries,
        settings.ingestion_worker_threads,
        settings.eval_worker_threads,
        settings.llm_provider,
        settings.llm_fallback_order,
        settings.langchain_tracing_v2,
    )
    yield
    job_manager.shutdown()


app = FastAPI(
    title="Advanced RAG Platform",
    description="Production-grade RAG: hybrid search, cross-encoder re-ranking, "
                "parent-child chunking, query transformation, RAG evaluation.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(CustomGZipMiddleware, minimum_size=1000)

_cors_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
for origin in ("http://localhost:3000", "http://localhost:3001"):
    if origin not in _cors_origins:
        _cors_origins.append(origin)
logger.info("CORS allowed origins: %s", _cors_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


from starlette.datastructures import MutableHeaders

class ProductionASGIStateMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        import uuid
        import time
        from starlette.datastructures import Headers

        headers = Headers(scope=scope)
        correlation_id = headers.get("x-correlation-id", str(uuid.uuid4())[:12])
        _cid_var.set(correlation_id)

        path = scope.get("path", "")
        method = scope.get("method", "")
        client = scope.get("client")
        client_ip = f"{client[0]}:{client[1]}" if client else "unknown"
        
        user_agent = "-"
        for k, v in scope.get("headers", []):
            if k == b"user-agent":
                user_agent = v.decode("utf-8", errors="ignore")
                break

        logger.info(
            "[%s] REQUEST  %s %s  client=%s  user_agent=%s",
            correlation_id, method, path, client_ip, user_agent[:80],
        )

        start_time = time.perf_counter()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                res_headers = MutableHeaders(scope=message)
                res_headers["X-Content-Type-Options"] = "nosniff"
                res_headers["X-Frame-Options"] = "DENY"
                res_headers["X-XSS-Protection"] = "1; mode=block"
                res_headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
                if scope.get("scheme") == "https":
                    res_headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
                
                res_headers["X-Correlation-ID"] = correlation_id
                res_headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"

                status_code = message.get("status", 200)
                logger.info(
                    "[%s] RESPONSE  %s %s  status=%d  elapsed=%.1fms",
                    correlation_id, method, path, status_code, elapsed_ms,
                )
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "[%s] ERROR     %s %s  status=500  elapsed=%.1fms  error=%s",
                correlation_id, method, path, elapsed_ms, str(exc)[:200],
            )
            raise

app.add_middleware(ProductionASGIStateMiddleware)


app.include_router(auth.router)
app.include_router(documents.router, dependencies=[Depends(auth._get_current_user)])
app.include_router(query.router, dependencies=[Depends(auth._get_current_user)])
app.include_router(evaluation.router, dependencies=[Depends(auth._get_current_user)])
app.include_router(misc.router)
