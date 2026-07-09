"""
Concurrency control for the whole backend.

Three independent bounded pools so one kind of load never starves another:

  - `query_semaphore`      caps concurrent end-to-end RAG queries (LLM calls +
                           embeddings + reranking) -- this is the "5-10
                           concurrent users" knob. Extra requests queue
                           in-process (asyncio.Semaphore) instead of being
                           rejected, so the frontend just sees slightly
                           higher latency under load rather than errors.
  - `ingestion_executor`   a small dedicated ThreadPoolExecutor for document
                           ingestion (loading, chunking, embedding). Kept
                           separate from FastAPI's default threadpool so a
                           burst of uploads can't starve query handling.
  - `eval_executor`        same idea for batch evaluation jobs (many LLM
                           calls back-to-back; can run for minutes).

FastAPI route handlers stay responsive (never blocked) because:
  1. Uploads / batch-eval submit work to these executors via
     `run_in_executor` and return a job_id immediately (202 Accepted).
  2. The frontend polls a lightweight `GET .../jobs/{id}` status endpoint.
  3. `/api/query` (fast, seconds not minutes) stays synchronous-in-threadpool
     but is still gated by `query_semaphore` so a spike of concurrent users
     doesn't overwhelm free-tier LLM rate limits all at once.
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from app.config import get_settings

logger = logging.getLogger("job_manager")
settings = get_settings()

# ---- bounded worker pools -------------------------------------------------
logger.info("Initializing ThreadPoolExecutors: Ingestion workers=%d, Eval workers=%d", settings.ingestion_worker_threads, settings.eval_worker_threads)
ingestion_executor = ThreadPoolExecutor(
    max_workers=settings.ingestion_worker_threads, thread_name_prefix="ingest"
)
eval_executor = ThreadPoolExecutor(
    max_workers=settings.eval_worker_threads, thread_name_prefix="eval"
)

# asyncio.Semaphore must be created lazily inside a running event loop in
# some environments; a module-level Semaphore works fine on modern asyncio
# but we guard construction to be safe under uvicorn's loop.
_query_semaphore: asyncio.Semaphore | None = None


def get_query_semaphore() -> asyncio.Semaphore:
    global _query_semaphore
    if _query_semaphore is None:
        logger.info("get_query_semaphore: Creating asyncio.Semaphore with limit=%d", settings.max_concurrent_queries)
        _query_semaphore = asyncio.Semaphore(settings.max_concurrent_queries)
    return _query_semaphore


def submit_ingestion(fn, *args, **kwargs):
    """Fire-and-forget a blocking function onto the ingestion pool."""
    logger.info("submit_ingestion: Submitting ingestion job to executor pool.")
    return ingestion_executor.submit(fn, *args, **kwargs)


def submit_eval(fn, *args, **kwargs):
    """Fire-and-forget a blocking function onto the eval pool."""
    logger.info("submit_eval: Submitting evaluation job to executor pool.")
    return eval_executor.submit(fn, *args, **kwargs)


def shutdown():
    logger.info("shutdown: Shutting down job executors.")
    ingestion_executor.shutdown(wait=False, cancel_futures=False)
    eval_executor.shutdown(wait=False, cancel_futures=False)
