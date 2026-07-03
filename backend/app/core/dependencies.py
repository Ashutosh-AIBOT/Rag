from fastapi import Request

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def get_embeddings(request: Request):
    return request.app.state.embeddings


def get_chroma_store(request: Request):
    return request.app.state.chroma_store


def get_llm_manager(request: Request):
    return request.app.state.llm_manager


def get_semaphore(request: Request):
    return request.app.state.semaphore
