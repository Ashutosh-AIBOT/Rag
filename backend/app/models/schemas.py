from pydantic import BaseModel
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    message: str


class QueryRequest(BaseModel):
    question: str
    k: int = 5
    strategy: str = "vector"
    rerank: bool = False
    rerank_top_k: int = 3
    filters: dict = None
    embedding_model: str = "huggingface"
    compress: bool = False


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    sources: list[str]
    trace: dict = None


class IngestionResult(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    duplicate: bool


class ErrorResponse(BaseModel):
    detail: str


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    total_pages: int
    upload_date: str
    tags: str
    chunk_count: int
    status: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
