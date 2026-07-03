from datetime import datetime
from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    total_pages: int
    upload_date: str
    tags: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


class QueryRequest(BaseModel):
    query: str
    strategy: str = "recursive"
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    source_chunks: list[dict]
    provider: str
    model: str
