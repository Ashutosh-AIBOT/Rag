from pydantic import BaseModel
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    message: str


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


class IngestionResult(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    duplicate: bool


class ErrorResponse(BaseModel):
    detail: str
