"""
SQLAlchemy models for document metadata, query history, pipeline traces,
evaluation results, and user authentication. Chunk *content + embeddings*
live in ChromaDB; this DB stores structured metadata used for the UI /
filtering / analytics.
"""
import uuid
import datetime as dt
from sqlalchemy import (
    create_engine, Column, String, Integer, Float, DateTime, Text, JSON, Boolean, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import get_settings

settings = get_settings()
engine = create_engine(
    f"sqlite:///{settings.sqlite_db_path}", connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _utcnow():
    return dt.datetime.now(dt.timezone.utc)
Base = declarative_base()


def gen_id() -> str:
    return uuid.uuid4().hex[:16]


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=gen_id)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Integer, default=0)
    total_pages = Column(Integer, default=0)
    tags = Column(JSON, default=list)
    upload_date = Column(DateTime, default=_utcnow)
    doc_date = Column(String, nullable=True)
    chunk_counts = Column(JSON, default=dict)
    status = Column(String, default="processing")
    file_path = Column(String, nullable=True)
    user_id = Column(String, ForeignKey("users.id"), index=True, nullable=True)


class User(Base):
    """User account model for authentication. Replaces in-memory _users_db."""
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=gen_id)
    email = Column(String, unique=True, nullable=False, index=True)
    full_name = Column(String, default="")
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(String, primary_key=True, default=gen_id)
    query = Column(Text, nullable=False)
    strategy = Column(String, nullable=False)
    filters = Column(JSON, default=dict)
    answer = Column(Text)
    trace = Column(JSON, default=dict)
    chunks = Column(JSON, default=list)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    latency_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=_utcnow)
    user_id = Column(String, ForeignKey("users.id"), index=True, nullable=True)


class EvalResult(Base):
    __tablename__ = "eval_results"

    id = Column(String, primary_key=True, default=gen_id)
    question = Column(Text, nullable=False)
    reference_answer = Column(Text)
    strategy = Column(String, nullable=False)
    generated_answer = Column(Text)
    faithfulness = Column(Float, default=0.0)
    answer_relevancy = Column(Float, default=0.0)
    context_precision = Column(Float, default=0.0)
    context_recall = Column(Float, default=0.0)
    passed = Column(Boolean, default=False)
    trace = Column(JSON, default=dict)
    batch_id = Column(String, index=True, default="")
    created_at = Column(DateTime, default=_utcnow)
    user_id = Column(String, ForeignKey("users.id"), index=True, nullable=True)


class IngestionJob(Base):
    """Tracks a background document-ingestion job so the upload endpoint can
    return immediately (202) and the frontend can poll progress instead of
    holding a request open / freezing the UI while chunking + embedding a
    large PDF runs on a worker thread."""
    __tablename__ = "ingestion_jobs"

    id = Column(String, primary_key=True, default=gen_id)
    doc_id = Column(String, index=True, nullable=False)
    filename = Column(String, nullable=False)
    status = Column(String, default="queued")
    progress = Column(Integer, default=0)
    message = Column(String, default="")
    error = Column(Text, nullable=True)
    result = Column(JSON, default=dict)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow)
    user_id = Column(String, ForeignKey("users.id"), index=True, nullable=True)


class EvalJob(Base):
    """Tracks a background batch-evaluation job (can take minutes across
    many strategies x many questions x 4 LLM-judge calls each)."""
    __tablename__ = "eval_jobs"

    id = Column(String, primary_key=True, default=gen_id)
    batch_id = Column(String, index=True, nullable=False)
    status = Column(String, default="queued")
    total_steps = Column(Integer, default=0)
    completed_steps = Column(Integer, default=0)
    strategies = Column(JSON, default=list)
    error = Column(Text, nullable=True)
    summary = Column(JSON, default=dict)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow)
    user_id = Column(String, ForeignKey("users.id"), index=True, nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
