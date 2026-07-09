from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator
import datetime as dt


# ---------- Documents ----------

class DocumentOut(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    total_pages: int
    tags: List[str] = []
    upload_date: dt.datetime
    doc_date: Optional[str] = None
    chunk_counts: Dict[str, int] = {}
    status: str
    user_id: Optional[str] = None

    model_config = {"from_attributes": True}


# ---------- Filters ----------

class MetadataFilters(BaseModel):
    source: Optional[Any] = None
    page_min: Optional[int] = None
    page_max: Optional[int] = None
    section: Optional[str] = None
    tags: Optional[List[str]] = None
    strategy: Optional[str] = None  # recursive | semantic | parent_child | section
    date_from: Optional[str] = None  # ISO date string, e.g. "2025-01-01" (matches upload date)
    date_to: Optional[str] = None
    filter_logic: str = "and"  # "and" | "or" -- how source/section/tags/page/date combine


# ---------- Query ----------

RETRIEVAL_STRATEGIES = [
    "basic_vector",
    "hybrid",
    "hybrid_rerank",
    "parent_child",
    "multi_query",
    "hyde",
    "decomposition",
    "step_back",
    "auto",
    "multi_vector",  # Bonus: retrieves by summaries + hypothetical questions
    "section_search",  # Searches section-based chunks (one chunk per H1/H2 section)
]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    strategy: Literal["basic_vector", "hybrid", "hybrid_rerank", "parent_child", "multi_query", "hyde", "decomposition", "step_back", "auto", "multi_vector", "section_search"] = "hybrid_rerank"
    filters: MetadataFilters = Field(default_factory=MetadataFilters)
    top_k_initial: int = Field(default=20, ge=1, le=100)
    top_k_final: int = Field(default=5, ge=1, le=50)
    semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    stream: bool = False
    compress_context: bool = False

    @field_validator("top_k_final")
    @classmethod
    def final_le_initial(cls, v, info):
        if "top_k_initial" in info.data and v > info.data["top_k_initial"]:
            raise ValueError("top_k_final must be <= top_k_initial")
        return v


class ChunkScore(BaseModel):
    chunk_id: str
    text: str
    source: str
    page: Optional[int] = None
    section: Optional[str] = None
    strategy: str
    semantic_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    original_rank: Optional[int] = None
    final_rank: Optional[int] = None
    child_text: Optional[str] = None
    token_count: Optional[int] = None


class PipelineStep(BaseModel):
    name: str
    detail: Dict[str, Any] = {}
    duration_ms: float = 0.0


class QueryResponse(BaseModel):
    query_id: str
    query: str
    strategy: str
    answer: str
    chunks: List[ChunkScore]
    pipeline: List[PipelineStep]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    estimated_cost_usd: float = 0.0
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None


class CompareRequest(BaseModel):
    query: str
    strategy_a: str
    strategy_b: str
    filters: MetadataFilters = Field(default_factory=MetadataFilters)
    score_quality: bool = True  # run the LLM-judge faithfulness/relevancy scorer for both sides


class CompareResponse(BaseModel):
    query: str
    result_a: QueryResponse
    result_b: QueryResponse
    overlap_chunk_ids: List[str]


class DocumentChunkOut(BaseModel):
    chunk_id: str
    text: str
    page: Optional[int] = None
    section: Optional[str] = None
    strategy: str


class DocumentChunksOut(BaseModel):
    doc_id: str
    filename: str
    by_strategy: Dict[str, List[DocumentChunkOut]]


# ---------- Evaluation ----------

class EvalItem(BaseModel):
    question: str
    reference_answer: str
    relevant_sources: List[str] = []


class EvalRequest(BaseModel):
    question: str
    reference_answer: str
    strategy: str = "hybrid_rerank"
    relevant_sources: List[str] = []
    use_ragas: bool = False


class EvalResultOut(BaseModel):
    id: str
    question: str
    strategy: str
    reference_answer: str
    generated_answer: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    passed: bool
    trace: Dict[str, Any] = {}
    user_id: Optional[str] = None

    model_config = {"from_attributes": True}


class BatchEvalRequest(BaseModel):
    strategies: List[Literal["basic_vector", "hybrid", "hybrid_rerank", "parent_child", "multi_query", "hyde", "decomposition", "step_back", "auto", "multi_vector", "section_search"]] = ["basic_vector", "hybrid", "hybrid_rerank", "parent_child", "hyde"]
    limit: Optional[int] = None
    use_ragas: bool = False


# ---------- Background jobs ----------

class IngestionJobOut(BaseModel):
    id: str
    doc_id: str
    filename: str
    status: str
    progress: int
    message: str
    error: Optional[str] = None
    result: Dict[str, Any] = {}
    user_id: Optional[str] = None

    model_config = {"from_attributes": True}


class UploadAccepted(BaseModel):
    document: DocumentOut
    job_id: str


class QueryLogOut(BaseModel):
    id: str
    query: str
    strategy: str
    filters: Dict[str, Any] = {}
    answer: str
    trace: List[Dict[str, Any]] = []
    chunks: List[ChunkScore] = []
    input_tokens: int
    output_tokens: int
    latency_ms: float
    created_at: dt.datetime
    user_id: Optional[str] = None

    model_config = {"from_attributes": True}

    @classmethod
    def model_validate(cls, obj, **kwargs):
        """Override to handle SQLAlchemy ORM rows where `chunks` is a raw
        JSON list[dict] (not already ChunkScore objects). Pydantic v2
        from_attributes does not auto-coerce nested JSON column values."""
        if hasattr(obj, "__dict__") or hasattr(obj, "_sa_instance_state"):
            # Convert ORM row to dict for coercion
            data = {
                "id": obj.id,
                "query": obj.query,
                "strategy": obj.strategy,
                "filters": obj.filters or {},
                "answer": obj.answer or "",
                "trace": obj.trace or [],
                "chunks": [
                    ChunkScore(**c) if isinstance(c, dict) else c
                    for c in (obj.chunks or [])
                ],
                "input_tokens": obj.input_tokens or 0,
                "output_tokens": obj.output_tokens or 0,
                "latency_ms": obj.latency_ms or 0.0,
                "created_at": obj.created_at,
                "user_id": getattr(obj, "user_id", None),
            }
            return cls(**data)
        return super().model_validate(obj, **kwargs)


class EvalJobOut(BaseModel):
    id: str
    batch_id: str
    status: str
    total_steps: int
    completed_steps: int
    strategies: List[str] = []
    error: Optional[str] = None
    summary: Dict[str, Any] = {}
    created_at: Optional[dt.datetime] = None
    user_id: Optional[str] = None

    model_config = {"from_attributes": True}

    @classmethod
    def model_validate(cls, obj, **kwargs):
        if hasattr(obj, "__dict__") or hasattr(obj, "_sa_instance_state"):
            data = {
                "id": obj.id,
                "batch_id": obj.batch_id,
                "status": obj.status,
                "total_steps": obj.total_steps,
                "completed_steps": obj.completed_steps,
                "strategies": obj.strategies or [],
                "error": obj.error,
                "summary": obj.summary or {},
                "created_at": obj.created_at,
                "user_id": getattr(obj, "user_id", None),
            }
            return cls(**data)
        return super().model_validate(obj, **kwargs)


class BatchEvalAccepted(BaseModel):
    job_id: str
    batch_id: str
