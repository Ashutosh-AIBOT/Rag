"""
Centralized application settings, loaded from environment variables / .env
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    # "auto" builds a fallback chain: NVIDIA NIM -> Groq -> Gemini (first
    # provider with a configured API key that doesn't error out wins).
    # Explicit values (openai|google|groq|nvidia) bypass the fallback chain.
    llm_provider: str = "auto"
    llm_fallback_order: str = "nvidia,groq,google"  # comma-separated, used when llm_provider=auto

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    google_api_key: str = ""
    google_model: str = "gemini-2.0-flash"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    nvidia_api_key: str = ""
    nvidia_model: str = "meta/llama-3.1-70b-instruct"

    # ---- LangSmith tracing ----
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "advanced-rag-platform"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # ---- Concurrency / background workers ----
    # Caps concurrent heavy operations (LLM calls, embeddings, reranking) so
    # 5-10 simultaneous users don't exhaust free-tier rate limits or CPU.
    max_concurrent_queries: int = 8
    ingestion_worker_threads: int = 4
    eval_worker_threads: int = 2
    eval_sleep_seconds: float = 6.0   # seconds to sleep between eval steps (free-tier safe default)
    eval_max_retries: int = 3          # per-step LLM retries before recording a failure

    # Embeddings
    embedding_provider: str = "local"  # local | openai
    local_embedding_model: str = "all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # Reranker
    reranker_provider: str = "local"  # local | cohere
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cohere_api_key: str = ""

    # Storage
    chroma_persist_dir: str = "./chroma_db"
    sqlite_db_path: str = "./rag_platform.db"
    upload_dir: str = "./uploads"

    # App
    cors_origins: str = "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,http://127.0.0.1:3001"
    app_env: str = "development"
    jwt_secret_key: str = "rag-platform-secret-key-change-in-production"

    # Retrieval defaults
    default_top_k_initial: int = 20
    default_top_k_final: int = 5
    rrf_k: int = 60
    semantic_weight: float = 0.7
    bm25_weight: float = 0.3


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    _apply_langsmith_env(s)
    return s


def _apply_langsmith_env(s: "Settings") -> None:
    """LangSmith auto-instruments any LangChain runnable purely via env vars
    (no code changes needed in chains). We set them once here so every
    ChatModel / retriever / LCEL chain built anywhere in the app gets traced
    automatically as soon as LANGCHAIN_TRACING_V2=true + an API key are set.
    """
    import os

    if s.langchain_tracing_v2 and s.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = s.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = s.langchain_project
        os.environ["LANGCHAIN_ENDPOINT"] = s.langchain_endpoint
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
