# Backend 04 — Service Architecture & Concurrency Rules

**File identity:** How a request moves through the backend, and — most
important for this project — exactly how concurrency safety is enforced
at the code level.

---

## Layers
- **Chain:** router → service → (LangChain retriever/chain) → response
  model. Simpler than a full CRUD layer stack since there's no
  ownership/tenant logic to enforce.
- **Router rule:** routers only validate input (Pydantic) and call a
  service function — no LangChain objects instantiated directly in a
  router.
- **Service rule:** all retrieval/chunking/reranking/evaluation logic
  lives in `services/`, one file per concern (matches the assignment's
  expected project structure).

## Service File Responsibilities
- `ingestion.py` — document loading & chunking orchestration
- `chunking_strategies.py` — all chunking implementations (recursive,
  parent-child, section-based; semantic chunking if built)
  - Semantic chunking (Good-to-Have): `SemanticChunker` — splits based
    on embedding similarity threshold between consecutive sentences
- `retrieval.py` — hybrid search, retriever construction, configurable
  hybrid weights (Good-to-Have: allow semantic vs BM25 ratio adjustment)
- `reranker.py` — cross-encoder re-ranking, with automatic fallback
  between local cross-encoder and Cohere Rerank API
- `query_transform.py` — multi-query, decomposition, step-back, HyDE
  - Query decomposition (Good-to-Have): break complex questions into
    sub-queries, retrieve for each, synthesize
  - Step-back prompting (Good-to-Have): generate broader question first,
    retrieve for both original + step-back
- `rag_chain.py` — LCEL RAG chains, the only place that knows the active
  LLM provider
- `evaluator.py` — RAG evaluation pipeline (faithfulness, relevancy,
  context precision, context recall)
  - Context precision + recall (Good-to-Have): rank-aware metrics
    requiring reference answers
  - Failure analysis (Good-to-Have): identify queries scoring below
    threshold, attach pipeline trace for debugging
- `pipeline_tracer.py` — logs every pipeline step with timing and scores,
  token accounting (Good-to-Have: count input/output tokens per query,
  break down by source chunk)

## Concurrency Rules (the core of this file)

- **Models loaded once at startup, never per-request.** Embedding model,
  cross-encoder, and BM25 index are instantiated in the FastAPI lifespan
  function and stored on `app.state`. Every service function receives
  them as arguments or reads them from `app.state` — never
  re-instantiates.
- **CPU-bound sync calls run in threadpool.** BM25 scoring and
  cross-encoder inference are synchronous and CPU-bound — every call site
  wraps them in `run_in_threadpool` so the async event loop keeps serving
  other users' requests while one user's re-ranking runs.
- **No shared mutable retriever state.** A filtered retriever (metadata
  filters applied) is built **fresh per request** from the shared
  ChromaDB collection. Never mutate a shared retriever object's filter
  attribute in place — two concurrent requests with different filters
  would overwrite each other's filter mid-flight.
- **LLM calls gated by a semaphore.** `asyncio.Semaphore(N)` wraps every
  LLM call (generation, HyDE, query expansion, LLM-as-judge) so a batch
  evaluation run doesn't starve concurrent live user queries, and so
  concurrent users don't collectively exceed the provider's free-tier
  rate limit.
- **Background tasks for ingestion.** Document upload returns
  immediately; chunking + embedding runs via `BackgroundTasks` so one
  user's large PDF doesn't block others' queries.

## Error Handling
- Errors raised in services as typed exceptions
  (`DocumentNotFoundError`, `StrategyInvalidError`, etc.), caught by a
  single FastAPI exception handler that maps them to the API error
  envelope — errors are never silently swallowed inside a service.
