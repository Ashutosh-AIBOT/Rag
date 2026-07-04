# 00 — Project Overview

**File identity:** Entry point. Answers the 6 gating questions for this
specific project — Advanced RAG Platform (Assignment #7).

---

## Project Identity
- **Name:** Advanced RAG Platform
- **One-line description:** Production-grade retrieval system with hybrid
  search, cross-encoder re-ranking, multi-strategy retrieval, and a RAG
  evaluation pipeline — built with FastAPI + LangChain + Next.js.
- **Who is this for:** Anyone uploading documents and querying them —
  single-user-facing demo/assignment, not a multi-tenant SaaS.
- **Scope:** Single application, not a multi-app platform — several
  sections below are marked N/A where the multi-tenant/multi-app template
  doesn't apply to a scoped 4-day project.

## The 6 Foundational Questions

### 1. Tenancy model
- **Chosen:** N/A — single-tenant. One shared SQLite DB + one ChromaDB
  instance, no tenant_id, no isolation needed.
- **Why:** This is a single-application assignment, not a multi-org
  platform. Building tenancy in would be scope creep against the 4-day
  timeline.

### 2. Shared vs per-app boundary
- **N/A** — one application. The equivalent boundary that matters here is
  **service vs. route**: retrieval/reranking/evaluation logic lives in
  `services/`, and is never duplicated across routers.

### 3. Security model
- **Auth method:** None required for the assignment demo. If exposing
  publicly, add a single shared API key checked via FastAPI dependency —
  not full user auth (out of scope for a 4-day technical assignment).
- **Role hierarchy:** N/A — single user role.
- **Ownership-check rule:** N/A — no per-user data ownership boundary.

### 4. Scaling model
- **Chosen:** Vertical + concurrency-safe design. This is the one question
  that matters most for "more people can use it concurrently":
  - All ML models (embedding model, cross-encoder, BM25 index) loaded
    **once** at app startup, stored on `app.state` — never re-instantiated
    per request.
  - CPU-bound sync calls (BM25 scoring, cross-encoder inference) wrapped in
    `run_in_threadpool` so they don't block the async event loop.
  - `asyncio.Semaphore` around LLM calls so concurrent users don't blow
    past provider rate limits together.
  - `uvicorn --workers N` once state-loading is correct, for real
    horizontal capacity on one machine.
- **Expected concurrent users:** 5-10 for demo/grading purposes; design
  should not silently degrade past that, but doesn't need to survive
  thousands.

### 5. Release model
- **Environments:** local (dev) → demo/submission build. No staging/prod
  split needed for a 4-day assignment.
- **Deploy gate:** eval dataset must run clean (no crashes) + a manual
  concurrent-query smoke test (Stage 8 in the staging plan) before calling
  it submission-ready.
- **Uptime/SLA target:** N/A — not a production service with customers.

### 6. Cost model
- **Which services are paid at launch:** None — everything on free tiers
  (local embeddings via sentence-transformers, Gemini/Groq free tier LLM,
  Cohere Rerank free tier as fallback, ChromaDB/SQLite self-hosted local).
- **Hard monthly budget cap:** $0 target.
- **What happens if a free tier is exceeded:** switch to the local
  self-hosted alternative already built as a fallback (e.g. local
  cross-encoder instead of Cohere Rerank).

---

## Compliance & Data Governance
- **N/A** — no real user data, no retention/deletion requirements for an
  assignment submission with sample documents.

## Plan Governance
- **Who approves plan changes:** self (solo project) — but this file
  should still be updated if the actual build deviates from the staged
  plan, so the README accurately reflects what was built.

---

## Feature Priority Tracker (Assignment Section 8)

### Must-Have (Core — Required) — 15 Items
- [ ] 1. LangChain-based document ingestion with at least 3 chunking strategies (recursive, parent-child, section-based)
- [ ] 2. Vector store (ChromaDB) with metadata storage per chunk (source, page, strategy, section)
- [ ] 3. Hybrid search combining semantic (vector) + keyword (BM25) with EnsembleRetriever
- [ ] 4. Reciprocal Rank Fusion (RRF) for merging hybrid search results
- [ ] 5. Cross-encoder re-ranking of top-20 results down to top-5 (sentence-transformers or Cohere)
- [ ] 6. Multi-Query retrieval: LLM generates query variants, merge results (MultiQueryRetriever)
- [ ] 7. HyDE implementation: generate hypothetical answer, embed, search
- [ ] 8. Metadata filtering: filter by source, page range, chunk strategy, user tags
- [ ] 9. RAG chain built with LangChain LCEL (not legacy chain classes)
- [ ] 10. Pipeline transparency: show retrieved chunks, scores, re-rank positions for every query
- [ ] 11. A/B strategy comparison: same query, two strategies, side-by-side results
- [ ] 12. Basic RAG evaluation: faithfulness and answer relevancy scoring (LLM-as-judge)
- [ ] 13. Evaluation dataset with at least 15 question-answer pairs included in repo
- [ ] 14. Next.js frontend with query UI, chunk inspector, strategy selector, and pipeline visualizer
- [ ] 15. Streaming LLM responses via SSE

### Good-to-Have (Intermediate) — 10 Items
- [ ] 1. Semantic chunking (split based on embedding similarity thresholds)
- [ ] 2. Query decomposition for complex multi-part questions
- [ ] 3. Step-back prompting for specific queries
- [ ] 4. Context precision and context recall evaluation metrics
- [ ] 5. Evaluation dashboard with charts comparing all strategies across all metrics
- [ ] 6. Failure analysis: identify and explain queries where RAG failed
- [ ] 7. Token accounting: show input/output tokens, cost breakdown per query
- [ ] 8. Configurable hybrid search weights (semantic vs. BM25 ratio)
- [ ] 9. Chunk overlap visualization: show how chunks overlap in the source document
- [ ] 10. Query history with ability to re-run and compare results over time

### Bonus (Advanced) — 8 Items
- [ ] 1. RAGAS library integration for standardized evaluation metrics
- [ ] 2. Contextual compression: LLM extracts only relevant sentences from chunks before main LLM
- [ ] 3. Multi-vector retrieval: summary + questions per chunk, retrieve by multiple representations
- [ ] 4. Auto-strategy selection: based on query type, auto-pick best strategy
- [ ] 5. LangSmith integration for tracing (or custom tracing with similar UI)
- [ ] 6. Embeddings comparison: different embedding models, compare retrieval quality
- [ ] 7. Docker + docker-compose deployment
- [ ] 8. Evaluation CI pipeline: run eval dataset on every code change, fail if metrics drop

### Evaluation Criteria Mapping
| Criteria | Weight | Features That Score Here |
|---|---|---|
| RAG Quality | 30% | Must #3-7 (hybrid, RRF, rerank, multi-query, HyDE) |
| LangChain Usage | 20% | Must #1, #9 (LCEL, proper retrievers/splitters) |
| Evaluation Pipeline | 15% | Must #12-13, Good #4-6 (metrics, dataset, comparison) |
| Code Quality | 15% | All backend services (modular, typed, clean) |
| UI/UX | 10% | Must #10-11, #14 (pipeline viz, A/B, chunk inspector) |
| Documentation | 5% | README with arch diagram + eval results table |
| Bonus | 5% | Bonus #1-8 |

