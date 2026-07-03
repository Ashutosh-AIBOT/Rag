# Shared 03 — AI/ML Model & Tool Selection

**File identity:** Every place this platform calls an AI/ML model — this
project is almost entirely AI tasks, so this file is central.

---

## Per-Task Decisions

| Task | Provider/Model | Cost | Cacheable? | Local option? |
|---|---|---|---|---|
| Dense embeddings | sentence-transformers `all-MiniLM-L6-v2` | Free | Yes — embeddings cached per chunk at ingestion time, not recomputed per query | Yes, this IS the local option |
| Sparse/keyword search | `rank_bm25` | Free | Index built once at ingestion, reused per query | Yes, local library |
| Re-ranking | CrossEncoder `ms-marco-MiniLM-L-6-v2` (local) / Cohere Rerank (fallback) | Free | No — depends on live query, not cached | Yes, local is primary choice |
| Answer generation | Gemini or Groq (free tier) | Free | No — live generation | No — cloud LLM required for quality |
| Multi-query expansion | Same LLM as generation | Free | Could cache per unique query text | No |
| HyDE hypothetical answer | Same LLM as generation | Free | Could cache per unique query text | No |
| Query decomposition | Same LLM as generation | Free | No | No |
| LLM-as-judge (evaluation) | Same LLM as generation | Free | No — needs fresh judgment per answer | No |

## Cost Optimization Rules
- Use the local embedding + local cross-encoder path by default — only
  the generation/judging calls actually need a cloud LLM, which keeps
  free-tier usage minimal.
- Cache query embeddings within a single A/B comparison run (same query
  embedded once, reused across strategies being compared), rather than
  re-embedding per strategy.
- Batch evaluation runs sequentially with the semaphore from
  `backend/04-service-architecture.md` rather than firing all 15-30 eval
  calls simultaneously — protects the free-tier rate limit.

## Fallback
- **Fallback provider:** Gemini ↔ Groq, swappable via `rag_chain.py`.
- **Re-ranker fallback:** local cross-encoder ↔ Cohere Rerank, swappable
  via `reranker.py` — if one is rate-limited or unavailable, the other
  takes over without touching any router or chain code.
