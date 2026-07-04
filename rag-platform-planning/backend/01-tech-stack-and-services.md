# Backend 01 — Tech Stack & Third-Party Services

**File identity:** Every dependency and vendor the backend relies on.

---

## Core Stack

- **Language / runtime:** Python 3.11+
- **Framework:** FastAPI (async)
- **RAG framework:** LangChain (`langchain`, `langchain-community`,
  `langchain-openai` or `langchain-google-genai`) — version >= 0.2, modular
  package structure. No deprecated imports (`LLMChain`, `SequentialChain`).
- **Chain construction:** LCEL (LangChain Expression Language) — all chains
  composable/declarative, not legacy chain classes.
- **Package manager:** pip + `requirements.txt`
- **Why this stack:** matches assignment spec exactly; LangChain gives
  battle-tested retrievers (`EnsembleRetriever`, `MultiQueryRetriever`,
  `ParentDocumentRetriever`) instead of reinventing them.

## Third-Party Services

| Service                                               | Purpose                                         | Free tier               | Becomes paid at      | Cheaper alt for early stage                                   | Swappable?                             | Key storage                                |
| ----------------------------------------------------- | ----------------------------------------------- | ----------------------- | -------------------- | ------------------------------------------------------------- | -------------------------------------- | ------------------------------------------ |
| Vector DB — ChromaDB                                  | Store chunks + embeddings                       | Fully free, self-hosted | N/A (local)          | —                                                             | Yes — abstracted behind `retrieval.py` | N/A, no key                                |
| Embeddings — sentence-transformers `all-MiniLM-L6-v2` | Dense vector embeddings                         | Free, local, no key     | N/A                  | OpenAI `text-embedding-3-small` if local quality insufficient | Yes                                    | N/A                                        |
| BM25 — `rank_bm25`                                    | Sparse/lexical search                           | Free, local library     | N/A                  | —                                                             | N/A, always local                      | N/A                                        |
| Re-ranker — CrossEncoder `ms-marco-MiniLM-L-6-v2`     | Re-rank top-20→top-5                            | Free, local             | N/A                  | Cohere Rerank API if local model too slow                     | Yes                                    | `.env` if Cohere used                      |
| Re-ranker (alt) — Cohere Rerank                       | Easier setup alternative                        | Free tier               | Past free tier limit | Fall back to local cross-encoder                              | Yes                                    | `.env` — `COHERE_API_KEY`                  |
| LLM — Gemini / Groq                                   | Generation, HyDE, query transform, LLM-as-judge | Free tier on both       | Past free tier limit | Switch provider via abstraction                               | Yes — behind `rag_chain.py`            | `.env` — `GOOGLE_API_KEY` / `GROQ_API_KEY` |
| DB — SQLite                                           | Query history, eval results, doc metadata       | Free, local file        | N/A                  | Postgres if concurrent-write issues appear                    | Yes, but migration effort              | N/A                                        |

## Fallback & Swap Rules

- **AI client abstraction:** `services/rag_chain.py` is the only place that
  knows which LLM provider is active — swapping Gemini↔Groq↔OpenAI never
  touches router code.
- **Re-ranker fallback:** if Cohere free tier is exhausted mid-demo, fall
  back automatically to the local cross-encoder (already loaded at
  startup per the staging plan Stage 0) — do not let a rate limit break
  the demo.
