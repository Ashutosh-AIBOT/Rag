# Shared 01 — File Structure & Growth Rules

**File identity:** Repo layout — directly from the assignment's expected
project structure.

---

## Repo Structure
```
advanced-rag-platform/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI entry
│   │   ├── config.py                  # Settings
│   │   ├── routers/
│   │   │   ├── documents.py
│   │   │   ├── query.py
│   │   │   └── evaluation.py
│   │   ├── services/
│   │   │   ├── ingestion.py
│   │   │   ├── chunking_strategies.py
│   │   │   ├── retrieval.py
│   │   │   ├── reranker.py
│   │   │   ├── query_transform.py
│   │   │   ├── rag_chain.py
│   │   │   ├── evaluator.py
│   │   │   └── pipeline_tracer.py
│   │   ├── models/
│   │   │   ├── database.py            # SQLAlchemy models
│   │   │   └── schemas.py             # Pydantic models
│   │   └── data/
│   │       └── eval_dataset.json
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx               # Query page
│   │   │   ├── documents/page.tsx
│   │   │   ├── compare/page.tsx
│   │   │   └── evaluate/page.tsx
│   │   └── components/                # (9 components, see frontend/02)
│   └── package.json
├── sample_documents/                  # 3-5 test docs
├── eval_dataset/                      # 15+ Q&A pairs
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

## Env Files
- `.env.example` committed at repo root, listing all required keys
  (`GOOGLE_API_KEY`, `GROQ_API_KEY`, `COHERE_API_KEY`).
- `.env` gitignored, never committed.

## Naming Convention
- Services: `<concern>.py` (e.g. `reranker.py`, `evaluator.py`) — matches
  the fixed structure above, no per-mini-app naming pattern needed since
  this is a single application, not a multi-app platform.

## Growth Rules
- **Adding a new retrieval strategy:** implement in `retrieval.py` or
  `query_transform.py`, register it in the strategy list returned by
  `GET /api/strategies` — the frontend `StrategySelector` picks it up
  automatically (see `frontend/02-features-and-pages.md`). Nothing else
  changes.
- **What must never be duplicated:** LLM provider logic — only
  `rag_chain.py` should ever call the LLM client directly.
