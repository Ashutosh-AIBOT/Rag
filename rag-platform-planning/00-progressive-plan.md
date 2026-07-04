# Progressive Plan — Assignment #7: Advanced RAG with LangChain

**Duration:** 4 Days
**Stack:** FastAPI + LangChain + ChromaDB + Next.js

---

## Day 1: Foundation + Ingestion & Chunking

### Stage 0 — Foundation (DONE)
- [x] FastAPI app skeleton with lifespan/startup
- [x] SQLite WAL mode for document tracking
- [x] Embedding model preloaded (`all-MiniLM-L6-v2`)
- [x] LLM fallback chain: Nvidia → Groq → Gemini
- [x] Semaphore for concurrent LLM calls
- [x] `.env.example` with all API keys
- [x] Health check endpoint
- **Milestone:** App boots, health responds, models loaded once

### Stage 1 — Document Ingestion & Basic Chunking
- [ ] Document loaders: PDF, TXT, DOCX, Markdown
- [ ] Metadata enrichment: filename, type, size, pages, upload date, tags
- [ ] RecursiveCharacterTextSplitter (500 tokens, 50 overlap)
- [ ] Upload endpoint → 202 Accepted + background processing
- [ ] Store chunks in ChromaDB with `strategy='recursive'` metadata
- [ ] List documents endpoint
- [ ] Delete document endpoint (ChromaDB + SQLite)
- [ ] Basic query endpoint (vector search + LLM)
- **Milestone:** Upload doc → chunks in ChromaDB → query returns answer
- **Endpoint:** `POST /api/documents/upload`, `GET /api/documents`, `DELETE /api/documents/{id}`, `POST /api/query`

---

## Day 2: Advanced Retrieval & Re-Ranking

### Stage 2 — Advanced Chunking Strategies
- [ ] Semantic Chunking (embedding similarity thresholds)
- [ ] Parent-Child Chunking (200-token children, 1000-token parents)
- [ ] Section-Based Chunking (split by H1/H2 headers)
- [ ] SQLite `parent_documents` table for parent content storage
- [ ] Parent-child swap: retrieve child → fetch parent from DB
- [ ] Chunk preview endpoint (see chunks without saving)
- **Milestone:** Upload doc → 4 strategies running → parent-child swap works
- **Endpoint:** `POST /api/documents/preview`

### Stage 3 — Hybrid Search + BM25 + RRF
- [ ] BM25 retriever using `rank_bm25`
- [ ] `EnsembleRetriever` combining vector + BM25
- [ ] Reciprocal Rank Fusion (RRF) with k=60
- [ ] Configurable weights (semantic vs BM25 ratio)
- [ ] CPU-bound BM25/cross-encoder wrapped in `run_in_threadpool`
- **Milestone:** Same query → vector / BM25 / hybrid results visibly different
- **Endpoint:** `GET /api/strategies`

### Stage 4 — Cross-Encoder Re-Ranking
- [ ] Load cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- [ ] Retrieve top-20 → re-rank → return top-5
- [ ] Score transparency: original rank, re-ranked position, relevance score
- [ ] Hybrid + Rerank as default strategy
- **Milestone:** Show before/after re-rank scores for one real query
- **Endpoint:** Part of `POST /api/query`

---

## Day 3: Query Transformation + Evaluation + Frontend Start

### Stage 5 — Query Transformation
- [ ] Multi-Query Expansion (LLM generates 3-5 variants, merge results)
- [ ] HyDE Chain (generate hypothetical answer → embed → search)
- [ ] Query Decomposition (break complex questions into sub-queries)
- [ ] Step-Back Prompting (generate broader question first)
- [ ] Pipeline tracer: log every step with timing and scores
- **Milestone:** Same query through all transformations → results visibly different
- **Endpoint:** Part of `POST /api/query`, `GET /api/query/{id}/pipeline`

### Stage 6 — RAG Evaluation Pipeline
- [ ] Faithfulness scoring (LLM-as-judge: claims vs context)
- [ ] Answer relevancy scoring (synthetic questions from answer)
- [ ] Context precision (rank-aware relevance)
- [ ] Context recall (coverage vs reference answer)
- [ ] Evaluation dataset: 20-30 Q&A pairs with reference answers
- [ ] Batch evaluation endpoint across all strategies
- [ ] Strategy comparison table with quantitative metrics
- **Milestone:** Comparison table with real numbers for 2+ strategies
- **Endpoint:** `POST /api/evaluate`, `POST /api/evaluate/batch`, `GET /api/evaluate/results`

### Stage 7 — Frontend (Query + Pipeline Visualizer)
- [ ] Next.js project with Tailwind + shadcn/ui
- [ ] Query interface with strategy selector + metadata filters
- [ ] Pipeline visualizer (step-by-step flow)
- [ ] Chunk inspector (click chunk → see all scores)
- [ ] Streaming via SSE
- **Milestone:** Query in browser → streamed answer with visible source chunks

---

## Day 4: A/B Testing + Dashboard + Polish

### Stage 8 — A/B Comparison + Eval Dashboard
- [ ] A/B comparison page: same query, two strategies, side-by-side
- [ ] Chunk comparison with overlap highlighting
- [ ] Evaluation dashboard with Recharts (bar charts for all metrics)
- [ ] Strategy leaderboard (ranked by average score)
- [ ] Failure analysis (queries below threshold with pipeline trace)
- **Milestone:** Demo-ready A/B comparison and eval dashboard

### Stage 9 — Production Hardening
- [ ] Concurrency test: 5-10 simultaneous queries via `asyncio.gather`
- [ ] Rate limiting (per-IP/session)
- [ ] Load test with `uvicorn --workers N`
- [ ] Token accounting (input/output tokens, cost breakdown)
- [ ] Query history with re-run capability
- **Milestone:** Concurrent load test passes, screenshot for README

### Stage 10 — Documentation & Submission
- [ ] README with architecture diagram
- [ ] Strategy explanations
- [ ] Evaluation results table
- [ ] Screenshots of pipeline visualizer + A/B comparison
- [ ] Setup instructions
- [ ] Demo video (5-7 min)
- **Milestone:** Submission ready

---

## API Endpoints (Final)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/documents/upload` | Upload & ingest document |
| GET | `/api/documents` | List ingested documents |
| DELETE | `/api/documents/{id}` | Delete document |
| POST | `/api/documents/preview` | Chunking preview |
| POST | `/api/query` | Query with strategy & filters |
| POST | `/api/query/compare` | A/B test two strategies |
| GET | `/api/query/{id}/pipeline` | Full pipeline trace |
| GET | `/api/query/{id}/chunks` | Retrieved chunks with scores |
| POST | `/api/evaluate` | Evaluate single Q&A pair |
| POST | `/api/evaluate/batch` | Batch evaluation |
| GET | `/api/evaluate/results` | Evaluation results |
| GET | `/api/strategies` | List available strategies |
| GET | `/api/chunks/search` | Direct chunk search (debug) |
| GET | `/api/stats` | Retrieval stats & latency |
| GET | `/api/health` | Health check |

---

## Retrieval Strategies

| Strategy | Component | How It Works |
|----------|-----------|--------------|
| Basic Vector | `Chroma.as_retriever()` | Cosine similarity on embeddings |
| Hybrid | `EnsembleRetriever` | BM25 + Vector via RRF |
| Hybrid + Rerank | `EnsembleRetriever` + Reranker | Hybrid → cross-encoder top-5 |
| Parent-Child | `ParentDocumentRetriever` | Small chunks → parent context |
| Multi-Query | `MultiQueryRetriever` | LLM variants, merge results |
| HyDE | Custom chain | Hypothetical answer → embed → search |
| Decomposition | Custom chain | Break into sub-queries |

---

## Evaluation Metrics

| Metric | What It Measures | Score Range |
|--------|------------------|-------------|
| Faithfulness | Answer supported by context | 0-1 |
| Answer Relevancy | Answer relevant to question | 0-1 |
| Context Precision | Retrieved chunks relevant | 0-1 |
| Context Recall | All needed info found | 0-1 |

---

## Evaluation Criteria Mapping

| Criteria | Weight | Stages |
|----------|--------|--------|
| RAG Quality | 30% | Stage 3, 4, 5 |
| LangChain Usage | 20% | Stage 1, 2, 5, 6 |
| Evaluation Pipeline | 15% | Stage 6 |
| Code Quality | 15% | All stages |
| UI/UX | 10% | Stage 7, 8 |
| Documentation | 5% | Stage 10 |
| Bonus | 5% | Stage 9 |

---

## Feature Priority

### Must-Have (Core) — 15 Items
1. LangChain ingestion with 3+ chunking strategies
2. ChromaDB with metadata storage
3. Hybrid search (semantic + BM25) with EnsembleRetriever
4. Reciprocal Rank Fusion (RRF)
5. Cross-encoder re-ranking top-20 → top-5
6. Multi-Query retrieval
7. HyDE implementation
8. Metadata filtering
9. LCEL RAG chain (not legacy)
10. Pipeline transparency
11. A/B strategy comparison
12. Faithfulness + relevancy scoring
13. Evaluation dataset (15+ Q&A pairs)
14. Next.js frontend with all components
15. Streaming via SSE

### Good-to-Have — 10 Items
1. Semantic chunking
2. Query decomposition
3. Step-back prompting
4. Context precision + recall metrics
5. Eval dashboard with charts
6. Failure analysis
7. Token accounting
8. Configurable hybrid weights
9. Chunk overlap visualization
10. Query history

### Bonus — 8 Items
1. RAGAS integration
2. Contextual compression
3. Multi-vector retrieval
4. Auto-strategy selection
5. LangSmith tracing
6. Embeddings comparison
7. Docker deployment
8. Eval CI pipeline
