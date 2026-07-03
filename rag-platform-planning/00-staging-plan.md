# 00 — Staging Plan (Stage-Wise Build Order)

**File identity:** This is the execution roadmap — the order you actually
build things in, day by day, with a milestone and a concurrency check at
every stage. It's deliberately separate from the architecture-decision
files (which say _what_ to build); this file says _in what order_ and
_how you know each stage is actually done_. This is the one file to look
at every morning during the 4-day build.

---

## Why Concurrency Is Decided at Stage 0, Not Later

The most common failure mode in a RAG demo isn't the API layer — it's
loading the embedding model, cross-encoder, or BM25 index **inside a
request handler**. If that happens, every concurrent user reloads the
model and the app grinds to a halt with 2-3 simultaneous users. This gets
fixed once, at the very start, not bolted on at the end.

---

## Stage 0 — Foundation

**Goal:** nothing feature-related yet — just the scaffolding that every
later stage depends on.

- [ ] FastAPI app skeleton with lifespan/startup event
- [ ] Load ALL ML models once at startup, store on `app.state`:
      embedding model, cross-encoder, BM25 index reference
- [ ] `asyncio.Semaphore(N)` around LLM calls
- [ ] SQLite in **WAL mode** (`PRAGMA journal_mode=WAL`) for query
      history/eval results — single-writer lock is fine for this scale,
      WAL avoids write-blocking reads
- [ ] `.env.example` with all required keys
- **Milestone:** app boots, health check responds, models are confirmed
  loaded exactly once (log a line at startup, not per-request)

## Stage 1 — Ingestion & Chunking (MVP slice #1)

**Goal:** a document goes in, chunks come out, correctly tagged.

- [ ] Loaders: PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader,
      UnstructuredMarkdownLoader
- [ ] Metadata enrichment: file name, type, pages, upload date, size, tags
- [ ] Chunking strategies (must-have 3): Recursive, Parent-Child,
      Section-Based
- [ ] Upload endpoint returns `202 Accepted` + doc ID immediately; actual
      chunking/embedding runs via `BackgroundTasks` — a big PDF upload must
      never block other users' requests
- **Milestone:** upload a doc, see chunks land in ChromaDB with correct
  `source`/`page`/`strategy` metadata

## Stage 2 — Retrieval Core (MVP slice #2)

**Goal:** three retrieval strategies, visibly different results.

- [ ] Basic vector retriever (`Chroma.as_retriever()`)
- [ ] BM25 retriever (`rank_bm25`)
- [ ] `EnsembleRetriever` (hybrid + RRF)
- [ ] BM25 scoring and cross-encoder inference wrapped in
      `run_in_threadpool` — both are CPU-bound and synchronous, must not
      block the event loop
- **Milestone:** same query run through vector / BM25 / hybrid, results
  are visibly different, RRF-merged list makes sense

## Stage 3 — Re-Ranking + Query Transformation (MVP slice #3)

**Goal:** measurable precision improvement, the thing that "impresses."

- [ ] Cross-encoder re-rank top-20 → top-5
- [ ] Score transparency: original rank, re-ranked position, relevance
      score all returned to the caller
- [ ] `MultiQueryRetriever`
- [ ] HyDE chain
- [ ] Query decomposition (Good-to-Have): break complex questions into
      sub-queries, retrieve for each, synthesize. E.g. "Compare pricing
      and features of Plan A vs Plan B" → 4 sub-questions.
- [ ] Step-back prompting (Good-to-Have): for specific queries, generate
      a broader "step-back" question first. E.g. "revenue in Q3 2025?"
      → step-back: "financial results for 2025?". Retrieve for both.
- **Milestone:** show before/after re-rank score changes for one real
  query — screenshot this, it goes straight in the README

## Stage 4 — LCEL RAG Chain + Metadata Filtering (MVP slice #4)

**Goal:** one real end-to-end query → answer pipeline.

- [ ] Full chain composed in LCEL (not `LLMChain`/`SequentialChain`)
- [ ] Metadata filters: source, page range, section, tags, chunk strategy
- [ ] Build the filtered retriever **fresh per request** from the shared
      vector store — never mutate a shared retriever's filter in place,
      two concurrent requests with different filters would clobber each
      other
- **Milestone:** full query → answer works end-to-end, filterable, with
  citations back to source chunks

## Stage 5 — Evaluation Pipeline (MVP slice #5)

**Goal:** quantitative proof one strategy beats another.

- [ ] Faithfulness + answer relevancy scoring (LLM-as-judge)
- [ ] Evaluation dataset: 15-30 Q&A pairs with reference answers, covering
      diverse question types:
      - Factual questions (direct fact lookup)
      - Conceptual questions (explain a concept)
      - Multi-hop questions (require combining info from multiple chunks)
      - Specific queries (IDs, numbers, error codes — tests BM25)
      - Comparative questions (tests decomposition)
- [ ] Batch evaluation endpoint across strategies
- [ ] Batch eval respects the same semaphore from Stage 0 — it must not
      starve live user queries if both run at the same time
- **Milestone:** comparison table, at least 2 strategies, real numbers

## Stage 6 — Frontend (can run in parallel with Stage 5)

**Goal:** the UI that makes the pipeline visible, not a black box.

- [ ] Query interface: input + strategy selector + filter panel
- [ ] Pipeline visualizer (step-by-step flow)
- [ ] Chunk inspector (click a chunk → see all scores)
- [ ] Streaming via SSE — keeps the connection open but backend work stays
      async, doesn't cost extra capacity beyond a normal request
- **Milestone:** a query typed in the browser returns a streamed answer
  with visible source chunks

## Stage 7 — A/B Comparison + Eval Dashboard

**Goal:** the demo centerpiece.

- [ ] A/B comparison page: same query, two strategies, side-by-side
- [ ] Eval dashboard with Recharts comparison across all metrics
- **Milestone:** this is the screen you record the demo video on

## Stage 8 — Concurrency Hardening Pass (do this before calling it done)

**Goal:** prove the concurrency design from Stage 0 actually holds.

- [ ] Run with `uvicorn --workers N` — confirms model loading is
      per-worker-at-startup, not per-request
- [ ] Load test: fire 5-10 concurrent queries (simple `asyncio.gather` or
      `httpx` concurrent script), confirm latency doesn't explode
- [ ] If latency does explode: it's almost always a model reloaded
      per-request, or a blocking call not wrapped in threadpool — check
      Stage 0 and Stage 2 first
- [ ] Basic per-IP/session rate limit so one user can't starve the demo
      for everyone else
- **Milestone:** concurrent load test passes, screenshot/log the result
  for the README

---

## Suggested Mapping to the 4-Day Timeline

| Day   | Stages                                |
| ----- | ------------------------------------- |
| Day 1 | Stage 0, Stage 1                      |
| Day 2 | Stage 2, Stage 3, Stage 4             |
| Day 3 | Stage 5, Stage 6                      |
| Day 4 | Stage 7, Stage 8, README + demo video |

## Build Order Rule

Do not start Stage 6 (frontend) content work until Stage 4 is proven
working end-to-end via API calls (curl/Postman) — evaluation numbers and
UI polish are meaningless if the retrieval pipeline underneath is still
shifting.
