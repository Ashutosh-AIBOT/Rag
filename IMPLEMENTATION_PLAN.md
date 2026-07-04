# RAG Platform - Complete Implementation Plan

## Project Overview
- **Assignment**: #7 Advanced RAG with LangChain
- **Stack**: FastAPI + LangChain + ChromaDB + Next.js
- **Target**: 5-10 concurrent users (Stage 0-7), scale to 1000+ post-assignment

---

## BRANCHING STRATEGY

```
main ──────────────────────────────────────────────────────────►
  │
  ├──► beta ──────────────────────────────────────────────────►
  │      │
  │      ├──► stage01 ──► fix ──► PR #6 ──► merge to beta
  │      │
  │      ├──► stage02 ──► fix ──► PR #7 ──► merge to beta
  │      │
  │      ├──► stage03 ──► implement ──► PR #8 ──► merge to beta
  │      │
  │      ├──► stage04 ──► implement ──► PR #9 ──► merge to beta
  │      │
  │      ├──► stage05 ──► implement ──► PR #10 ──► merge to beta
  │      │
  │      ├──► stage06 ──► implement ──► PR #11 ──► merge to beta
  │      │
  │      └──► stage07 ──► implement ──► PR #12 ──► merge to beta
  │
  └──► (sync from beta periodically)
```

## WORKFLOW RULES

1. **Before creating new branch**: `git checkout beta && git pull origin beta`
2. **Create branch**: `git checkout -b <stage-name>`
3. **Commit format**: `[RAG-XXX] type(api): description`
4. **PR format**: `gh pr create --base beta --title "Stage XX: Description"`
5. **Merge**: `gh pr merge --merge`

---

## STAGE 01: Document Ingestion & Basic Chunking (12 Items)

### Incomplete Items
| # | Item | Detail |
|---|------|--------|
| 1 | SQLite documents table missing status column | startup.py:16-25 CREATE TABLE has 8 columns, database.py:31 INSERT uses 9 values — crashes on first upload |
| 2 | SQLite timeout is 10s not 30s | startup.py:11,46 uses timeout=10, assignment requires 30.0 seconds |
| 3 | asyncio.Semaphore not created | llm/manager.py has no semaphore, query.py:6 tries to import _semaphore — ImportError |
| 4 | get_health_status() missing | health.py:3 imports it from lifespan.py — function doesn't exist — ImportError |
| 5 | core/exceptions.py missing | rag_chain.py:4 imports NoLLMProviderException — file doesn't exist — ImportError |
| 6 | core/dependencies.py missing | query.py imports from it — file doesn't exist — ImportError |
| 7 | get_rag_chain() missing | query.py:5 imports it from rag_chain.py — function doesn't exist (has query_rag) — ImportError |
| 8 | LLM manager never initialized | lifespan.py never calls llm_manager.initialize() — all LLM calls will crash |
| 9 | RAG chain not built with LCEL | rag_chain.py manually builds prompts + calls llm_manager.generate() instead of `prompt | llm | StrOutputParser` |
| 10 | LLM providers not wrapped as Runnables | llm/base.py uses custom generate()/stream() instead of LangChain's Runnable.invoke() |
| 11 | .env.example wrong DB format | Uses sqlite+aiosqlite:///./rag_database.db but code uses sqlite3.connect() which needs ./rag_database.db |
| 12 | Missing packages in requirements.txt | langchain-nvidia-ai-endpoints, langchain-groq, langchain-google-genai not listed |

### Files to Modify
```
backend/app/main.py                    - Fix syntax, add CORS, add /api prefix
backend/app/core/startup.py            - Add status column, timeout=30
backend/app/core/lifespan.py           - Add LLM init, embeddings naming
backend/app/database/database.py       - Add status column, CRUD functions
backend/app/embeddings/sentence_transformer.py - Add get_embedding_model singleton
backend/app/llm/models.py             - Fix protected_namespaces
backend/app/llm/manager.py            - Add get_llm, get_llm_chain, _semaphore
backend/app/llm/__init__.py           - Add exports
backend/app/llm/base.py               - Use LangChain Runnable patterns
backend/app/routers/query.py          - Fix imports, LCEL chain
backend/app/routers/health.py         - Fix health check
backend/app/services/rag_chain.py     - LCEL chain with pipe operator
backend/app/services/retrieval.py     - RunnableLambda wrapper
backend/app/models/schemas.py         - Add QueryRequest/QueryResponse
backend/.env.example                  - Fix DATABASE_URL format
backend/requirements.txt              - Add missing packages
```

### Implementation Steps
1. Fix main.py syntax error (remove cd command)
2. Add CORS middleware for localhost:3000
3. Add /api prefix to all routers
4. Fix database schema - add status column
5. Fix SQLite timeout to 30s
6. Add LLM manager initialization in lifespan
7. Create asyncio.Semaphore in llm/manager.py
8. Fix health check endpoint
9. Build LCEL RAG chain with pipe operator
10. Fix LLM providers to use LangChain Runnable patterns
11. Fix .env.example format
12. Add missing packages to requirements.txt

---

## STAGE 02: Advanced Chunking Strategies (6 Items)

### Incomplete Items
| # | Item | Detail |
|---|------|--------|
| 13 | 4 strategies run sequentially not parallel | chunking_strategies.py:29-41 runs each strategy one after another — should use RunnableParallel |
| 14 | Parent-child swapping not in LCEL chain | retrieval.py is a plain function, not a Runnable — can't compose into LCEL pipeline |
| 15 | Upload endpoint missing tags parameter | Assignment says users can assign tags during upload — endpoint has no tags field |
| 16 | Upload endpoint missing total_pages extraction | Assignment says extract total pages — code passes 0 always |
| 17 | database/schemas.py empty | Should have Pydantic models or be removed |
| 18 | services/chunking.py unused duplicate | Same logic exists in chunking_strategies.py:15-26 — dead code |

### Files to Modify
```
backend/app/services/chunking_strategies.py  - Use RunnableParallel
backend/app/services/retrieval.py            - Wrap as Runnable
backend/app/routers/documents.py             - Add tags, total_pages
backend/app/database/database.py             - Add parent_documents table
backend/app/services/semantic_chunker.py     - Fix embedding similarity logic
backend/app/services/parent_child_chunker.py - Fix parent mapping storage
backend/app/services/section_chunker.py      - Fix H1/H2 heading split
```

### Implementation Steps
1. Use RunnableParallel for concurrent chunking strategies
2. Wrap retrieval as RunnableLambda
3. Add tags parameter to upload endpoint
4. Extract total_pages from PDF
5. Create parent_documents table in SQLite
6. Remove duplicate services/chunking.py

---

## STAGE 03: BM25 Hybrid Search (6 Items)

### Incomplete Items
| # | Item | Detail |
|---|------|--------|
| 19 | BM25 retriever not implemented | No rank_bm25 usage anywhere in codebase |
| 20 | BM25 index persistence not implemented | No pickle/SQLite cache for BM25 index |
| 21 | BM25 not offloaded to thread pool | No asyncio.to_thread for BM25 operations |
| 22 | EnsembleRetriever not used | No LangChain EnsembleRetriever combining vector + BM25 |
| 23 | RRF (Reciprocal Rank Fusion) not implemented | No RRF merging logic |
| 24 | Configurable weights (50/50, 70/30) not implemented | No weight parameter for dense vs sparse |

### Files to Create
```
backend/app/services/bm25_retriever.py      - BM25 with rank_bm25
backend/app/services/hybrid_retriever.py    - EnsembleRetriever
backend/app/services/rrf.py                 - Reciprocal Rank Fusion
backend/app/routers/hybrid.py               - GET /api/strategies
```

### Files to Modify
```
backend/app/config.py                       - Add BM25 weights config
backend/app/services/retrieval.py           - Add hybrid search support
```

### Implementation Steps
1. Create BM25 retriever with rank_bm25 library
2. Implement BM25 index persistence (pickle/SQLite cache)
3. Wrap BM25 in asyncio.to_thread for async safety
4. Use LangChain EnsembleRetriever for vector + BM25
5. Implement RRF with k=60
6. Add configurable weights (50/50, 70/30)

---

## STAGE 04: Query Transforms & Re-ranking (9 Items)

### Incomplete Items
| # | Item | Detail |
|---|------|--------|
| 25 | Cross-encoder re-ranker not implemented | No sentence-transformers CrossEncoder usage |
| 26 | Cohere Rerank API not implemented | No Cohere integration |
| 27 | Re-rank score metadata not injected | No original_rank, reranked_position, relevance_score in chunk metadata |
| 28 | Multi-Query expansion not implemented | No MultiQueryRetriever or LLM query variation generation |
| 29 | HyDE not implemented | No hypothetical document embedding chain |
| 30 | Query decomposition not implemented | No sub-query splitting logic |
| 31 | Step-back prompting not implemented | No broader query generation |
| 32 | LangSmith tracing not configured | No LANGCHAIN_TRACING_V2 or LangSmith setup |
| 33 | No prompt templates for transforms | No hyde.py, decomposition.py, step_back.py files |

### Files to Create
```
backend/app/services/reranker.py            - Cross-encoder reranking
backend/app/services/scoring.py             - Score transparency
backend/app/services/hyde.py                - Hypothetical document embedding
backend/app/services/decomposition.py       - Query decomposition
backend/app/services/step_back.py           - Step-back prompting
backend/app/routers/rerank.py               - POST /api/rerank
```

### Files to Modify
```
backend/app/config.py                       - Add reranker model config
backend/app/services/rag_chain.py           - Add query transforms
```

### Implementation Steps
1. Load cross-encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2)
2. Retrieve top-20 → re-rank → return top-5
3. Inject original_rank, reranked_position, relevance_score metadata
4. Implement Multi-Query expansion (LLM generates 3-5 variants)
5. Implement HyDE chain (generate hypothetical answer → embed → search)
6. Implement Query decomposition (break complex questions into sub-queries)
7. Implement Step-back prompting (generate broader question first)
8. Configure LangSmith tracing
9. Create prompt templates for all transforms

---

## STAGE 05: RAGAS Evaluation (14 Items)

### Incomplete Items
| # | Item | Detail |
|---|------|--------|
| 34 | No evaluation endpoints | No POST /api/evaluate, POST /api/evaluate/batch, GET /api/evaluate/results |
| 35 | No query_history table | No SQLite table for storing queries and answers |
| 36 | No pipeline_traces table | No SQLite table for execution metadata |
| 37 | No eval_results table | No SQLite table for evaluation scores |
| 38 | No faithfulness scoring | No LLM-as-judge for faithfulness |
| 39 | No answer relevancy scoring | No LLM-as-judge for relevancy |
| 40 | No context precision scoring | No rank-aware precision metric |
| 41 | No context recall scoring | No coverage metric against reference |
| 42 | No RAGAS library integration | Not in requirements.txt, not used |
| 43 | No evaluation dataset | No eval_dataset.json with 15+ Q&A pairs |
| 44 | No token/cost accounting | No callback handlers for token counting |
| 45 | No /api/query/{id}/pipeline endpoint | No pipeline trace retrieval |
| 46 | No /api/strategies endpoint | No strategy listing |
| 47 | No /api/stats endpoint | No aggregate stats |

### Files to Create
```
backend/app/services/evaluator.py           - RAGAS evaluation logic
backend/app/services/scoring.py             - LLM-as-judge scoring
backend/app/routers/evaluation.py           - POST /api/evaluate
backend/app/data/eval_dataset.json          - 15+ Q&A pairs
```

### Files to Modify
```
backend/app/database/database.py            - Add 3 new tables
backend/app/routers/query.py               - Add pipeline trace
backend/requirements.txt                    - Add ragas library
```

### Implementation Steps
1. Create query_history, pipeline_traces, eval_results tables
2. Implement faithfulness scoring (LLM-as-judge)
3. Implement answer relevancy scoring
4. Implement context precision scoring
5. Implement context recall scoring
6. Integrate RAGAS library
7. Create evaluation dataset with 15+ Q&A pairs
8. Add token/cost accounting with callback handlers
9. Create evaluation endpoints
10. Add pipeline trace retrieval endpoint
11. Add strategy listing endpoint
12. Add aggregate stats endpoint

---

## STAGE 06: Next.js Frontend (7 Items)

### Incomplete Items
| # | Item | Detail |
|---|------|--------|
| 48 | No Next.js app initialized | frontend/ directory exists but no package.json, no components |
| 49 | No SSE streaming parser | No event: trace / event: delta handling |
| 50 | No Pipeline Visualizer component | No step-by-step flow display |
| 51 | No Chunk Inspector component | No chunk detail sidebar |
| 52 | No Document Management page | No /documents page with upload + preview |
| 53 | No strategy selector dropdown | No UI to pick retrieval strategy |
| 54 | No metadata filter panel | No source/page/section/tag filters |

### Files to Create
```
frontend/package.json                       - Next.js dependencies
frontend/src/app/page.tsx                   - Main page
frontend/src/components/ChatInterface.tsx   - Chat UI
frontend/src/components/PipelineVisualizer.tsx - Step-by-step flow
frontend/src/components/ChunkInspector.tsx  - Chunk details
frontend/src/components/DocumentManager.tsx - Upload + preview
frontend/src/components/StrategySelector.tsx - Strategy dropdown
```

### Implementation Steps
1. Initialize Next.js project with Tailwind + shadcn/ui
2. Create ChatGPT-like interface with SSE streaming
3. Build Pipeline Visualizer component
4. Build Chunk Inspector sidebar
5. Build Document Management page
6. Add strategy selector dropdown
7. Add metadata filter panel

---

## STAGE 07: A/B Testing & Docker (6 Items)

### Incomplete Items
| # | Item | Detail |
|---|------|--------|
| 55 | No A/B comparison page | No /compare page |
| 56 | No evaluation dashboard | No /evaluate page with Recharts |
| 57 | No leaderboard | No strategy ranking by average score |
| 58 | No Dockerfile | No backend/Dockerfile or frontend/Dockerfile |
| 59 | No docker-compose.yml | No multi-container setup |
| 60 | No multi-worker safety | No SQLite write lock retry for --workers 4 |

### Files to Create
```
frontend/src/app/compare/page.tsx          - A/B comparison
frontend/src/app/evaluate/page.tsx         - Evaluation dashboard
frontend/src/components/Leaderboard.tsx     - Strategy ranking
backend/Dockerfile                          - Backend container
frontend/Dockerfile                         - Frontend container
docker-compose.yml                          - Multi-container setup
```

### Implementation Steps
1. Create A/B comparison page
2. Create evaluation dashboard with Recharts
3. Build leaderboard component
4. Create Dockerfiles for backend and frontend
5. Create docker-compose.yml
6. Add SQLite write lock retry for multi-worker safety

---

## DO's (Must Follow)

| # | Rule | Why |
|---|------|-----|
| 1 | Use LCEL pipe `\|` operator for all chains | Assignment says "LangChain LCEL with composable chains" — 20% weight |
| 2 | Use RunnableLambda, RunnableParallel, RunnablePassthrough | These are the LCEL primitives expected |
| 3 | Use ChatPromptTemplate with `\|` pipe to LLM | Standard LCEL pattern |
| 4 | Use StrOutputParser() after LLM | Extracts string from AIMessage cleanly |
| 5 | Use primary.with_fallbacks([fallback1, fallback2]) for LLM fallback | LangChain native fallback, not custom class |
| 6 | Use EnsembleRetriever for hybrid search | LangChain native, handles RRF internally |
| 7 | Use RecursiveCharacterTextSplitter for chunking | Assignment requires it |
| 8 | Use PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader | Assignment requires these exact loaders |
| 9 | Use sentence-transformers/all-MiniLM-L6-v2 for embeddings | Assignment specifies this model |
| 10 | Use cross-encoder/ms-marco-MiniLM-L-6-v2 for re-ranking | Assignment specifies this model |
| 11 | Store strategy metadata on every chunk | Required for A/B comparison |
| 12 | Store source, page, doc_id, section metadata on chunks | Required for metadata filtering |
| 13 | Wrap all services in RunnableLambda | Makes them composable in LCEL chains |
| 14 | Use BackgroundTasks for ingestion | Don't block the main event loop |
| 15 | Use async def for all FastAPI endpoints | Assignment says "FastAPI with async support" |
| 16 | Return status 202 for upload endpoint | Assignment spec says 202 |
| 17 | Use asyncio.Semaphore to throttle LLM calls | Assignment requires concurrency control |
| 18 | Use asyncio.to_thread for CPU-heavy operations | Prevents blocking FastAPI event loop |
| 19 | Create documents table with status column | Must match INSERT statement |
| 20 | Use timeout=30 for all sqlite3.connect() | Assignment requires 30 seconds |
| 21 | Use app.state.embeddings not app.state.embedding_model | Assignment spec naming |
| 22 | Use Pydantic BaseModel for all request/response schemas | Code quality requirement |
| 23 | Initialize LLM manager in lifespan.py startup | Must call llm_manager.initialize() |
| 24 | Use langchain >= 0.2 modular imports | langchain_core, langchain_community, etc. |
| 25 | Include langchain-nvidia-ai-endpoints, langchain-groq, langchain-google-genai in requirements.txt | These are imported but not declared |
| 26 | Use sentence-transformers library for CrossEncoder re-ranking | Assignment specifies it |
| 27 | Use rank_bm25 library for BM25 | Assignment specifies it |
| 28 | Wrap re-ranking in asyncio.to_thread | Prevents blocking event loop |
| 29 | Inject original_rank, reranked_position, relevance_score into chunk metadata | Assignment requires score transparency |
| 30 | Use RunnableParallel for concurrent chunking strategies | Assignment says "run simultaneously" |
| 31 | Create evaluation dataset with 15+ Q&A pairs | Minimum requirement |
| 32 | Store parent documents in SQLite parent_documents table | Keeps ChromaDB lightweight |
| 33 | Swap child→parent content during retrieval | Assignment requires parent-child swapping flow |
| 34 | Use SSE streaming for LLM responses | Assignment requires streaming |
| 35 | Use StreamingResponse with media_type="text/event-stream" | FastAPI SSE pattern |

---

## DON'Ts (Must Avoid)

| # | Rule | Why |
|---|------|-----|
| 1 | Don't use LLMChain, SequentialChain, ConversationalChain | Deprecated — assignment explicitly says "do NOT use deprecated APIs" |
| 2 | Don't use RetrievalQA chain | Deprecated — use LCEL instead |
| 3 | Don't use ConversationChain | Deprecated — use LCEL instead |
| 4 | Don't use legacy initialize_agent | Deprecated — use LCEL agent patterns |
| 5 | Don't use from langchain.llms import ... | Old import path — use langchain_community or langchain_core |
| 6 | Don't use from langchain.chat_models import ... | Old import path — use langchain_community.chat_models |
| 7 | Don't use from langchain.embeddings import ... | Old import path — use langchain_huggingface |
| 8 | Don't use from langchain.vectorstores import ... | Old import path — use langchain_chroma |
| 9 | Don't hardcode API keys in code | Always use .env + pydantic-settings |
| 10 | Don't use print() for logging | Use logging module via get_logger() |
| 11 | Don't block the event loop with sync operations | Use asyncio.to_thread for sync code |
| 12 | Don't use os.environ directly | Use pydantic-settings Settings class |
| 13 | Don't create new embedding model instances per request | Use singleton pattern — load once on startup |
| 14 | Don't create new ChromaDB client per request | Use app.state.vectorstore from lifespan |
| 15 | Don't use @app.on_event("startup") | Deprecated — use lifespan context manager |
| 16 | Don't return raw SQLite rows from endpoints | Use Pydantic models |
| 17 | Don't use from langchain.text_splitter import ... | Old path — use from langchain_text_splitters import ... |
| 18 | Don't use from langchain.document_loaders import ... | Old path — use from langchain_community.document_loaders import ... |
| 19 | Don't skip metadata on chunks | Every chunk needs source, page, strategy, doc_id |
| 20 | Don't use temperature=0 for evaluation LLM | Use small value like 0.1 for consistency |
| 21 | Don't forget status column in documents table | INSERT will crash without it |
| 22 | Don't use sqlite+aiosqlite:/// format with sqlite3.connect() | Code uses sync sqlite3, needs plain file path |
| 23 | Don't create duplicate files | services/chunking.py duplicates chunking_strategies.py — remove one |
| 24 | Don't use custom fallback handler when LangChain has with_fallbacks() | Assignment expects LangChain native patterns |
| 25 | Don't skip the evaluation dataset | Assignment requires 15+ Q&A pairs for benchmarking |
| 26 | Don't use synchronous requests library | Use httpx or aiohttp for async |
| 27 | Don't forget CORS middleware | Frontend at localhost:3000 needs access |
| 28 | Don't use cd in Python files | main.py:1 had shell command — must be removed |
| 29 | Don't skip error handling in endpoints | Every endpoint needs try/except with proper HTTP status codes |
| 30 | Don't use from pydantic import BaseModel with class Config | Use model_config = ConfigDict(...) in Pydantic v2 |

---

## CRITICAL PATTERNS

### LCEL RAG Chain Pattern
```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### LCEL Ingestion Pattern
```python
chain = (
    RunnableLambda(load)
    | RunnableLambda(split)
    | RunnableLambda(store)
)
```

### LLM Fallback Pattern
```python
primary = ChatNVIDIA(...)
fallback1 = ChatGroq(...)
fallback2 = ChatGoogleGenerativeAI(...)
chain = primary.with_fallbacks([fallback1, fallback2])
```

### Hybrid Search Pattern
```python
ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)
```

---

## EXECUTION ORDER

1. **Stage 01** → Fix 12 items → PR to beta → Merge
2. **Stage 02** → Fix 6 items → PR to beta → Merge
3. **Stage 03** → Implement 6 items → PR to beta → Merge
4. **Stage 04** → Implement 9 items → PR to beta → Merge
5. **Stage 05** → Implement 14 items → PR to beta → Merge
6. **Stage 06** → Implement 7 items → PR to beta → Merge
7. **Stage 07** → Implement 6 items → PR to beta → Merge

**Total**: 60 items across 7 stages
