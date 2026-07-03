# Backend 02 — Database

**File identity:** Data storage — both the vector store and the
relational metadata store.

---

## Engine & Topology
- **Vector store:** ChromaDB (local, persistent client) — stores chunk
  embeddings + metadata (`source`, `page_number`, `strategy`, `section`,
  `tags`, `upload_date`).
- **Relational store:** SQLite — query history, evaluation results,
  document metadata table.
- **Tenant isolation:** N/A — single-tenant project (see
  `00-project-overview.md`).
- **Connection handling:** SQLite opened in **WAL mode**
  (`PRAGMA journal_mode=WAL`) so concurrent reads aren't blocked by writes
  during query logging — this is the one concurrency-relevant DB decision
  for this project's scale.
- **Query timeout:** not critical at this scale; default driver timeout is
  fine.

## Indexing & Performance
- ChromaDB metadata fields used in filters (`source`, `strategy`, `section`,
  `tags`) should be set up for metadata filtering per Chroma's filter API —
  confirm the fields are actually queryable, not just stored.
- SQLite: index on `document_id` in the chunks-metadata table and on
  `query_id` in the eval-results table, since both get looked up by ID from
  the frontend (chunk inspector, pipeline trace).

## Migrations & Seed Data
- **Migration tool:** none needed for SQLite at this scope — a single
  `init_db.py` that creates tables if they don't exist is sufficient.
- **Seed data:** `sample_documents/` (3-5 test docs) + `eval_dataset/`
  (15+ Q&A pairs) ingested via a one-time setup script, not manual UI
  upload, so grading/demo setup is repeatable.

## Backup & Disaster Recovery
- **N/A** — local demo project, no production data to protect. If this
  becomes a real deployed service later, revisit with real backup
  frequency and RTO/RPO targets.

## Tables

### SQLite tables
- **documents** — id, filename, file_type, total_pages, upload_date,
  file_size, tags
- **query_history** — id, query_text, strategy, filters_applied, answer,
  latency_ms, token_count, created_at
- **pipeline_traces** — id, query_id, step_name, step_data (JSON), timing_ms
- **eval_results** — id, query_id, strategy, faithfulness_score,
  relevancy_score, precision_score, recall_score

### ChromaDB collection
- **chunks** — one collection, metadata field `strategy` distinguishes
  recursive/parent-child/section-based chunks stored in the same
  collection, per the assignment's "run all strategies simultaneously"
  requirement.
