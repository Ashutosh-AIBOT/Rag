# Backend 06 — Background Jobs

**File identity:** Anything that runs outside the request/response cycle.
For this project's scale, `FastAPI BackgroundTasks` is sufficient — no
Redis/RabbitMQ queue needed.

---

## Async Operations
- **Document ingestion:** upload endpoint returns `202 Accepted`
  immediately with a doc ID and status `processing`; the actual
  load → chunk (all 3 strategies) → embed → store pipeline runs via
  `BackgroundTasks`. Frontend polls `GET /api/documents` or the doc's
  status field to know when it's ready.
- **Batch evaluation:** `/api/evaluate/batch` can take a while (15-30
  Q&A pairs × multiple strategies × LLM-as-judge calls) — also runs via
  `BackgroundTasks`, with results written to `eval_results` as they
  complete rather than the endpoint blocking until everything finishes.

## Reliability
- **Retry policy:** one retry on transient LLM API failures (timeout/5xx),
  no retry on validation errors. No dead-letter queue needed at this
  scale — a failed ingestion just marks the document status `failed` with
  an error message visible in the UI.

## No Scheduled/Cron Jobs
- Not needed for this project — no recurring cleanup, digest emails, or
  scheduled scans apply to a single-user demo assignment.
