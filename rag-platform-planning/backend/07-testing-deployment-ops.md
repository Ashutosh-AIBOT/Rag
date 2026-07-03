# Backend 07 — Testing, Deployment & Ops

**File identity:** How you prove it works, how it ships, and how you find
out if it breaks — including the concurrency load test that proves
Stage 0/2's design decisions actually hold.

---

## Testing
- **Test runner:** pytest
- **Unit test scope:** each chunking strategy produces expected chunk
  count/metadata on a sample doc; RRF merge logic; re-rank ordering
  actually changes rank (catches the "misconfigured cross-encoder" mistake
  the assignment explicitly warns about).
- **Integration test scope:** full ingest → query → answer flow against
  a real sample document, for each of the 7 retrieval strategies.
- **Evaluation as test:** the eval dataset (15+ Q&A pairs) doubles as the
  most meaningful test — a strategy scoring far below others on
  faithfulness/relevancy signals a real bug, not just "this strategy is
  weaker."
- **Concurrency test:** a small script firing 5-10 concurrent queries via
  `asyncio.gather` or `httpx`, asserting latency stays within a sane bound
  and no request fails — this is Stage 8 of the staging plan, and it's the
  test most likely to catch the model-reloaded-per-request bug.

## Deployment
- **Containerization:** Dockerfile for backend, `docker-compose.yml`
  wiring backend + frontend (per assignment's expected project structure).
- **Environments:** local only for the assignment; docker-compose doubles
  as the "demo-ready" environment for the submission video.
- **Run command:** `uvicorn app.main:app --workers N` — multiple workers
  only make sense once Stage 0's per-worker-startup model loading is
  confirmed correct; running `--workers N` before that just multiplies
  the reload-per-request bug.

## Monitoring
- `GET /api/health` — basic liveness.
- `GET /api/stats` — retrieval latency, token usage; this is also your
  informal concurrency dashboard during the load test.

## Documentation (submission requirements)
- **README.md** — architecture diagram of the full RAG pipeline,
  explanation of each of the 7 retrieval strategies, evaluation results
  table, screenshots of pipeline visualizer + A/B comparison, setup
  instructions.
- **`.env.example`** — all required API keys.
- **Demo video** — 5-7 min: ingest, query with different strategies,
  pipeline visualization, A/B comparison, evaluation dashboard.
