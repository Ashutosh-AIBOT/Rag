# Shared 02 — Cost & Budget

**File identity:** Money for this project — target is $0.

---

## Cost Categories

| Category | Choice | Est. monthly cost |
|---|---|---|
| Hosting/compute | Local machine / docker-compose | $0 |
| Vector DB | ChromaDB, self-hosted local | $0 |
| Relational DB | SQLite, local file | $0 |
| Embeddings | sentence-transformers, local | $0 |
| Re-ranking | Local cross-encoder (Cohere free tier as backup) | $0 |
| LLM | Gemini or Groq free tier | $0 |
| CI/CD | None required for submission | $0 |
| **Total** | | **$0** |

## Optimization Rules
- Everything runs local/free-tier by design — no paid service is required
  to complete or demo this assignment.
- If a free-tier LLM rate limit is hit during heavy testing, switch
  provider via the abstraction in `rag_chain.py` (Gemini ↔ Groq) rather
  than paying for a higher tier.

## Budget Cap
- **Cap:** $0. If any service would require payment to function at
  assignment scale, that's a signal to swap it for the free/local
  alternative already listed in `backend/01-tech-stack-and-services.md`,
  not to start paying.

## Product-Facing Billing
- **N/A** — this is a technical assignment/demo, not a product with
  paying customers.
