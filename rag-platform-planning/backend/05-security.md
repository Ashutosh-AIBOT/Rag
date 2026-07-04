# Backend 05 — Security

**File identity:** Access control and abuse prevention, scoped to what
actually matters for a 4-day technical assignment (not a full production
security posture — see Project Overview for the reasoning).

---

## Auth
- **Method:** none required for local demo/grading. If deployed publicly
  for the demo video, add one shared API key checked via a FastAPI
  dependency (`X-API-Key` header) — not full user accounts, which would be
  scope creep against the assignment's actual grading criteria.
- **Secrets management:** all API keys (`GOOGLE_API_KEY`, `GROQ_API_KEY`,
  `COHERE_API_KEY`) in `.env`, never committed — `.env.example` lists the
  required keys with placeholder values, per the submission requirements.

## Input Handling
- **File upload validation:** enforce file type allowlist (PDF, TXT, DOCX,
  MD only) and a max file size, before the file reaches any LangChain
  loader — prevents oversized or malicious uploads from consuming worker
  capacity.
- **Input sanitization:** query text passed to LLM prompts is not
  HTML-rendered anywhere, so XSS isn't a concern server-side; frontend
  should still escape any user text rendered in the UI (chunk previews,
  query history).

## Rate Limiting (abuse prevention, not tiers)
- Soft per-IP limit on `/api/query` and `/api/evaluate/batch` (see API
  Design file) — this is the actual concurrency-abuse concern for this
  project: one user firing rapid queries during a demo shouldn't exhaust
  the LLM free-tier quota for everyone else.

## AI-Specific Safety
- **Prompt-injection surface:** uploaded document content becomes part of
  the LLM context — a malicious document could contain text trying to
  override the system prompt. Mitigation: keep the system prompt
  instruction to "answer only from the provided context" explicit and
  reinforced in the prompt template, and don't let retrieved-chunk text
  be interpreted as instructions in the chain construction.
- **No user-generated content moderation needed** — this system answers
  from uploaded documents, it doesn't generate free-form public-facing
  content, so the fuller AI-safety checklist (abuse escalation, content
  filtering) from the master plan doesn't apply at this scope.

## What's Explicitly Out of Scope Here
- MFA, role hierarchy, IDOR/ownership checks, CORS whitelist beyond
  allowing the local frontend origin, log redaction — all N/A for a
  single-user local/demo assignment. Revisit if this becomes a real
  multi-user deployment.
