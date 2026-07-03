# Frontend 01 — Tech Stack

**File identity:** Client-side dependency choices, per assignment spec.

---

## Core
- **Framework:** Next.js 14+ with TypeScript
- **Styling:** Tailwind CSS + shadcn/ui components
- **Charts:** Recharts (evaluation metric visualization)
- **Diff view:** react-diff-viewer (A/B comparison of answers)
- **Code display:** react-syntax-highlighter (LLM prompts, pipeline
  details)
- **State management:** Zustand (or React Context — Zustand preferred for
  the query/pipeline state that multiple components need to read)

## Rendering Strategy
- All pages client-rendered (CSR) — this is an interactive tool
  (query → live pipeline visualization → streamed answer), not
  content that benefits from SSR/SSG.
- No multi-app routing boundary needed — single application, routes are
  just the 5 screens (query, documents, compare, evaluate).

## Streaming
- SSE connection from `/api/query` — answer tokens rendered incrementally
  as they arrive, pipeline-step events (retrieval done, re-ranking done)
  update the pipeline visualizer in real time before the answer starts
  streaming.

## Auth on Client
- N/A for MVP (no auth). If the shared API key is added later, store it
  in an environment variable injected at build time, sent as a header on
  every request — not stored in localStorage.
