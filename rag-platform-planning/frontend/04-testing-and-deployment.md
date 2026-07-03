# Frontend 04 — Testing & Deployment

**File identity:** How the frontend is verified and shipped for the
assignment submission.

---

## Testing
- Manual flow testing over automated test suites, given the 4-day scope:
  - Upload a document, confirm it appears with correct chunk counts
  - Run each of the 7 strategies on the same query, confirm visibly
    different results
  - Run A/B comparison, confirm side-by-side rendering and chunk overlap
    highlighting work
  - Run batch evaluation, confirm the dashboard renders real scores
- If time allows: a handful of component tests (React Testing Library) on
  `StrategySelector` and `MetadataFilters` since those drive query
  correctness.

## Build & Deployment
- **Build:** standard `next build`
- **Hosting:** local via `docker-compose` for the demo/submission —
  matches the assignment's expected `docker-compose.yml` at repo root.
  Vercel deployment optional/bonus, not required.
- **Env vars:** `NEXT_PUBLIC_API_URL` pointing at the FastAPI backend,
  injected per environment (local vs. docker-compose network).

## Performance
- Not a primary grading criterion for this assignment — reasonable bundle
  size via Next.js defaults is sufficient, no dedicated performance budget
  needed.
