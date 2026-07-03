# Frontend 03 — Design System & UX

**File identity:** Visual conventions, mainly around score/relevance
color-coding since transparency is the core UX goal of this project.

---

## Score Color Coding (used across chunk inspector, pipeline viz, eval dashboard)
- **Green:** high relevance/score (e.g. re-rank score, faithfulness > 0.7)
- **Yellow:** medium (0.4–0.7)
- **Red:** low relevance/score (< 0.4)
- Applied consistently to: similarity scores, BM25 scores, re-rank scores,
  and all 4 evaluation metrics.

## Layout
- **Query page:** left panel = answer, right panel = retrieved chunks list
  with rank badges (#1, #2, ...), bottom = collapsible pipeline
  visualization flowchart.
- **Pipeline visualizer:** each step is an expandable node, collapsed by
  default for a clean initial view.
- **Chunk inspector:** modal or side panel on chunk click — shows source
  doc, page, chunk text, embedding score, BM25 score, re-rank score,
  strategy used.

## Component Library
- shadcn/ui for primitives (buttons, dropdowns, modals, badges for rank
  numbers and score colors).

## Accessibility & Responsive
- Desktop-first — this is a technical/analytical tool used at a desk, not
  a mobile-first consumer app. Basic keyboard navigation on interactive
  elements (strategy selector, filters) is sufficient; full WCAG audit is
  out of scope for the assignment.

## Internationalization
- **N/A** — single-language (English) assignment, no i18n requirement.
