# Frontend 02 — Features & Pages

**File identity:** The 5 screens and their components, directly from the
assignment's UI spec.

---

## Pages

| Page | Route | Purpose |
|---|---|---|
| Query Interface (main) | `/` | Query input, strategy selector, filters, answer + chunks, pipeline viz |
| Document Management | `/documents` | Upload, list, delete, inspect chunks per document |
| A/B Comparison | `/compare` | Same query, two strategies, side-by-side |
| Evaluation Dashboard | `/evaluate` | Run eval, charts, leaderboard, failure analysis |

## Components

| Component | Used on | Responsibility |
|---|---|---|
| `QueryPanel.tsx` | Query page | Query input + strategy selector |
| `AnswerDisplay.tsx` | Query page | Streamed answer + source citations |
| `PipelineVisualizer.tsx` | Query page | Full pipeline steps, expandable, color-coded by score |
| `ChunkInspector.tsx` | Query page | Click a chunk → source, page, scores, strategy |
| `ComparisonView.tsx` | Compare page | Side-by-side answers + chunk overlap highlighting |
| `EvalDashboard.tsx` | Evaluate page | Bar charts across all metrics + strategies |
| `MetadataFilters.tsx` | Query page | Source/page-range/section/tag/strategy filter controls |
| `DocumentList.tsx` | Documents page | Uploaded docs with chunk counts per strategy |
| `StrategySelector.tsx` | Query + Compare pages | Dropdown for the 7 retrieval strategies |

## Pipeline Visualizer Steps (rendered in order)
1. Original Query
2. Query Transformation (if applied — multi-query variants, HyDE
   hypothetical answer, decomposed sub-questions)
3. Initial Retrieval (top-20, with scores)
4. Re-Ranking (score changes, before/after)
5. Context Assembly (token count)
6. LLM Generation (full prompt shown via `react-syntax-highlighter`)

## New Feature Checklist (if a new retrieval strategy is added later)
- [ ] Backend: implement in `services/retrieval.py` or `query_transform.py`
- [ ] Backend: add to `GET /api/strategies` response
- [ ] Frontend: appears automatically in `StrategySelector.tsx` since it
      reads from `/api/strategies` — no hardcoded strategy list in the UI

---

## Missing Features Added (from assignment Section 3.7, Section 7, Section 8)

### Token Accounting (Good-to-Have #7)
- Show per-query: input tokens (context + query), output tokens (answer),
  total estimated cost
- Break down context tokens by source chunk (which chunk contributed how
  many tokens)
- Display in the pipeline visualizer's "Context Assembly" step and in a
  summary badge on the answer

### Context Window View (Must-Have #10 detail)
- Show exactly what context was sent to the LLM: system prompt + context
  chunks + user question
- Rendered via `react-syntax-highlighter` in the pipeline visualizer's
  "LLM Generation" step
- Users can copy the full prompt for debugging

### Chunk Overlap Visualization (Good-to-Have #9)
- In the document detail view, show how chunks from different strategies
  overlap in the source document
- Highlight overlapping text ranges with different colors per strategy
- Helps users understand how chunking strategies slice the same document
  differently

### Query History (Good-to-Have #10)
- Sidebar or dedicated section on the query page showing past queries
- Each entry: query text, strategy used, timestamp, latency
- Click to re-run with same or different strategy
- Compare a past query's results to a new run
- Data stored in SQLite `query_history` table (already planned in
  `backend/02-database.md`)

### Document Management — Chunk Preview Detail (Assignment Screen 5)
- Click any document → see all its chunks grouped by strategy
  (recursive, parent-child, section-based, semantic)
- Each group shows: chunk count, average chunk size
- Chunk text previews (truncated, expandable)
- Chunk metadata visible: page number, section header, strategy tag

