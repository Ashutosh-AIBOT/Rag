# Pending Tasks — Re-Ranker

## Re-Ranking Strategy

### Current: Local Cross-Encoder (Active)
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Runs locally via `sentence-transformers`
- No API key required
- Implemented in: `services/reranker.py`

### Future: Cohere Rerank API (Later Stages)
- Add `COHERE_API_KEY` support in `.env`
- Use Cohere Rerank API as fallback when local model is slow or unavailable
- Free tier available
- Will be integrated after core pipeline is complete

## Notes
- For now, ignore `COHERE_API_KEY` in `.env`
- Local cross-encoder is the primary choice
- Cohere integration deferred to later stages
