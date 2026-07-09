"""
A tiny tracer used to record every step of a RAG query
(query transform -> initial retrieval -> rerank -> context assembly -> generation)
with timings and scores, so the frontend PipelineVisualizer can render it
step by step, and the ChunkInspector / Retrieval Debugger can show why a
particular chunk was (or wasn't) used.
"""
import time
from contextlib import contextmanager
from typing import List, Dict, Any


class PipelineTracer:
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []

    @contextmanager
    def step(self, name: str, **detail):
        start = time.perf_counter()
        record = {"name": name, "detail": detail}
        try:
            yield record
        finally:
            record["duration_ms"] = round((time.perf_counter() - start) * 1000, 2)
            self.steps.append(record)

    def as_list(self) -> List[Dict[str, Any]]:
        return self.steps
