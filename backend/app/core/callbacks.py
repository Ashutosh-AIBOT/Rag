import time
from typing import Dict, Any, List, Optional
from uuid import UUID
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from app.core.logging import get_logger

logger = get_logger(__name__)


class MetricsTrackingCallbackHandler(AsyncCallbackHandler):
    """
    Standard AsyncCallbackHandler for capturing token usage, LLM latency, 
    and prompt details in compliance with standard LCEL patterns.
    """
    def __init__(self):
        super().__init__()
        self.start_time: float = 0.0
        self.latency_ms: float = 0.0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0

    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.start_time = time.time()
        logger.info(f"LLM run started. Run ID: {run_id}. Prompts count: {len(prompts)}")

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self.latency_ms = (time.time() - self.start_time) * 1000
        
        # Parse standard token usage metadata if available
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            self.prompt_tokens = usage.get("prompt_tokens", 0)
            self.completion_tokens = usage.get("completion_tokens", 0)
            self.total_tokens = usage.get("total_tokens", 0)
            
        logger.info(
            f"LLM run ended. Run ID: {run_id}. Latency: {self.latency_ms:.2f}ms. "
            f"Tokens -> Prompt: {self.prompt_tokens}, Completion: {self.completion_tokens}, Total: {self.total_tokens}"
        )
