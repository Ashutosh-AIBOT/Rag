from typing import AsyncIterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from app.llm.base import BaseLLMProvider
from app.core.logging import get_logger

logger = get_logger(__name__)


class NvidiaProvider(BaseLLMProvider):
    """NVIDIA NIM LLM provider."""

    def __init__(self, model_name: str = "meta/llama-3.3-70b-instruct"):
        super().__init__(provider_name="nvidia", model_name=model_name)

    def load_model(self) -> BaseChatModel:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        logger.info(f"Loading NVIDIA model: {self.model_name}")
        return ChatNVIDIA(
            model=self.model_name,
            temperature=0.3,
            max_retries=1,
        )

    async def generate(self, messages: list[BaseMessage], **kwargs) -> str:
        model = self.get_model()
        response = await model.ainvoke(messages, **kwargs)
        logger.info(f"NVIDIA response generated")
        return response.content

    async def stream(self, messages: list[BaseMessage], **kwargs) -> AsyncIterator[str]:
        model = self.get_model()
        async for chunk in model.astream(messages, **kwargs):
            if chunk.content:
                yield chunk.content
