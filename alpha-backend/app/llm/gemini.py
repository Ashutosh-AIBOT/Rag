from typing import AsyncIterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from app.llm.base import BaseLLMProvider
from app.core.logging import get_logger

logger = get_logger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider."""

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        super().__init__(provider_name="gemini", model_name=model_name)

    def load_model(self) -> BaseChatModel:
        from langchain_google_genai import ChatGoogleGenerativeAI

        logger.info(f"Loading Gemini model: {self.model_name}")
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0.3,
            max_retries=1,
        )

    async def generate(self, messages: list[BaseMessage], **kwargs) -> str:
        model = self.get_model()
        response = await model.ainvoke(messages, **kwargs)
        logger.info(f"Gemini response generated, tokens: {response.usage_metadata}")
        return response.content

    async def stream(self, messages: list[BaseMessage], **kwargs) -> AsyncIterator[str]:
        model = self.get_model()
        async for chunk in model.astream(messages, **kwargs):
            if chunk.content:
                yield chunk.content
