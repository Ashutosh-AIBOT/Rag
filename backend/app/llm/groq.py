from typing import AsyncIterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from app.llm.base import BaseLLMProvider
from app.core.logging import get_logger

logger = get_logger(__name__)


class GroqProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        super().__init__(provider_name="groq", model_name=model_name)

    def load_model(self) -> BaseChatModel:
        from langchain_groq import ChatGroq

        logger.info(f"Loading Groq model: {self.model_name}")
        return ChatGroq(model=self.model_name, temperature=0.3)

    async def generate(self, messages: list[BaseMessage], **kwargs) -> str:
        model = self.get_model()
        response = await model.ainvoke(messages, **kwargs)
        logger.info(f"Groq response generated")
        return response.content

    async def stream(self, messages: list[BaseMessage], **kwargs) -> AsyncIterator[str]:
        model = self.get_model()
        async for chunk in model.astream(messages, **kwargs):
            if chunk.content:
                yield chunk.content
