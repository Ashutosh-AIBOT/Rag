from abc import ABC, abstractmethod
from typing import AsyncIterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from app.core.logging import get_logger

logger = get_logger(__name__)


class BaseLLMProvider(ABC):
    def __init__(self, provider_name: str, model_name: str):
        self.provider_name = provider_name
        self.model_name = model_name
        self._model: BaseChatModel | None = None
        logger.info(f"Initializing {provider_name} provider with model: {model_name}")

    @abstractmethod
    def load_model(self) -> BaseChatModel:
        ...

    @abstractmethod
    async def generate(self, messages: list[BaseMessage], **kwargs) -> str:
        ...

    @abstractmethod
    async def stream(self, messages: list[BaseMessage], **kwargs) -> AsyncIterator[str]:
        ...

    def get_model(self) -> BaseChatModel:
        if self._model is None:
            self._model = self.load_model()
            logger.info(f"Model loaded: {self.provider_name}/{self.model_name}")
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
