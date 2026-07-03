from typing import AsyncIterator

from langchain_core.messages import BaseMessage

from app.llm.base import BaseLLMProvider
from app.llm.gemini import GeminiProvider
from app.llm.groq import GroqProvider
from app.llm.nvidia import NvidiaProvider
from app.llm.fallback import FallbackHandler
from app.llm.models import ProviderName
from app.core.logging import get_logger

logger = get_logger(__name__)


class LLMManager:
    def __init__(self):
        self._providers: dict[ProviderName, BaseLLMProvider] = {}
        self._fallback: FallbackHandler | None = None
        self._initialized = False

    def initialize(self, primary: ProviderName = ProviderName.NVIDIA) -> None:
        if self._initialized:
            return

        self._providers[ProviderName.GEMINI] = GeminiProvider()
        self._providers[ProviderName.GROQ] = GroqProvider()
        self._providers[ProviderName.NVIDIA] = NvidiaProvider()

        self._fallback = FallbackHandler(self._providers)
        self._fallback.set_active(primary)
        self._initialized = True
        logger.info(f"LLMManager initialized, primary: {primary.value}")

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("LLMManager not initialized")

    def get_active_provider(self) -> BaseLLMProvider:
        self._ensure_initialized()
        provider = self._fallback.get_provider(self._fallback.active_provider)
        if provider is None:
            raise RuntimeError("No active LLM provider available")
        return provider

    async def generate(self, messages: list[BaseMessage], **kwargs) -> str:
        self._ensure_initialized()

        async def _generate(provider: BaseLLMProvider, msgs, opts):
            return await provider.generate(msgs, **opts)

        return await self._fallback.execute_with_fallback(_generate, messages, kwargs)

    async def stream(self, messages: list[BaseMessage], **kwargs) -> AsyncIterator[str]:
        self._ensure_initialized()
        provider = self.get_active_provider()
        async for chunk in provider.stream(messages, **kwargs):
            yield chunk

    def load_all_models(self) -> None:
        self._ensure_initialized()
        for name, provider in self._providers.items():
            try:
                provider.get_model()
                logger.info(f"Model loaded: {name.value}")
            except Exception as e:
                logger.error(f"Failed to load model {name.value}: {e}")

    @property
    def status(self) -> dict:
        self._ensure_initialized()
        return {
            "active_provider": self._fallback.active_provider.value,
            "loaded_providers": [name.value for name, p in self._providers.items() if p.is_loaded],
        }


llm_manager = LLMManager()
