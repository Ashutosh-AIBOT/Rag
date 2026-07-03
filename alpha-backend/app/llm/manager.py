from typing import AsyncIterator
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel

from app.config import settings
from app.llm.gemini import GeminiProvider
from app.llm.groq import GroqProvider
from app.llm.nvidia import NvidiaProvider
from app.llm.models import ProviderName
from app.core.logging import get_logger

logger = get_logger(__name__)


class LLMManager:
    """Manages LLM providers and chains them natively via LangChain's with_fallbacks."""

    def __init__(self):
        self._providers = {}
        self._runnable = None
        self._initialized = False

    def initialize(self, primary: ProviderName = ProviderName.GEMINI) -> None:
        """Initializes providers and chains them natively in LangChain."""
        if self._initialized:
            logger.warning("LLMManager is already initialized.")
            return

        # 1. Initialize the providers with configured model names
        self._providers[ProviderName.GEMINI] = GeminiProvider(model_name=settings.GEMINI_MODEL)
        self._providers[ProviderName.GROQ] = GroqProvider(model_name=settings.GROQ_MODEL)
        self._providers[ProviderName.NVIDIA] = NvidiaProvider(model_name=settings.NVIDIA_MODEL)

        # 2. Build fallback order sequence: Nvidia -> Groq -> Gemini
        fallback_sequence = [ProviderName.NVIDIA, ProviderName.GROQ, ProviderName.GEMINI]
        ordered = [primary] + [p for p in fallback_sequence if p != primary]

        logger.info(f"Setting up LLM fallback chain. Primary: {primary.value}. Order: {[p.value for p in ordered]}")

        # 3. Load LangChain BaseChatModel instances
        models = []
        for name in ordered:
            try:
                provider = self._providers[name]
                models.append(provider.get_model())
                logger.info(f"Successfully preloaded provider: {name.value}")
            except Exception as e:
                logger.error(f"Failed to preload provider {name.value}: {e}", exc_info=True)

        if not models:
            logger.critical("No LLM models could be successfully preloaded!")
            raise RuntimeError("No LLM models could be loaded.")

        # 4. Chain natively using LangChain's .with_fallbacks()
        primary_model = models[0]
        if len(models) > 1:
            self._runnable = primary_model.with_fallbacks(models[1:])
            logger.info("LangChain native fallback chain constructed successfully.")
        else:
            self._runnable = primary_model

        self._initialized = True

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("LLMManager not initialized. Call initialize() first.")

    def get_llm(self) -> BaseChatModel:
        self._ensure_initialized()
        return self._runnable

    async def generate(self, messages: list[BaseMessage], **kwargs) -> str:
        self._ensure_initialized()
        try:
            logger.info("Sending prompt to LangChain native fallback runnable...")
            response = await self._runnable.ainvoke(messages, **kwargs)
            logger.info("Generation succeeded.")
            return response.content
        except Exception as e:
            logger.error(f"All LLMs in the fallback chain failed: {e}", exc_info=True)
            raise e

    async def stream(self, messages: list[BaseMessage], **kwargs) -> AsyncIterator[str]:
        self._ensure_initialized()
        try:
            logger.info("Streaming response from LangChain native fallback runnable...")
            async for chunk in self._runnable.astream(messages, **kwargs):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Streaming failed: all fallbacks exhausted. Error: {e}", exc_info=True)
            raise e


llm_manager = LLMManager()
