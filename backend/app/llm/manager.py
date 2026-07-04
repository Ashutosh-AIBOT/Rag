import asyncio
import os
from typing import Any
from langchain_core.runnables import RunnableWithFallbacks
from langchain_core.language_models.llms import LLM
from app.config import settings
from app.core.logging import get_logger
from app.llm.base import BaseLLMProvider

logger = get_logger(__name__)
_semaphore = asyncio.Semaphore(5)


class LLMManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self):
        if self._initialized:
            return
        logger.info("Initializing LLMManager...")
        self._initialized = True
        self._loaded_llms = {}

    def load_all_models(self):
        self._load_llm("nvidia")
        self._load_llm("groq")
        self._load_llm("gemini")

    def _load_llm(self, provider_name: str):
        try:
            if provider_name == "nvidia":
                llm = self._load_nvidia_llm()
            elif provider_name == "groq":
                llm = self._load_groq_llm()
            elif provider_name == "gemini":
                llm = self._load_gemini_llm()
            else:
                logger.warning(f"Unknown provider: {provider_name}")
                return

            self._loaded_llms[provider_name] = llm
            logger.info(f"{provider_name} LLM loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load {provider_name}: {e}")

    def _load_nvidia_llm(self) -> LLM:
        provider = BaseLLMProvider(
            api_key=settings.NVIDIA_API_KEY,
            base_url="https://integrate.api.nvidia.com/v1",
            model="meta/llama-3.1-70b-instruct",
        )
        return provider.get_llm()

    def _load_groq_llm(self) -> LLM:
        provider = BaseLLMProvider(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.3-70b-versatile",
        )
        return provider.get_llm()

    def _load_gemini_llm(self) -> LLM:
        provider = BaseLLMProvider(
            api_key=settings.GOOGLE_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            model="gemini-2.0-flash",
        )
        return provider.get_llm()

    def get_llm_chain(self) -> RunnableWithFallbacks:
        if not self._initialized:
            self.initialize()
            self.load_all_models()
        primary = self._loaded_llms.get("nvidia")
        fallbacks = [
            llm for name, llm in self._loaded_llms.items() if name != "nvidia"
        ]
        if not primary:
            raise RuntimeError("No primary LLM loaded (nvidia)")
        return primary.with_fallbacks(fallbacks)

    def get_llm(self) -> Any:
        return self.get_llm_chain()

    def is_loaded(self) -> bool:
        return bool(self._loaded_llms)


llm_manager = LLMManager()


def get_llm():
    return llm_manager.get_llm()


def get_llm_chain():
    return llm_manager.get_llm_chain()


async def invoke_with_semaphore(chain, input_data: dict, config: dict = None) -> str:
    async with _semaphore:
        return await chain.ainvoke(input_data, config=config)
