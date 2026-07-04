from app.llm.base import BaseLLMProvider
from app.llm.models import ProviderName
from app.core.logging import get_logger

logger = get_logger(__name__)

FALLBACK_ORDER = [
    ProviderName.NVIDIA,
    ProviderName.GROQ,
    ProviderName.GEMINI,
]


class FallbackHandler:
    def __init__(self, providers: dict[ProviderName, BaseLLMProvider]):
        self.providers = providers
        self._current_provider: ProviderName | None = None

    @property
    def active_provider(self) -> ProviderName | None:
        return self._current_provider

    def set_active(self, provider: ProviderName) -> None:
        self._current_provider = provider
        logger.info(f"Active LLM provider set to: {provider.value}")

    def get_provider(self, name: ProviderName) -> BaseLLMProvider | None:
        return self.providers.get(name)

    async def execute_with_fallback(self, func, *args, **kwargs):
        ordered_providers = [self._current_provider] if self._current_provider else []
        ordered_providers += [p for p in FALLBACK_ORDER if p not in ordered_providers]

        for provider_name in ordered_providers:
            provider = self.providers.get(provider_name)
            if provider is None:
                continue
            try:
                logger.info(f"Trying provider: {provider_name.value}")
                result = await func(provider, *args, **kwargs)
                self._current_provider = provider_name
                logger.info(f"Success with provider: {provider_name.value}")
                return result
            except Exception as e:
                logger.warning(f"Provider {provider_name.value} failed: {e}")
                continue

        logger.error("All LLM providers failed")
        raise RuntimeError("All LLM providers failed")
