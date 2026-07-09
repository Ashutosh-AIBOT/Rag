"""
Factories that build the LangChain LLM / Embeddings objects.

Rest of the codebase only ever imports `get_llm()` / `get_embeddings()` and
never hardcodes a provider. Two modes:

  - Explicit provider (LLM_PROVIDER=openai|google|groq|nvidia): always use
    that one provider, error if it fails (predictable, useful for local dev
    with a single key).

  - LLM_PROVIDER=auto (default): builds a single Runnable that tries
    providers in `LLM_FALLBACK_ORDER` (default: nvidia -> groq -> google)
    using LangChain's native `.with_fallbacks()`. If NVIDIA NIM is down,
    rate-limited, or its key is missing, the request transparently retries
    on Groq, then on Gemini, without the caller ever knowing a fallback
    happened (it's just a Runnable like any other). This is what makes the
    RAG pipeline resilient under concurrent load against free-tier limits.

LangSmith tracing (see config.py) instruments every one of these Runnables
automatically via env vars -- no extra code needed here.
"""
import logging
from functools import lru_cache

from app.config import get_settings

logger = logging.getLogger("llm_factory")
settings = get_settings()


def _build_provider_llm(provider: str, temperature: float):
    """Constructs a single ChatModel for one named provider. Raises if the
    provider's SDK isn't installed or its API key is missing, so callers
    (the fallback chain) can catch that and move to the next provider."""
    provider = provider.lower().strip()

    if provider == "nvidia":
        if not settings.nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY not configured")
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        logger.info("Initializing ChatNVIDIA model %s (temp=%.2f, timeout=5s)", settings.nvidia_model, temperature)
        return ChatNVIDIA(
            model=settings.nvidia_model,
            api_key=settings.nvidia_api_key,
            temperature=temperature,
            timeout=5,
            max_retries=1,
        ).with_config({"tags": ["provider:nvidia"], "run_name": "nvidia-llm"})

    if provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY not configured")
        from langchain_groq import ChatGroq
        logger.info("Initializing ChatGroq model %s (temp=%.2f, timeout=5s)", settings.groq_model, temperature)
        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=temperature,
            timeout=5,
            max_retries=1,
            streaming=True,
        ).with_config({"tags": ["provider:groq"], "run_name": "groq-llm"})

    if provider == "google":
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY not configured")
        from langchain_google_genai import ChatGoogleGenerativeAI
        logger.info("Initializing ChatGoogleGenerativeAI model %s (temp=%.2f, timeout=5s)", settings.google_model, temperature)
        return ChatGoogleGenerativeAI(
            model=settings.google_model,
            google_api_key=settings.google_api_key,
            temperature=temperature,
            timeout=5,
            max_retries=1,
        ).with_config({"tags": ["provider:google"], "run_name": "gemini-llm"})

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY not configured")
        from langchain_openai import ChatOpenAI
        logger.info("Initializing ChatOpenAI model %s (temp=%.2f, timeout=5s)", settings.openai_model, temperature)
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=temperature,
            timeout=5,
            max_retries=1,
        ).with_config({"tags": ["provider:openai"], "run_name": "openai-llm"})

    if provider == "mock":
        from unittest.mock import MagicMock
        from langchain_core.messages import AIMessage
        logger.info("Returning mock LLM for testing")
        mock = MagicMock()
        msg = AIMessage(content="This is a mock answer based on the provided context.")
        mock.invoke.return_value = msg
        mock.return_value = msg
        return mock

    raise ValueError(f"Unknown LLM provider: {provider}")


from langchain_core.runnables import Runnable

class RobustFallbackRunnable(Runnable):
    def __init__(self, models):
        self.models = models

    def invoke(self, input, config=None, **kwargs):
        last_err = None
        for model in self.models:
            try:
                logger.info("RobustFallbackRunnable: Attempting invoke with model %s", model)
                return model.invoke(input, config, **kwargs)
            except Exception as e:
                logger.warning("RobustFallbackRunnable: Model %s failed in invoke: %s", model, e)
                last_err = e
        if last_err:
            raise last_err

    async def ainvoke(self, input, config=None, **kwargs):
        last_err = None
        for model in self.models:
            try:
                logger.info("RobustFallbackRunnable: Attempting ainvoke with model %s", model)
                return await model.ainvoke(input, config, **kwargs)
            except Exception as e:
                logger.warning("RobustFallbackRunnable: Model %s failed in ainvoke: %s", model, e)
                last_err = e
        if last_err:
            raise last_err

    def stream(self, input, config=None, **kwargs):
        last_err = None
        for model in self.models:
            try:
                logger.info("RobustFallbackRunnable: Attempting stream with model %s", model)
                iterator = model.stream(input, config, **kwargs)
                first_chunk = next(iterator)
                def generator():
                    yield first_chunk
                    try:
                        for chunk in iterator:
                            yield chunk
                    except Exception as e:
                        logger.error("RobustFallbackRunnable: Model %s failed mid-stream: %s", model, e)
                        raise e
                return generator()
            except StopIteration:
                def empty_gen():
                    if False:
                        yield None
                return empty_gen()
            except Exception as e:
                logger.warning("RobustFallbackRunnable: Model %s failed to start stream: %s", model, e)
                last_err = e
        if last_err:
            raise last_err

    async def astream(self, input, config=None, **kwargs):
        last_err = None
        for model in self.models:
            try:
                logger.info("RobustFallbackRunnable: Attempting astream with model %s", model)
                async for chunk in model.astream(input, config, **kwargs):
                    yield chunk
                return
            except Exception as e:
                logger.warning("RobustFallbackRunnable: Model %s failed in astream: %s", model, e)
                last_err = e
        if last_err:
            raise last_err


def _build_fallback_chain(temperature: float):
    """Builds RobustFallbackRunnable chain in LLM_FALLBACK_ORDER, skipping any
    provider whose key is missing at build time (so an unconfigured
    provider never becomes the "primary" that immediately fails)."""
    order = [p.strip().lower() for p in settings.llm_fallback_order.split(",") if p.strip()]
    logger.info("Constructing LLM fallback chain. Order configuration: %s", order)
    built = []
    for provider in order:
        try:
            built.append(_build_provider_llm(provider, temperature))
        except Exception as e:
            logger.warning("Skipping LLM provider '%s' in fallback chain: %s", provider, e)

    if not built:
        raise RuntimeError(
            "No LLM provider is configured. Set at least one of "
            "NVIDIA_API_KEY / GROQ_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY in .env"
        )

    logger.info("Fallback chain construction completed with RobustFallbackRunnable. Active providers: %s", [x.config.get("run_name") for x in built])
    return RobustFallbackRunnable(built)


@lru_cache(maxsize=16)
def get_llm(temperature: float = 0.0, provider: str = None):
    if provider is None:
        provider = settings.llm_provider.lower()
    else:
        provider = provider.lower().strip()
    logger.info("get_llm: Fetching LLM handler with provider=%s, temp=%.2f", provider, temperature)
    if provider == "auto":
        return _build_fallback_chain(temperature)
    return _build_provider_llm(provider, temperature)


@lru_cache
def get_embeddings():
    provider = settings.embedding_provider.lower()
    logger.info("get_embeddings: Initializing embedding provider=%s", provider)

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        logger.info("Using OpenAIEmbeddings model: %s", settings.openai_embedding_model)
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
    else:
        # Use fastembed — lightweight ONNX-based embeddings, no PyTorch/CUDA needed.
        # Supports all-MiniLM-L6-v2 out of the box and is compatible with
        # the assignment requirement (sentence-transformers all-MiniLM-L6-v2).
        try:
            from langchain_community.embeddings import FastEmbedEmbeddings
            model_name = settings.local_embedding_model
            if model_name == "all-MiniLM-L6-v2":
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
            logger.info("Using FastEmbedEmbeddings (ONNX) model: %s", model_name)
            return FastEmbedEmbeddings(model_name=model_name)
        except Exception as e:
            logger.warning("FastEmbedEmbeddings failed (%s), falling back to HuggingFaceEmbeddings", e)
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                from langchain_community.embeddings import HuggingFaceEmbeddings
            logger.info("Using local HuggingFaceEmbeddings model: %s", settings.local_embedding_model)
            return HuggingFaceEmbeddings(
                model_name=settings.local_embedding_model,
                model_kwargs={"device": "cpu", "model_kwargs": {"low_cpu_mem_usage": False}},
            )


def llm_provider_status() -> dict:
    """Used by /api/health to show which providers are configured & which
    one is currently primary, without making a network call."""
    order = [p.strip().lower() for p in settings.llm_fallback_order.split(",") if p.strip()]
    key_map = {
        "nvidia": bool(settings.nvidia_api_key),
        "groq": bool(settings.groq_api_key),
        "google": bool(settings.google_api_key),
        "openai": bool(settings.openai_api_key),
    }
    return {
        "mode": settings.llm_provider,
        "fallback_order": order if settings.llm_provider == "auto" else [settings.llm_provider],
        "configured": key_map,
        "langsmith_tracing": settings.langchain_tracing_v2 and bool(settings.langchain_api_key),
    }


# Rough published per-1M-token pricing, used only to give the UI a ballpark
# cost estimate (not billing-accurate) -- $ per 1,000,000 tokens.
MODEL_PRICING = {
    "openai": {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00)},
    "google": {"gemini-2.0-flash": (0.075, 0.30), "gemini-1.5-pro": (1.25, 5.00)},
    "groq": {"llama-3.3-70b-versatile": (0.59, 0.79)},
    "nvidia": {"meta/llama-3.1-70b-instruct": (0.0, 0.0)},  # NIM free tier
}


def current_primary_provider() -> str:
    if settings.llm_provider != "auto":
        return settings.llm_provider.lower()
    order = [p.strip().lower() for p in settings.llm_fallback_order.split(",") if p.strip()]
    key_map = {
        "nvidia": bool(settings.nvidia_api_key), "groq": bool(settings.groq_api_key),
        "google": bool(settings.google_api_key), "openai": bool(settings.openai_api_key),
    }
    for p in order:
        if key_map.get(p):
            return p
    return order[0] if order else "unknown"


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Best-effort $ cost estimate for a single query, based on the
    currently-active provider/model. Returns 0.0 for free-tier / unknown
    providers rather than guessing."""
    provider = current_primary_provider()
    model_attr = {
        "openai": settings.openai_model, "google": settings.google_model,
        "groq": settings.groq_model, "nvidia": settings.nvidia_model,
    }.get(provider)
    prices = MODEL_PRICING.get(provider, {}).get(model_attr)
    if not prices:
        return 0.0
    in_price, out_price = prices
    return round((input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price, 6)
