import asyncio
from langchain_core.runnables import RunnableWithFallbacks, RunnableLambda
from app.llm.base import create_llm
from app.llm.models import get_llm_configs
from app.core.logging import get_logger

logger = get_logger(__name__)

_semaphore = asyncio.Semaphore(5)


def get_llm_chain() -> RunnableWithFallbacks:
    configs = get_llm_configs()

    primary = create_llm(configs[0])
    fallbacks = [create_llm(c) for c in configs[1:]]

    chain = primary.with_fallbacks(fallbacks)

    logger.info("LLM fallback chain ready")
    return chain


_llm_chain = None


def get_llm():
    global _llm_chain
    if _llm_chain is None:
        _llm_chain = get_llm_chain()
    return _llm_chain


async def invoke_with_semaphore(query: str) -> str:
    llm = get_llm()
    async with _semaphore:
        result = await llm.ainvoke(query)
        return result.content
