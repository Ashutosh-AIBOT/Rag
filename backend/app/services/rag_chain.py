from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logging import get_logger
from app.core.exceptions import NoLLMProviderException

logger = get_logger(__name__)

RAG_SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question based on the provided context.
If the context doesn't contain enough information, say so honestly.
Always cite the source when possible."""

RAG_USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""


def build_rag_prompt(query: str, chunks: list[dict]) -> list[SystemMessage | HumanMessage]:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("source", "unknown")
        page = chunk["metadata"].get("page", "?")
        context_parts.append(f"[{i}] (Source: {source}, Page: {page})\n{chunk['content']}")

    context = "\n\n".join(context_parts)

    return [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(content=RAG_USER_PROMPT_TEMPLATE.format(context=context, question=query)),
    ]


async def query_rag(llm_manager, semaphore, query: str, chunks: list[dict]) -> dict:
    messages = build_rag_prompt(query, chunks)

    async with semaphore:
        logger.info(f"Querying LLM with {len(chunks)} context chunks")
        try:
            answer = await llm_manager.generate(messages)
            provider_status = llm_manager.status
            logger.info(f"LLM response received from: {provider_status.get('active_provider')}")
            return {
                "answer": answer,
                "provider": provider_status.get("active_provider", "unknown"),
                "model": provider_status.get("active_provider", "unknown"),
            }
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            raise NoLLMProviderException()
