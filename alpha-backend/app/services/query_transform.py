from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, NumberedListOutputParser
from app.prompts import MULTI_QUERY_PROMPT, HYDE_PROMPT, DECOMPOSITION_PROMPT, STEP_BACK_PROMPT
from app.core.logging import get_logger

logger = get_logger(__name__)

async def generate_multi_queries(query: str, llm) -> List[str]:
    """Generates 3 query variants using the LLM."""
    logger.info(f"Generating multi-queries for: {query}")
    try:
        prompt = ChatPromptTemplate.from_template(MULTI_QUERY_PROMPT)
        chain = prompt | llm | NumberedListOutputParser()
        
        queries = await chain.ainvoke({"question": query})
        queries = [q.strip() for q in queries if q.strip()]
        
        # Include original query as fallback
        if query not in queries:
            queries.insert(0, query)
            
        logger.info(f"Generated queries: {queries}")
        return queries[:4] # Original + 3 variants
    except Exception as e:
        logger.error(f"Multi-query generation failed: {e}", exc_info=True)
        return [query]

async def generate_hyde_answer(query: str, llm) -> str:
    """Generates a hypothetical answer for HyDE retrieval."""
    logger.info(f"Generating HyDE hypothetical answer for: {query}")
    try:
        prompt = ChatPromptTemplate.from_template(HYDE_PROMPT)
        chain = prompt | llm | StrOutputParser()
        
        response = await chain.ainvoke({"question": query})
        logger.info("HyDE hypothetical answer generated successfully.")
        return response.strip()
    except Exception as e:
        logger.error(f"HyDE generation failed: {e}", exc_info=True)
        return query

async def decompose_query(query: str, llm) -> List[str]:
    """Decomposes a complex query into simpler sub-queries."""
    logger.info(f"Decomposing query: {query}")
    try:
        prompt = ChatPromptTemplate.from_template(DECOMPOSITION_PROMPT)
        chain = prompt | llm | NumberedListOutputParser()
        
        sub_queries = await chain.ainvoke({"question": query})
        sub_queries = [q.strip() for q in sub_queries if q.strip()]
        
        if not sub_queries:
            sub_queries = [query]
            
        logger.info(f"Decomposed sub-queries: {sub_queries}")
        return sub_queries
    except Exception as e:
        logger.error(f"Query decomposition failed: {e}", exc_info=True)
        return [query]

async def generate_step_back_query(query: str, llm) -> str:
    """Generates a step-back query."""
    logger.info(f"Generating step-back query for: {query}")
    try:
        prompt = ChatPromptTemplate.from_template(STEP_BACK_PROMPT)
        chain = prompt | llm | StrOutputParser()
        
        response = await chain.ainvoke({"question": query})
        logger.info(f"Generated step-back query: {response.strip()}")
        return response.strip()
    except Exception as e:
        logger.error(f"Step-back query generation failed: {e}", exc_info=True)
        return query
