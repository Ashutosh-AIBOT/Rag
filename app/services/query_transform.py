"""
Query transformation strategies, all built with LCEL (prompt | llm | parser)
-- no legacy LLMChain classes.
"""
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document as LCDocument

from app.services.llm_factory import get_llm


MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(
    "You are an AI assistant helping generate alternative phrasings of a "
    "user question to improve document retrieval.\n"
    "Generate exactly 4 different rephrasings of the question below. "
    "Vary vocabulary and sentence structure while preserving intent. "
    "Return ONE rephrasing per line, no numbering, no extra commentary.\n\n"
    "Question: {question}"
)

HYDE_PROMPT = ChatPromptTemplate.from_template(
    "Write a short, plausible passage (3-5 sentences) that would directly "
    "answer the question below, as if it were an excerpt from an authoritative "
    "document. It does not need to be factually verified -- it only needs to "
    "read like a real answer, since it will be used purely to guide semantic "
    "search embeddings.\n\nQuestion: {question}\n\nHypothetical passage:"
)

DECOMPOSITION_PROMPT = ChatPromptTemplate.from_template(
    "Break the following complex question into 2-4 simpler, independent "
    "sub-questions that together would let someone fully answer the "
    "original question. Return ONE sub-question per line, no numbering.\n\n"
    "Question: {question}"
)

STEP_BACK_PROMPT = ChatPromptTemplate.from_template(
    "Given a specific question, write one more general 'step-back' question "
    "that captures the broader topic. This is used to retrieve broader "
    "supporting context alongside the specific question.\n\n"
    "Specific question: {question}\n"
    "Step-back question:"
)


import logging
logger = logging.getLogger("query_transform")

def _lines(text: str) -> List[str]:
    return [ln.strip("-• \t") for ln in text.strip().split("\n") if ln.strip()]


def multi_query_expand(question: str) -> List[str]:
    try:
        chain = MULTI_QUERY_PROMPT | get_llm() | StrOutputParser()
        result = chain.invoke({"question": question})
        variants = _lines(result)
        return [question] + variants[:5]
    except Exception as e:
        logger.warning("multi_query_expand failed: %s. Returning fallback single-item list.", e)
        return [question]


def hyde_generate(question: str) -> str:
    try:
        chain = HYDE_PROMPT | get_llm() | StrOutputParser()
        return chain.invoke({"question": question})
    except Exception as e:
        logger.warning("hyde_generate failed: %s. Returning fallback original query.", e)
        return question


def decompose_query(question: str) -> List[str]:
    try:
        chain = DECOMPOSITION_PROMPT | get_llm() | StrOutputParser()
        result = chain.invoke({"question": question})
        sub_qs = _lines(result)
        return sub_qs if sub_qs else [question]
    except Exception as e:
        logger.warning("decompose_query failed: %s. Returning fallback original query list.", e)
        return [question]


def step_back_query(question: str) -> str:
    try:
        chain = STEP_BACK_PROMPT | get_llm() | StrOutputParser()
        return chain.invoke({"question": question}).strip()
    except Exception as e:
        logger.warning("step_back_query failed: %s. Returning fallback original query.", e)
        return question
