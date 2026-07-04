from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.llm import get_llm_chain
from app.core.logging import get_logger
import json

logger = get_logger(__name__)

DECOMPOSITION_PROMPT_TEMPLATE = """You are a helpful assistant that breaks down complex questions into simpler sub-questions.

Given the following question, generate 3-5 sub-questions that together would help answer the main question.

Question: {question}

Return ONLY a JSON array of sub-questions, like:
["sub-question 1", "sub-question 2", "sub-question 3"]"""


def build_decomposition_chain():
    prompt = ChatPromptTemplate.from_template(DECOMPOSITION_PROMPT_TEMPLATE)
    chain = (
        RunnablePassthrough.assign(question=lambda x: x["question"])
        | prompt
        | get_llm_chain()
        | StrOutputParser()
    )
    return chain


def parse_sub_queries(output: str) -> list[str]:
    try:
        sub_queries = json.loads(output)
        if isinstance(sub_queries, list):
            logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
            return sub_queries
    except json.JSONDecodeError:
        logger.warning("Failed to parse sub-queries, returning original")
    return []
