from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.llm import get_llm_chain
from app.core.logging import get_logger

logger = get_logger(__name__)

RELEVANCY_PROMPT = """You are an answer relevancy evaluator. Given a question and an answer, determine if the answer is relevant to the question.

Question: {question}
Answer: {answer}

Score the relevancy from 0 to 1 (1 = completely relevant, 0 = completely irrelevant).
Return ONLY the score as a decimal number."""


def build_relevancy_chain():
    prompt = ChatPromptTemplate.from_template(RELEVANCY_PROMPT)
    chain = prompt | get_llm_chain() | StrOutputParser()
    return chain


async def evaluate_relevancy(question: str, answer: str) -> float:
    chain = build_relevancy_chain()
    try:
        result = await chain.ainvoke({
            "question": question,
            "answer": answer,
        })
        score = float(result.strip())
        logger.info(f"Relevancy score: {score}")
        return score
    except Exception as e:
        logger.error(f"Relevancy evaluation failed: {e}")
        return 0.0


PRECISION_PROMPT = """You are a context precision evaluator. Given a question and a list of context documents, determine how precise the context is in answering the question.

Question: {question}
Context: {context}

Score the precision from 0 to 1 (1 = highly precise, 0 = not precise).
Return ONLY the score as a decimal number."""


def build_precision_chain():
    prompt = ChatPromptTemplate.from_template(PRECISION_PROMPT)
    chain = prompt | get_llm_chain() | StrOutputParser()
    return chain


async def evaluate_precision(question: str, context: str) -> float:
    chain = build_precision_chain()
    try:
        result = await chain.ainvoke({
            "question": question,
            "context": context,
        })
        score = float(result.strip())
        logger.info(f"Precision score: {score}")
        return score
    except Exception as e:
        logger.error(f"Precision evaluation failed: {e}")
        return 0.0


RECALL_PROMPT = """You are a context recall evaluator. Given a reference answer and a context, determine how well the context covers the information needed to answer the question.

Reference Answer: {reference}
Context: {context}

Score the recall from 0 to 1 (1 = complete coverage, 0 = no coverage).
Return ONLY the score as a decimal number."""


def build_recall_chain():
    prompt = ChatPromptTemplate.from_template(RECALL_PROMPT)
    chain = prompt | get_llm_chain() | StrOutputParser()
    return chain


async def evaluate_recall(reference: str, context: str) -> float:
    chain = build_recall_chain()
    try:
        result = await chain.ainvoke({
            "reference": reference,
            "context": context,
        })
        score = float(result.strip())
        logger.info(f"Recall score: {score}")
        return score
    except Exception as e:
        logger.error(f"Recall evaluation failed: {e}")
        return 0.0
