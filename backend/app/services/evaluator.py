from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.llm import get_llm_chain
from app.core.logging import get_logger

logger = get_logger(__name__)

FAITHFULNESS_PROMPT = """You are a faithful evaluator. Given a question, context, and answer, determine if the answer is faithful to the context.

Question: {question}
Context: {context}
Answer: {answer}

Score the faithfulness from 0 to 1 (1 = completely faithful, 0 = completely unfaithful).
Return ONLY the score as a decimal number."""


def build_faithfulness_chain():
    prompt = ChatPromptTemplate.from_template(FAITHFULNESS_PROMPT)
    chain = prompt | get_llm_chain() | StrOutputParser()
    return chain


async def evaluate_faithfulness(question: str, context: str, answer: str) -> float:
    chain = build_faithfulness_chain()
    try:
        result = await chain.ainvoke({
            "question": question,
            "context": context,
            "answer": answer,
        })
        score = float(result.strip())
        logger.info(f"Faithfulness score: {score}")
        return score
    except Exception as e:
        logger.error(f"Faithfulness evaluation failed: {e}")
        return 0.0
