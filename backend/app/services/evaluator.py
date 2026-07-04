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


def evaluate_with_ragas(question: str, context: str, answer: str, reference: str) -> dict:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness as r_faith, answer_relevancy as r_rel, context_precision as r_prec, context_recall as r_rec
        from app.llm import get_llm_chain
        from app.embeddings.sentence_transformer import get_embedding_model

        data_dict = {
            "user_input": [question],
            "response": [answer],
            "retrieved_contexts": [[context]],
            "reference": [reference]
        }
        dataset = Dataset.from_dict(data_dict)
        llm = get_llm_chain()
        embeddings = get_embedding_model()

        result = evaluate(
            dataset=dataset,
            metrics=[r_faith, r_rel, r_prec, r_rec],
            llm=llm,
            embeddings=embeddings,
        )
        scores = {
            "faithfulness": float(result.get("faithfulness", 0.0)),
            "relevancy": float(result.get("answer_relevancy", 0.0)),
            "precision": float(result.get("context_precision", 0.0)),
            "recall": float(result.get("context_recall", 0.0)),
        }
        logger.info(f"Ragas evaluation success: {scores}")
        return scores
    except Exception as e:
        logger.error(f"Ragas evaluation failed, using custom metric fallback: {e}")
        return {}
