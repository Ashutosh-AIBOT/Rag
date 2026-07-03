import time
import re
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.core.logging import get_logger
from app.prompts.evaluation import (
    FAITHFULNESS_PROMPT,
    RELEVANCY_PROMPT,
    CONTEXT_PRECISION_PROMPT,
    CONTEXT_RECALL_PROMPT
)
from app.services.rag_chain import execute_rag_pipeline
from app.database.database import save_evaluation_run, get_evaluation_runs

logger = get_logger(__name__)

# Pydantic schema for structured evaluation outputs
class EvaluationResult(BaseModel):
    reason: str = Field(description="A brief explanation justifying the assigned score.")
    score: float = Field(description="The evaluation score between 0.0 and 1.0.")

async def evaluate_context_precision(question: str, chunks: List[str], llm) -> Tuple[float, str]:
    """
    Computes context precision by evaluating the relevance of each retrieved chunk.
    Score is the average of scores of individual chunks.
    """
    if not chunks:
        return 0.0, "No chunks retrieved"
    
    total_score = 0.0
    reasons = []
    
    try:
        structured_llm = llm.with_structured_output(EvaluationResult)
    except Exception as e:
        logger.warning(f"Structured output not supported natively by primary LLM: {e}. Using raw prompt fallback.")
        structured_llm = None
    
    # We evaluate each chunk (usually up to 5)
    for idx, chunk in enumerate(chunks[:5]):
        try:
            if structured_llm:
                prompt = ChatPromptTemplate.from_template(
                    CONTEXT_PRECISION_PROMPT + "\nRespond with a structured JSON object containing 'reason' and 'score' fields."
                )
                chain = prompt | structured_llm
                res = await chain.ainvoke({"question": question, "context_chunk": chunk})
                score = float(res.score)
                reason = res.reason
            else:
                prompt = ChatPromptTemplate.from_template(CONTEXT_PRECISION_PROMPT)
                chain = prompt | llm | StrOutputParser()
                res_str = await chain.ainvoke({"question": question, "context_chunk": chunk})
                lines = [line.strip() for line in res_str.split("\n") if line.strip()]
                score = 0.0
                reason = "Evaluated without details"
                if len(lines) >= 2:
                    reason = lines[0]
                    match = re.search(r"\d+\.\d+|\d+", lines[1])
                    if match:
                        score = float(match.group())
            
            reasons.append(f"Chunk {idx+1}: {reason} (Score: {score})")
            total_score += score
        except Exception as e:
            logger.error(f"Failed to evaluate context precision for chunk {idx}: {e}")
            reasons.append(f"Chunk {idx+1}: Error (Score: 0.0)")
            
    avg_score = total_score / min(len(chunks), 5)
    return avg_score, "; ".join(reasons)

async def evaluate_context_recall(question: str, reference: str, context: str, llm) -> Tuple[float, str]:
    """
    Computes context recall by checking if the context contains enough information
    to cover the ground-truth reference answer.
    """
    try:
        try:
            structured_llm = llm.with_structured_output(EvaluationResult)
        except Exception:
            structured_llm = None
            
        if structured_llm:
            prompt = ChatPromptTemplate.from_template(
                CONTEXT_RECALL_PROMPT + "\nRespond with a structured JSON object containing 'reason' and 'score' fields."
            )
            chain = prompt | structured_llm
            res = await chain.ainvoke({"question": question, "reference": reference, "context": context})
            return float(res.score), res.reason
        else:
            prompt = ChatPromptTemplate.from_template(CONTEXT_RECALL_PROMPT)
            chain = prompt | llm | StrOutputParser()
            res_str = await chain.ainvoke({"question": question, "reference": reference, "context": context})
            lines = [line.strip() for line in res_str.split("\n") if line.strip()]
            score = 0.0
            reason = "No details returned"
            if len(lines) >= 2:
                reason = lines[0]
                match = re.search(r"\d+\.\d+|\d+", lines[1])
                if match:
                    score = float(match.group())
            return score, reason
    except Exception as e:
        logger.error(f"Failed to evaluate context recall: {e}")
        return 0.0, f"Error: {e}"

async def evaluate_rag_response(
    question: str,
    answer: str,
    context: str,
    llm
) -> Tuple[float, float, str, str]:
    """
    Evaluates a generated RAG answer using LLM-as-judge prompts.
    Returns (faithfulness_score, relevancy_score, faithfulness_reason, relevancy_reason)
    """
    try:
        structured_llm = llm.with_structured_output(EvaluationResult)
    except Exception:
        structured_llm = None

    # 1. Evaluate Faithfulness
    faithfulness_score = 1.0
    faithfulness_reason = "Supported by context"
    try:
        if structured_llm:
            prompt = ChatPromptTemplate.from_template(
                FAITHFULNESS_PROMPT + "\nRespond with a structured JSON object containing 'reason' and 'score' fields."
            )
            chain = prompt | structured_llm
            res = await chain.ainvoke({"context": context, "answer": answer})
            faithfulness_score = float(res.score)
            faithfulness_reason = res.reason
        else:
            f_prompt = ChatPromptTemplate.from_template(FAITHFULNESS_PROMPT)
            f_chain = f_prompt | llm | StrOutputParser()
            f_res = await f_chain.ainvoke({"context": context, "answer": answer})
            lines = [line.strip() for line in f_res.split("\n") if line.strip()]
            if len(lines) >= 2:
                faithfulness_reason = lines[0]
                match = re.search(r"\d+\.\d+|\d+", lines[1])
                if match:
                    faithfulness_score = float(match.group())
    except Exception as e:
        logger.error(f"Faithfulness evaluation failed: {e}")
        faithfulness_reason = f"Error: {e}"

    # 2. Evaluate Relevancy
    relevancy_score = 1.0
    relevancy_reason = "Relevant to question"
    try:
        if structured_llm:
            prompt = ChatPromptTemplate.from_template(
                RELEVANCY_PROMPT + "\nRespond with a structured JSON object containing 'reason' and 'score' fields."
            )
            chain = prompt | structured_llm
            res = await chain.ainvoke({"question": question, "answer": answer})
            relevancy_score = float(res.score)
            relevancy_reason = res.reason
        else:
            r_prompt = ChatPromptTemplate.from_template(RELEVANCY_PROMPT)
            r_chain = r_prompt | llm | StrOutputParser()
            r_res = await r_chain.ainvoke({"question": question, "answer": answer})
            lines = [line.strip() for line in r_res.split("\n") if line.strip()]
            if len(lines) >= 2:
                relevancy_reason = lines[0]
                match = re.search(r"\d+\.\d+|\d+", lines[1])
                if match:
                    relevancy_score = float(match.group())
    except Exception as e:
        logger.error(f"Relevancy evaluation failed: {e}")
        relevancy_reason = f"Error: {e}"

    return faithfulness_score, relevancy_score, faithfulness_reason, relevancy_reason

async def run_strategy_evaluation(strategy: str, app) -> Dict[str, Any]:
    """
    Runs the 15-question evaluation dataset through the RAG pipeline.
    Calculates average faithfulness, relevancy, context precision, context recall, and latency.
    """
    logger.info(f"Starting batch evaluation for strategy: {strategy}...")
    start_time = time.time()
    
    # Fetch LLM
    llm = app.state.llm_manager.get_llm()
    
    detailed_scores = {}
    total_faithfulness = 0.0
    total_relevancy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    
    for item in EVALUATION_DATASET:
        q_id = item["id"]
        question = item["question"]
        reference = item["reference"]
        
        try:
            # 1. Run query through pipeline
            res = await execute_rag_pipeline(
                query=question,
                strategy=strategy,
                filters={},
                app=app
            )
            answer = res["answer"]
            trace = res["trace"]
            
            # Format context
            context_chunks = [c["content"] for c in trace.get("context_assembled", [])]
            context_str = "\n".join(context_chunks)
            
            # 2. Score using judge LLM
            faith_score, rel_score, faith_reason, rel_reason = await evaluate_rag_response(
                question=question,
                answer=answer,
                context=context_str,
                llm=llm
            )
            
            # 3. Score context precision and recall
            precision_score, precision_reason = await evaluate_context_precision(
                question=question,
                chunks=context_chunks,
                llm=llm
            )
            
            recall_score, recall_reason = await evaluate_context_recall(
                question=question,
                reference=reference,
                context=context_str,
                llm=llm
            )
            
            detailed_scores[q_id] = {
                "question": question,
                "category": item["category"],
                "answer": answer,
                "reference": reference,
                "faithfulness": faith_score,
                "relevancy": rel_score,
                "precision": precision_score,
                "recall": recall_score,
                "faithfulness_reason": faith_reason,
                "relevancy_reason": rel_reason,
                "precision_reason": precision_reason,
                "recall_reason": recall_reason,
                "latency_ms": trace.get("latency_ms", 0)
            }
            total_faithfulness += faith_score
            total_relevancy += rel_score
            total_precision += precision_score
            total_recall += recall_score
            
        except Exception as e:
            logger.error(f"Failed to evaluate item {q_id}: {e}", exc_info=True)
            detailed_scores[q_id] = {
                "question": question,
                "category": item["category"],
                "reference": reference,
                "answer": "",
                "error": str(e)
            }

    num_items = len(EVALUATION_DATASET)
    avg_faith = total_faithfulness / num_items if num_items > 0 else 0.0
    avg_rel = total_relevancy / num_items if num_items > 0 else 0.0
    avg_prec = total_precision / num_items if num_items > 0 else 0.0
    avg_rec = total_recall / num_items if num_items > 0 else 0.0
    total_latency_ms = int((time.time() - start_time) * 1000)
    
    # Save results to centralized SQLite database
    save_evaluation_run(
        strategy=strategy,
        latency_ms=total_latency_ms,
        avg_faithfulness=avg_faith,
        avg_relevancy=avg_rel,
        avg_precision=avg_prec,
        avg_recall=avg_rec,
        details=detailed_scores
    )
    
    return {
        "strategy": strategy,
        "avg_faithfulness": avg_faith,
        "avg_relevancy": avg_rel,
        "avg_precision": avg_prec,
        "avg_recall": avg_rec,
        "total_latency_ms": total_latency_ms,
        "detailed": detailed_scores
    }

def get_eval_history() -> List[Dict[str, Any]]:
    """Gets history of all completed evaluation runs using central SQLite helper."""
    return get_evaluation_runs()
