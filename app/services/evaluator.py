"""
LLM-as-judge RAG evaluation, implementing the 4 core RAGAS-style metrics
without requiring the RAGAS library itself (kept as an optional bonus
integration -- see evaluate_with_ragas below).

  - Faithfulness:       claims in the answer must be supported by context
  - Answer Relevancy:   does the answer actually address the question?
  - Context Precision:  are the *retrieved* chunks relevant (rank-aware)?
  - Context Recall:     does retrieved context cover the reference answer?

All judges are implemented as LCEL chains returning structured JSON via a
strict "respond with JSON only" instruction + parsing, which is the most
portable approach across providers (OpenAI/Google/Groq) without relying on
provider-specific structured-output APIs.
"""
import json
import logging
import re
from typing import List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.services.llm_factory import get_llm

logger = logging.getLogger("evaluator")


def _parse_json(raw: str) -> Any:
    raw = raw.strip()
    raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}|\[.*\]", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return None


FAITHFULNESS_PROMPT = ChatPromptTemplate.from_template(
    "You are evaluating factual faithfulness of an AI-generated answer.\n"
    "1. Break the ANSWER into a list of atomic factual claims.\n"
    "2. For each claim, decide if it is directly supported by the CONTEXT "
    "(true), contradicted or not mentioned (false).\n"
    "Respond with ONLY a JSON object: "
    '{{"claims": [{{"claim": "...", "supported": true}}, ...], '
    '"faithfulness_score": <float between 0 and 1, = supported_claims/total_claims>}}\n\n'
    "CONTEXT:\n{context}\n\nANSWER:\n{answer}"
)

RELEVANCY_PROMPT = ChatPromptTemplate.from_template(
    "You are evaluating whether an ANSWER is relevant to a QUESTION "
    "(regardless of factual correctness).\n"
    "Generate 3 questions that the ANSWER would be a good response to, then "
    "judge how similar in intent/topic each is to the original QUESTION.\n"
    "Respond with ONLY a JSON object: "
    '{{"generated_questions": ["...", "...", "..."], '
    '"relevancy_score": <float 0-1, average similarity to the original question>}}\n\n'
    "QUESTION: {question}\n\nANSWER:\n{answer}"
)

CONTEXT_PRECISION_PROMPT = ChatPromptTemplate.from_template(
    "You are evaluating retrieval quality. Given the QUESTION and a ranked "
    "list of retrieved CONTEXT CHUNKS (rank 1 = most confident), judge for "
    "each chunk whether it is actually relevant/useful for answering the "
    "question. Precision should reward relevant chunks appearing at higher "
    "(earlier) ranks.\n"
    "Respond with ONLY a JSON object: "
    '{{"chunk_relevance": [true, false, ...], '
    '"context_precision_score": <float 0-1>}}\n\n'
    "QUESTION: {question}\n\nCHUNKS (in rank order):\n{chunks}"
)

CONTEXT_RECALL_PROMPT = ChatPromptTemplate.from_template(
    "You are evaluating whether retrieved CONTEXT contains all information "
    "present in the REFERENCE ANSWER (ground truth). Break the REFERENCE "
    "ANSWER into atomic statements, and check whether each is present "
    "somewhere in the CONTEXT.\n"
    "Respond with ONLY a JSON object: "
    '{{"statements": [{{"statement": "...", "found_in_context": true}}, ...], '
    '"context_recall_score": <float 0-1, = found_statements/total_statements>}}\n\n'
    "REFERENCE ANSWER:\n{reference_answer}\n\nCONTEXT:\n{context}"
)


def evaluate_faithfulness(context: str, answer: str) -> Dict[str, Any]:
    chain = FAITHFULNESS_PROMPT | get_llm() | StrOutputParser()
    try:
        raw = chain.invoke({"context": context, "answer": answer})
    except Exception as e:
        logger.warning("Faithfulness eval LLM call failed: %s", e)
        return {"score": 0.0, "claims": [], "error": str(e)}
    parsed = _parse_json(raw) or {}
    score = float(parsed.get("faithfulness_score", 0.0))
    logger.info("Faithfulness eval complete: score=%.3f claims=%d", score, len(parsed.get("claims", [])))
    return {
        "score": score,
        "claims": parsed.get("claims", []),
    }


def evaluate_relevancy(question: str, answer: str) -> Dict[str, Any]:
    chain = RELEVANCY_PROMPT | get_llm() | StrOutputParser()
    try:
        raw = chain.invoke({"question": question, "answer": answer})
    except Exception as e:
        logger.warning("Relevancy eval LLM call failed: %s", e)
        return {"score": 0.0, "generated_questions": [], "error": str(e)}
    parsed = _parse_json(raw) or {}
    score = float(parsed.get("relevancy_score", 0.0))
    logger.info("Relevancy eval complete: score=%.3f", score)
    return {
        "score": score,
        "generated_questions": parsed.get("generated_questions", []),
    }


def evaluate_context_precision(question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    chunk_text = "\n\n".join(f"[{i+1}] {c['text'][:600]}" for i, c in enumerate(chunks))
    chain = CONTEXT_PRECISION_PROMPT | get_llm() | StrOutputParser()
    try:
        raw = chain.invoke({"question": question, "chunks": chunk_text})
    except Exception as e:
        logger.warning("Context precision eval LLM call failed: %s", e)
        return {"score": 0.0, "llm_holistic_score": 0.0, "chunk_relevance": [], "error": str(e)}
    parsed = _parse_json(raw) or {}
    chunk_relevance = parsed.get("chunk_relevance", [])

    # Rank-aware precision computed deterministically from the LLM's
    # per-chunk relevance judgments (standard "precision@k" / average
    # precision formula used by RAGAS), rather than trusting the LLM's own
    # holistic "context_precision_score" number, which has no guaranteed
    # relationship to rank at all.
    computed_score = _average_precision(chunk_relevance) if chunk_relevance else None
    llm_score = float(parsed.get("context_precision_score", 0.0))

    return {
        "score": computed_score if computed_score is not None else llm_score,
        "llm_holistic_score": llm_score,
        "chunk_relevance": chunk_relevance,
    }


def _average_precision(chunk_relevance: List[bool]) -> float:
    """precision@k averaged over every rank where a relevant chunk appears,
    i.e. Average Precision -- rewards relevant chunks appearing at higher
    (earlier) ranks, unlike a flat relevant/total ratio."""
    num_relevant_seen = 0
    precisions = []
    for i, relevant in enumerate(chunk_relevance, start=1):
        if relevant:
            num_relevant_seen += 1
            precisions.append(num_relevant_seen / i)
    if not precisions:
        return 0.0
    return sum(precisions) / len(precisions)


def source_match_recall(chunks: List[Dict[str, Any]], relevant_sources: List[str]) -> Optional[float]:
    """Ground-truth cross-check using the eval dataset's `relevant_sources`
    field (previously stored but never actually used -- context_recall was
    100% LLM-judge opinion with no ground-truth verification that the right
    *document* was even retrieved)."""
    if not relevant_sources:
        return None
    retrieved_sources = {c.get("source") for c in chunks}
    hits = sum(1 for s in relevant_sources if s in retrieved_sources)
    return hits / len(relevant_sources)


def evaluate_context_recall(reference_answer: str, context: str) -> Dict[str, Any]:
    chain = CONTEXT_RECALL_PROMPT | get_llm() | StrOutputParser()
    try:
        raw = chain.invoke({"reference_answer": reference_answer, "context": context})
    except Exception as e:
        logger.warning("Context recall eval LLM call failed: %s", e)
        return {"score": 0.0, "statements": [], "error": str(e)}
    parsed = _parse_json(raw) or {}
    score = float(parsed.get("context_recall_score", 0.0))
    logger.info("Context recall eval complete: score=%.3f", score)
    return {
        "score": score,
        "statements": parsed.get("statements", []),
    }


def evaluate_full(question: str, reference_answer: str, generated_answer: str,
                   context: str, chunks: List[Dict[str, Any]],
                   relevant_sources: Optional[List[str]] = None,
                   pass_threshold: float = 0.6, include_ragas: bool = False,
                   context_list: Optional[List[str]] = None) -> Dict[str, Any]:
    faithfulness = evaluate_faithfulness(context, generated_answer)
    relevancy = evaluate_relevancy(question, generated_answer)
    precision = evaluate_context_precision(question, chunks)
    recall = evaluate_context_recall(reference_answer, context)

    src_recall = source_match_recall(chunks, relevant_sources or [])
    # Blend the ground-truth "was the right document even retrieved" signal
    # into context_recall (50/50) when we have ground truth to check against;
    # otherwise fall back to the pure LLM-judge recall as before.
    final_recall = recall["score"] if src_recall is None else (recall["score"] + src_recall) / 2

    avg = (faithfulness["score"] + relevancy["score"] + precision["score"] + final_recall) / 4

    detail = {
        "faithfulness": faithfulness,
        "relevancy": relevancy,
        "precision": precision,
        "recall": recall,
        "source_match_recall": src_recall,
    }

    if include_ragas:
        detail["ragas"] = evaluate_with_ragas(
            question, reference_answer, generated_answer, context_list or [context]
        )

    passed = avg >= pass_threshold
    logger.info("Eval complete: faith=%.3f rel=%.3f prec=%.3f recall=%.3f avg=%.3f passed=%s",
                faithfulness["score"], relevancy["score"], precision["score"], final_recall, avg, passed)

    return {
        "faithfulness": faithfulness["score"],
        "answer_relevancy": relevancy["score"],
        "context_precision": precision["score"],
        "context_recall": final_recall,
        "average_score": avg,
        "passed": passed,
        "detail": detail,
    }


def evaluate_with_ragas(question: str, reference_answer: str, generated_answer: str,
                         context_list: List[str]) -> Dict[str, Any]:
    """Optional bonus integration point for the RAGAS library. Kept isolated
    so the core evaluator above works even if `ragas` isn't installed."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

        ds = Dataset.from_dict({
            "question": [question],
            "answer": [generated_answer],
            "contexts": [context_list],
            "ground_truth": [reference_answer],
        })
        result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
        return dict(result)
    except ImportError:
        return {"error": "ragas not installed - pip install ragas datasets to enable this bonus feature"}
