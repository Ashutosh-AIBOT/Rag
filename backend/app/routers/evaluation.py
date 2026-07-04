import uuid
from fastapi import APIRouter, Request
from pydantic import BaseModel
from app.core.logging import get_logger
from app.database.database import (
    insert_query_history, insert_eval_result, list_eval_results, get_query_history
)
from app.services.evaluator import evaluate_faithfulness
from app.services.scoring import evaluate_relevancy, evaluate_precision, evaluate_recall
from app.services.retrieval import retrieval_service
from app.services.rag_chain import build_rag_chain
from app.llm import invoke_with_semaphore
import asyncio
import time

logger = get_logger(__name__)
router = APIRouter(prefix="/evaluate", tags=["evaluate"])


class EvalRequest(BaseModel):
    question: str
    reference_answer: str
    strategy: str = "vector"
    k: int = 5


class EvalResponse(BaseModel):
    eval_id: str
    question: str
    answer: str
    faithfulness: float
    relevancy: float
    precision: float
    recall: float
    latency_ms: int


class BatchEvalRequest(BaseModel):
    dataset: list[EvalRequest]


class BatchEvalResponse(BaseModel):
    results: list[EvalResponse]
    avg_faithfulness: float
    avg_relevancy: float
    avg_precision: float
    avg_recall: float


@router.post("", response_model=EvalResponse)
async def evaluate_single(request: EvalRequest, req: Request):
    try:
        start_time = time.time()

        docs = retrieval_service.invoke({"query": request.question, "k": request.k})
        context = "\n\n".join([doc.page_content for doc in docs])

        chain = build_rag_chain()
        answer = await invoke_with_semaphore(chain, {
            "context": context,
            "question": request.question,
        })

        faithfulness = await evaluate_faithfulness(request.question, context, answer)
        relevancy = await evaluate_relevancy(request.question, answer)
        precision = await evaluate_precision(request.question, context)
        recall = await evaluate_recall(request.reference_answer, context)

        latency_ms = int((time.time() - start_time) * 1000)

        query_id = str(uuid.uuid4())
        eval_id = str(uuid.uuid4())

        insert_query_history(query_id, request.question, answer, request.strategy, latency_ms)
        insert_eval_result(eval_id, query_id, faithfulness, relevancy, precision, recall)

        logger.info(f"Eval completed: faithfulness={faithfulness}, relevancy={relevancy}")
        return EvalResponse(
            eval_id=eval_id,
            question=request.question,
            answer=answer,
            faithfulness=faithfulness,
            relevancy=relevancy,
            precision=precision,
            recall=recall,
            latency_ms=latency_ms,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


@router.post("/batch", response_model=BatchEvalResponse)
async def evaluate_batch(request: BatchEvalRequest, req: Request):
    try:
        results = []
        for item in request.dataset:
            result = await evaluate_single(item, req)
            results.append(result)

        avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
        avg_relevancy = sum(r.relevancy for r in results) / len(results)
        avg_precision = sum(r.precision for r in results) / len(results)
        avg_recall = sum(r.recall for r in results) / len(results)

        logger.info(f"Batch eval completed: {len(results)} items")
        return BatchEvalResponse(
            results=results,
            avg_faithfulness=avg_faithfulness,
            avg_relevancy=avg_relevancy,
            avg_precision=avg_precision,
            avg_recall=avg_recall,
        )
    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}")
        raise


@router.get("/results")
async def get_eval_results():
    try:
        results = list_eval_results()
        logger.info(f"Listed {len(results)} eval results")
        return {"results": results}
    except Exception as e:
        logger.error(f"Get eval results failed: {e}")
        raise


@router.get("/results/{eval_id}")
async def get_eval_result(eval_id: str):
    try:
        results = list_eval_results()
        for result in results:
            if result["id"] == eval_id:
                return result
        return {"error": "Eval result not found"}
    except Exception as e:
        logger.error(f"Get eval result failed: {e}")
        raise
