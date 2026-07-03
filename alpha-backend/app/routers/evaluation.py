from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from app.core.logging import get_logger
from app.services.evaluation import (
    run_strategy_evaluation,
    get_eval_history,
    evaluate_rag_response,
    evaluate_context_precision,
    evaluate_context_recall
)

logger = get_logger(__name__)

# Router with prefix '/evaluate' as required by the assignment
router = APIRouter(prefix="/evaluate", tags=["Evaluation"])

# Legacy router with prefix '/evaluation' to support the current Streamlit app
legacy_router = APIRouter(prefix="/evaluation", tags=["Legacy Evaluation"])

class SingleEvalRequest(BaseModel):
    question: str = Field(..., example="What is complexity per layer for Self-Attention?")
    answer: str = Field(..., example="O(n^2 * d)")
    context: str = Field(..., example="Self-Attention has O(n^2 * d) complexity per layer...")
    reference: Optional[str] = Field(default=None, example="O(n^2 * d) where n is sequence length.")

class RunEvalRequest(BaseModel):
    strategy: str = Field(..., example="hybrid_rerank")

@router.post("")
@router.post("/")
async def evaluate_single(request: Request, payload: SingleEvalRequest):
    """
    Evaluates a single question-answer-context pair.
    Uses LLM-as-judge prompts to compute faithfulness, relevancy, context precision, and context recall.
    """
    logger.info(f"Single evaluation request received for question: '{payload.question}'")
    llm = request.app.state.llm_manager.get_llm()
    try:
        # Evaluate faithfulness and relevancy
        faith_score, rel_score, faith_reason, rel_reason = await evaluate_rag_response(
            question=payload.question,
            answer=payload.answer,
            context=payload.context,
            llm=llm
        )
        
        # Evaluate context precision (split context chunks by double newlines)
        chunks = [c.strip() for c in payload.context.split("\n\n") if c.strip()]
        precision_score, precision_reason = await evaluate_context_precision(
            question=payload.question,
            chunks=chunks,
            llm=llm
        )
        
        # Evaluate context recall if reference is provided
        recall_score = 0.0
        recall_reason = "No reference provided for recall calculation"
        if payload.reference:
            recall_score, recall_reason = await evaluate_context_recall(
                question=payload.question,
                reference=payload.reference,
                context=payload.context,
                llm=llm
            )
            
        return {
            "faithfulness": faith_score,
            "relevancy": rel_score,
            "context_precision": precision_score,
            "context_recall": recall_score,
            "reasons": {
                "faithfulness": faith_reason,
                "relevancy": rel_reason,
                "context_precision": precision_reason,
                "context_recall": recall_reason
            }
        }
    except Exception as e:
        logger.error(f"Single evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def run_evaluation_batch(request: Request, payload: RunEvalRequest):
    """Runs a batch evaluation run using the 15-question dataset for the selected strategy."""
    logger.info(f"Batch evaluation requested for strategy: {payload.strategy}")
    valid_strategies = [
        "dense", "sparse", "hybrid", "hybrid_rerank",
        "multiquery_hybrid_rerank", "hyde_hybrid_rerank", "decomposed_hybrid_rerank",
        "step_back_hybrid_rerank"
    ]
    if payload.strategy not in valid_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy '{payload.strategy}'."
        )
        
    try:
        results = await run_strategy_evaluation(payload.strategy, request.app)
        return results
    except Exception as e:
        logger.error(f"Failed to run batch evaluation for strategy {payload.strategy}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results")
async def get_results():
    """Gets history of all completed evaluation runs."""
    logger.info("Retrieving evaluation results history.")
    return get_eval_history()

# Map legacy endpoints to preserve Streamlit compatibility
@legacy_router.post("/run")
async def legacy_run_evaluation(request: Request, payload: RunEvalRequest):
    return await run_evaluation_batch(request, payload)

@legacy_router.get("/history")
async def legacy_get_history():
    return await get_results()
