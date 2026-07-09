"""
Evaluation endpoints. Single-question eval is fast enough to run inline;
batch eval (N strategies x M questions x 4 LLM-judge calls) runs as a
background job so the frontend can show a live progress bar instead of
holding one HTTP request open for minutes.
"""
import datetime as dt
import json
import logging
import os
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.models.database import get_db, EvalResult, EvalJob, SessionLocal, User
from app.models.schemas import EvalRequest, EvalResultOut, BatchEvalRequest, EvalJobOut, BatchEvalAccepted
from app.services.rag_chain import run_rag_query
from app.services.evaluator import evaluate_full
from app.services.job_manager import submit_eval
from app.routers.auth import _get_current_user

logger = logging.getLogger("evaluation_router")
router = APIRouter(prefix="/api/evaluate", tags=["evaluation"])

# Single canonical eval dataset lives at backend/app/data/eval_dataset.json;
# the top-level eval_dataset/ directory (expected by the submission
# guidelines) is a symlink to this file so there's exactly one source of
# truth instead of two copies that can silently drift out of sync.
EVAL_DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "eval_dataset.json")


def _load_eval_dataset() -> List[dict]:
    with open(EVAL_DATASET_PATH) as f:
        return json.load(f)


@router.post("")
def evaluate_single(req: EvalRequest, db: Session = Depends(get_db), current_user: User = Depends(_get_current_user)):
    logger.info("[%s] Eval single started: strategy=%s question=%r", current_user.email, req.strategy, req.question[:80])
    result = run_rag_query(req.question, req.strategy, user_id=current_user.id)
    scores = evaluate_full(req.question, req.reference_answer, result["answer"],
                            result["context"], result["chunks"],
                            relevant_sources=req.relevant_sources, include_ragas=req.use_ragas)

    row = EvalResult(
        question=req.question, reference_answer=req.reference_answer, strategy=req.strategy,
        generated_answer=result["answer"], faithfulness=scores["faithfulness"],
        answer_relevancy=scores["answer_relevancy"], context_precision=scores["context_precision"],
        context_recall=scores["context_recall"], passed=scores["passed"],
        trace={"pipeline": result["pipeline"], "eval_detail": scores["detail"]},
        batch_id="single", user_id=current_user.id,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    logger.info("[%s] Eval single completed: faith=%.3f rel=%.3f ctx_p=%.3f ctx_r=%.3f passed=%s",
                current_user.email, scores["faithfulness"], scores["answer_relevancy"],
                scores["context_precision"], scores["context_recall"], scores["passed"])
    return {"result": EvalResultOut.model_validate(row), "detail": scores["detail"]}


def _run_batch_eval_job(job_id: str, batch_id: str, strategies: List[str], dataset: List[dict],
                         use_ragas: bool = False, user_id: str = None) -> None:
    """Runs on the dedicated eval worker pool. Updates progress after every
    (strategy, question) pair so the frontend progress bar moves smoothly.

    Sleep / retry behaviour is driven by config so operators can tune it for
    their LLM provider tier without touching code:
      EVAL_SLEEP_SECONDS  — pause between each question (default 6.0s)
      EVAL_MAX_RETRIES    — per-step retry attempts with exponential backoff
    """
    import time
    from app.config import get_settings
    cfg = get_settings()
    sleep_secs = cfg.eval_sleep_seconds
    max_retries = cfg.eval_max_retries

    db = SessionLocal()
    try:
        job = db.query(EvalJob).filter(EvalJob.id == job_id).first()
        if not job:
            return
        job.status = "running"
        db.commit()

        logger.info("[uid:%s] Batch eval started: job_id=%s strategies=%s questions=%d", user_id, job_id, strategies, len(dataset))

        summary = {}
        done = 0
        for strategy in strategies:
            strategy_scores = {"faithfulness": [], "answer_relevancy": [],
                                "context_precision": [], "context_recall": []}
            for item in dataset:
                # Per-step retry with exponential backoff to survive 429s
                last_exc = None
                for attempt in range(max_retries):
                    try:
                        result = run_rag_query(item["question"], strategy, user_id=user_id)
                        scores = evaluate_full(item["question"], item["reference_answer"],
                                                result["answer"], result["context"], result["chunks"],
                                                relevant_sources=item.get("relevant_sources", []),
                                                include_ragas=use_ragas)
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                        backoff = sleep_secs * (2 ** attempt)
                        time.sleep(backoff)

                if last_exc is not None:
                    # Record a zero-score row so the table shows the failure
                    scores = {
                        "faithfulness": 0.0, "answer_relevancy": 0.0,
                        "context_precision": 0.0, "context_recall": 0.0,
                        "average_score": 0.0, "passed": False,
                        "detail": {"error": str(last_exc)},
                    }
                    result = {"answer": f"[Eval failed: {last_exc}]", "pipeline": [], "context": ""}

                row = EvalResult(
                    question=item["question"], reference_answer=item["reference_answer"], strategy=strategy,
                    generated_answer=result["answer"], faithfulness=scores["faithfulness"],
                    answer_relevancy=scores["answer_relevancy"], context_precision=scores["context_precision"],
                    context_recall=scores["context_recall"], passed=scores.get("passed", False),
                    trace={"pipeline": result.get("pipeline", []), "eval_detail": scores.get("detail", {})},
                    batch_id=batch_id, user_id=user_id,
                )
                db.add(row)
                for k in strategy_scores:
                    strategy_scores[k].append(scores[k])

                done += 1
                job.completed_steps = done
                job.updated_at = dt.datetime.now(dt.timezone.utc)
                db.commit()
                time.sleep(sleep_secs)  # configurable pause between questions

            summary[strategy] = {
                k: round(sum(v) / len(v), 3) if v else 0.0 for k, v in strategy_scores.items()
            }
            summary[strategy]["average"] = round(sum(summary[strategy].values()) / 4, 3)

        job.status = "done"
        job.summary = summary
        db.commit()
        logger.info("[uid:%s] Batch eval completed: job_id=%s summary=%s", user_id, job_id, summary)
    except Exception as e:  # noqa: BLE001
        logger.error("[uid:%s] Batch eval failed: job_id=%s error=%s", user_id, job_id, e, exc_info=True)
        db.rollback()
        job = db.query(EvalJob).filter(EvalJob.id == job_id).first()
        if job:
            job.status = "failed"
            job.error = str(e)
            db.commit()
    finally:
        db.close()



@router.post("/batch", response_model=BatchEvalAccepted, status_code=status.HTTP_202_ACCEPTED)
def evaluate_batch(req: BatchEvalRequest, db: Session = Depends(get_db), current_user: User = Depends(_get_current_user)):
    dataset = _load_eval_dataset()
    if req.limit:
        dataset = dataset[: req.limit]

    batch_id = uuid.uuid4().hex[:12]
    total_steps = len(req.strategies) * len(dataset)

    logger.info("[%s] Batch eval requested: strategies=%s questions=%d total_steps=%d",
                current_user.email, req.strategies, len(dataset), total_steps)

    job = EvalJob(batch_id=batch_id, status="queued", total_steps=total_steps,
                  completed_steps=0, strategies=req.strategies, user_id=current_user.id)
    db.add(job)
    db.commit()
    db.refresh(job)

    submit_eval(_run_batch_eval_job, job.id, batch_id, req.strategies, dataset, req.use_ragas, current_user.id)

    logger.info("[%s] Batch eval accepted: job_id=%s batch_id=%s", current_user.email, job.id, batch_id)
    return BatchEvalAccepted(job_id=job.id, batch_id=batch_id)


@router.get("/batch/{job_id}", response_model=EvalJobOut)
def get_eval_job(job_id: str, db: Session = Depends(get_db), current_user: User = Depends(_get_current_user)):
    job = db.query(EvalJob).filter(EvalJob.id == job_id, EvalJob.user_id == current_user.id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    logger.info("[%s] Get eval job: job_id=%s status=%s progress=%d/%d",
                current_user.email, job_id, job.status, job.completed_steps, job.total_steps)
    return job


@router.get("/results")
def get_results(batch_id: str = None, db: Session = Depends(get_db), current_user: User = Depends(_get_current_user)):
    q = db.query(EvalResult).filter(EvalResult.user_id == current_user.id)
    if batch_id:
        q = q.filter(EvalResult.batch_id == batch_id)
    rows = q.order_by(EvalResult.created_at.desc()).limit(500).all()
    logger.info("[%s] Get eval results: batch_id=%s count=%d", current_user.email, batch_id, len(rows))
    return [EvalResultOut.model_validate(r) for r in rows]


@router.get("/batches")
def list_batches(db: Session = Depends(get_db), current_user: User = Depends(_get_current_user)):
    """Lists past batch runs (id + strategies + when), so the frontend can
    let users browse previous evaluation runs instead of only ever seeing
    the most recent one."""
    jobs = db.query(EvalJob).filter(EvalJob.user_id == current_user.id).order_by(EvalJob.created_at.desc()).limit(50).all()
    logger.info("[%s] List eval batches: count=%d", current_user.email, len(jobs))
    return [EvalJobOut.model_validate(j) for j in jobs]
