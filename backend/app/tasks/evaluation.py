import json
import uuid
import time
import asyncio
from app.core.celery_app import celery_app
from app.core.logging import get_logger
from app.database.database import (
    create_job, update_job_status, get_job_status,
    insert_query_history, insert_pipeline_trace, insert_eval_result
)
from app.services.evaluator import evaluate_faithfulness, evaluate_with_ragas
from app.services.scoring import evaluate_relevancy, evaluate_precision, evaluate_recall
from app.services.search_service import search_documents
from app.services.rag_chain import build_rag_chain
from app.llm import invoke_with_semaphore
from app.embeddings.sentence_transformer import load_embedding_model
from app.vectorstore.chroma import initialize_chroma

logger = get_logger(__name__)


async def evaluate_single_internal(question: str, reference_answer: str, strategy: str, k: int, vectorstore) -> dict:
    start_time = time.time()
    # 1. Search
    search_res = search_documents(
        query=question,
        k=k,
        strategy=strategy,
        vectorstore=vectorstore
    )
    docs = search_res["documents"]
    context = "\n\n".join([doc.page_content for doc in docs])

    # 2. Answer
    chain = build_rag_chain()
    answer = await invoke_with_semaphore(chain, {
        "context": context,
        "question": question,
    })

    # 3. Score
    ragas_scores = evaluate_with_ragas(
        question=question,
        context=context,
        answer=answer,
        reference=reference_answer
    )
    if ragas_scores:
        faithfulness = ragas_scores.get("faithfulness", 0.0)
        relevancy = ragas_scores.get("relevancy", 0.0)
        precision = ragas_scores.get("precision", 0.0)
        recall = ragas_scores.get("recall", 0.0)
    else:
        faithfulness = await evaluate_faithfulness(question, context, answer)
        relevancy = await evaluate_relevancy(question, answer)
        precision = await evaluate_precision(question, context)
        recall = await evaluate_recall(reference_answer, context)

    latency_ms = int((time.time() - start_time) * 1000)

    query_id = str(uuid.uuid4())
    eval_id = str(uuid.uuid4())

    insert_query_history(query_id, question, answer, strategy, latency_ms)
    insert_eval_result(eval_id, query_id, faithfulness, relevancy, precision, recall)

    return {
        "eval_id": eval_id,
        "question": question,
        "answer": answer,
        "faithfulness": faithfulness,
        "relevancy": relevancy,
        "precision": precision,
        "recall": recall,
        "latency_ms": latency_ms
    }


@celery_app.task(name="app.tasks.evaluation.process_evaluation_job_task")
def process_evaluation_job_task(job_id: str, dataset: list):
    async def run():
        try:
            # Check cancellation state
            job = get_job_status(job_id)
            if job and job["status"] == "cancelled":
                logger.info(f"Evaluation job {job_id} cancelled before start.")
                return

            update_job_status(job_id, "processing", 0.0)

            # Load default vectorstore
            embeddings = load_embedding_model()
            vectorstore = initialize_chroma(embeddings)

            results = []
            total = len(dataset)

            for i, item in enumerate(dataset):
                # Check cancellation between items
                job = get_job_status(job_id)
                if job and job["status"] == "cancelled":
                    logger.info(f"Evaluation job {job_id} was cancelled on iteration {i}.")
                    return

                res = await evaluate_single_internal(
                    question=item["question"],
                    reference_answer=item["reference_answer"],
                    strategy=item.get("strategy", "vector"),
                    k=item.get("k", 5),
                    vectorstore=vectorstore
                )
                results.append(res)
                progress = (i + 1) / total
                update_job_status(job_id, "processing", progress)

            avg_faithfulness = sum(r["faithfulness"] for r in results) / total
            avg_relevancy = sum(r["relevancy"] for r in results) / total
            avg_precision = sum(r["precision"] for r in results) / total
            avg_recall = sum(r["recall"] for r in results) / total

            result_payload = {
                "results": results,
                "avg_faithfulness": avg_faithfulness,
                "avg_relevancy": avg_relevancy,
                "avg_precision": avg_precision,
                "avg_recall": avg_recall,
            }

            update_job_status(job_id, "completed", 1.0, json.dumps(result_payload))
        except Exception as e:
            logger.error(f"Celery evaluation job {job_id} failed: {e}")
            update_job_status(job_id, "failed", 1.0, error=str(e))
            raise

    asyncio.run(run())


@celery_app.task(name="app.tasks.evaluation.process_comparison_job_task")
def process_comparison_job_task(
    job_id: str,
    question: str,
    k: int,
    strategy_a: str,
    strategy_b: str,
    rerank_a: bool,
    rerank_b: bool,
    rerank_top_k_a: int,
    rerank_top_k_b: int,
    filters: dict,
    embedding_model_a: str,
    embedding_model_b: str,
    compress_a: bool = False,
    compress_b: bool = False,
):
    async def run_single(strat: str, r_flag: bool, r_k: int, embed_name: str, comp: bool) -> dict:
        start_time = time.time()
        
        # Load embedding and vectorstore
        embeddings = load_embedding_model() # default or specific
        if embed_name == "gemini":
            from app.vectorstore.chroma import initialize_chroma_gemini
            vectorstore = initialize_chroma_gemini()
        else:
            vectorstore = initialize_chroma(embeddings)

        search_res = search_documents(
            query=question,
            k=k,
            strategy=strat,
            filters=filters,
            rerank=r_flag,
            rerank_top_k=r_k,
            vectorstore=vectorstore,
            compress=comp
        )
        docs = search_res["documents"]
        trace = search_res["trace"]
        context = "\n\n".join([doc.page_content for doc in docs])
        
        chain = build_rag_chain()
        answer = await invoke_with_semaphore(chain, {
            "context": context,
            "question": question,
        })
        
        latency_ms = int((time.time() - start_time) * 1000)
        query_id = str(uuid.uuid4())
        
        insert_query_history(query_id, question, answer, strat, latency_ms)
        trace["answer"] = answer
        trace["latency_ms"] = latency_ms
        insert_pipeline_trace(str(uuid.uuid4()), query_id, json.dumps(trace))
        
        return {
            "query_id": query_id,
            "answer": answer,
            "docs": docs,
            "trace": trace,
            "latency_ms": latency_ms
        }

    async def run():
        try:
            job = get_job_status(job_id)
            if job and job["status"] == "cancelled":
                logger.info(f"Comparison job {job_id} cancelled before start.")
                return

            update_job_status(job_id, "processing", 0.1)

            res_a = await run_single(strategy_a, rerank_a, rerank_top_k_a, embedding_model_a, compress_a)
            res_b = await run_single(strategy_b, rerank_b, rerank_top_k_b, embedding_model_b, compress_b)

            job = get_job_status(job_id)
            if job and job["status"] == "cancelled":
                logger.info(f"Comparison job {job_id} cancelled during processing.")
                return

            result_payload = {
                "query_id_a": res_a["query_id"],
                "answer_a": res_a["answer"],
                "sources_a": [doc.metadata.get("source", "") for doc in res_a["docs"]],
                "trace_a": res_a["trace"],
                "latency_ms_a": res_a["latency_ms"],
                "query_id_b": res_b["query_id"],
                "answer_b": res_b["answer"],
                "sources_b": [doc.metadata.get("source", "") for doc in res_b["docs"]],
                "trace_b": res_b["trace"],
                "latency_ms_b": res_b["latency_ms"],
            }

            update_job_status(job_id, "completed", 1.0, json.dumps(result_payload))
        except Exception as e:
            logger.error(f"Celery comparison job {job_id} failed: {e}")
            update_job_status(job_id, "failed", 1.0, error=str(e))
            raise

    asyncio.run(run())
