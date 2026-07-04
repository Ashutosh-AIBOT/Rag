import sys
import os
import json
import asyncio
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.search_service import search_documents
from app.services.rag_chain import build_rag_chain
from app.services.evaluator import evaluate_faithfulness, evaluate_with_ragas
from app.services.scoring import evaluate_relevancy, evaluate_precision, evaluate_recall
from app.embeddings.sentence_transformer import load_embedding_model
from app.vectorstore.chroma import initialize_chroma
from app.llm.manager import llm_manager


async def run_ci_evaluation():
    print("Initializing embedding model, LLM manager, and Chroma vector store...")
    embedding_model = load_embedding_model()
    vectorstore = initialize_chroma(embedding_model)
    llm_manager.initialize()
    llm_manager.load_all_models()

    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../app/data/eval_dataset.json"))
    print(f"Loading test dataset from {dataset_path}...")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    if not dataset:
        print("Error: Empty dataset!")
        sys.exit(1)

    print(f"Loaded {len(dataset)} evaluation items. Commencing evaluation...")
    results = []
    rag_chain = build_rag_chain()

    for idx, item in enumerate(dataset):
        print(f"\n[{idx+1}/{len(dataset)}] Evaluating: '{item['question']}'")
        try:
            # 1. Search
            search_res = search_documents(
                query=item["question"],
                k=5,
                strategy="hybrid",
                vectorstore=vectorstore
            )
            docs = search_res["documents"]
            context = "\n\n".join([doc.page_content for doc in docs])

            # 2. Answer
            answer = await rag_chain.ainvoke({
                "context": context,
                "question": item["question"]
            })

            # 3. Score
            ragas_scores = evaluate_with_ragas(
                question=item["question"],
                context=context,
                answer=answer,
                reference=item["reference_answer"]
            )
            if ragas_scores:
                faith = ragas_scores.get("faithfulness", 0.0)
                rel = ragas_scores.get("relevancy", 0.0)
                prec = ragas_scores.get("precision", 0.0)
                rec = ragas_scores.get("recall", 0.0)
            else:
                faith = await evaluate_faithfulness(item["question"], context, answer)
                rel = await evaluate_relevancy(item["question"], answer)
                prec = await evaluate_precision(item["question"], context)
                rec = await evaluate_recall(item["reference_answer"], context)

            results.append({
                "question": item["question"],
                "faithfulness": faith,
                "relevancy": rel,
                "precision": prec,
                "recall": rec
            })
            print(f"  Scores -> Faithfulness: {faith:.2f}, Relevancy: {rel:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}")
        except Exception as e:
            print(f"  Error evaluating item: {e}")
            results.append({
                "question": item["question"],
                "faithfulness": 0.0,
                "relevancy": 0.0,
                "precision": 0.0,
                "recall": 0.0
            })

    total = len(results)
    avg_faith = sum(r["faithfulness"] for r in results) / total
    avg_rel = sum(r["relevancy"] for r in results) / total
    avg_prec = sum(r["precision"] for r in results) / total
    avg_rec = sum(r["recall"] for r in results) / total
    overall_avg = (avg_faith + avg_rel + avg_prec + avg_rec) / 4.0

    print("\n" + "="*50)
    print("            CI EVALUATION REPORT")
    print("="*50)
    print(f"Average Faithfulness: {avg_faith:.2f}")
    print(f"Average Relevancy:    {avg_rel:.2f}")
    print(f"Average Precision:    {avg_prec:.2f}")
    print(f"Average Recall:       {avg_rec:.2f}")
    print("-"*50)
    print(f"Overall Average RAG Score: {overall_avg:.2f}")
    print("="*50)

    threshold = 0.70
    if overall_avg >= threshold:
        print(f"SUCCESS: Overall RAG score ({overall_avg:.2f}) meets baseline threshold ({threshold:.2f}). CI PASSED!")
        sys.exit(0)
    else:
        print(f"FAILURE: Overall RAG score ({overall_avg:.2f}) falls below baseline threshold ({threshold:.2f}). Regression detected!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_ci_evaluation())
