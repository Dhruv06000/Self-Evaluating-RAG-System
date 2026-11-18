# evals/eval_rag_pipeline.py
"""End-to-end evaluation of the whole RAG pipeline (retriever + generator together).

Unlike eval_generator.py (which feeds golden context), the context here is
whatever the retriever ACTUALLY returns, so a low score can come from either
component. Compare with the retriever and generator evals to see which one.

    uv run python -m evals.eval_rag_pipeline
"""
import os, json
from datetime import datetime

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)

from src.embedding import get_embedding_model
from src.faiss_index import get_or_create_vector_store
from src.reranker import get_reranking_retriever
from src.context_builder import get_prompt_template
from src.generator import get_llm
from src.pipeline import run_rag
from evals.eval_retriever import (
    load_kb_chunks, KB_INDEX_PATH, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,FETCH_K
)   # reuse the retriever eval's index + settings so both evals test the same retriever
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = "goldens/faithfulness_dataset.json"   # only the queries are used
GEN_MODEL = "gpt-6-luna"
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7
LIMIT = None # 3 for a smoke test


def run(retriever, llm, prompt):
    goldens = load_goldens(GOLDEN_PATH)[:LIMIT]

    # Run the REAL pipeline (src/pipeline.py) per query: retrieve -> build context -> generate.
    test_cases = []
    details = []
    for g in goldens:
        out = run_rag(retriever, prompt, llm, g["query"])   # same code path as main.py
        retrieved = out["docs"]
        answer = out["answer"]

        details.append({
            "id": g.get("id"),
            "query": g["query"],
            "retrieved": [
                {
                    "rank": i + 1,
                    "session": doc.metadata.get("session"),
                    "text": doc.page_content,

                }
                for i, doc in enumerate(retrieved)
            ],
            "answer": answer,
        })

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=answer,
                retrieval_context=[doc.page_content for doc in retrieved],
            )
        )

    # The three RAG-triad metrics.
    metrics = [
        ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "generator_model": GEN_MODEL,
            "judge_model": JUDGE_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k": TOP_K,
            "fetch_k": FETCH_K,
            "reranker": "flashrank-MiniLM-L-12",
            "golden_set": GOLDEN_PATH,
            "num_goldens": len(goldens),
        },
    )

    # Attach each metric's score and reason to its question (matched by input text).
    by_input = {tr.input: tr for tr in result.test_results}
    for d in details:
        tr = by_input.get(d["query"])
        d["metrics"] = {
            m.name: {"score": m.score, "reason": m.reason}
            for m in (getattr(tr, "metrics_data", None) or [])
        }

    return summarize_by_metric(result), details


def run_local():
    store = get_or_create_vector_store(
        get_embedding_model(), load_kb_chunks, KB_INDEX_PATH
    )
    retriever = get_reranking_retriever(store, k=TOP_K, fetch_k=FETCH_K)
    llm = get_llm(model_name=GEN_MODEL, temperature=0)
    prompt = get_prompt_template()
    return run(retriever, llm, prompt)


if __name__ == "__main__":
    summary, details = run_local()
    print_summary("rag_pipeline", summary)

    if LIMIT is not None:
        print("\nSmoke test (LIMIT is set): not saving a result file.")
    else:
        n = len(details)
        complete = len(summary) == 3 and all(s["n"] == n for s in summary.values())
        prefix = "" if complete else "INCOMPLETE_"
        out_dir = "result/pipeline_evaluation_result"
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            out_dir, f"{prefix}pipeline_rerank_{GEN_MODEL}_chunk{CHUNK_SIZE}_k{TOP_K}_{stamp}.json"
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "complete": complete,
                "config": {
                    "generator_model": GEN_MODEL, 
                    "judge_model": JUDGE_MODEL,
                    "embedding_model": EMBEDDING_MODEL, 
                    "chunk_size": CHUNK_SIZE,
                    "chunk_overlap": CHUNK_OVERLAP, 
                    "top_k": TOP_K,
                    "fetch_k": FETCH_K ,
                    "reranker": "flashrank-MiniLM-L-12",
                    "golden_set": GOLDEN_PATH, 
                    "num_goldens": n,
                },
                "summary": summary,
                "details": details,
            }, f, indent=4, ensure_ascii=False)
        print(f"Saved to {path}")
        if not complete:
            print("INCOMPLETE run: some metrics have no score. Don't use it as a baseline.")