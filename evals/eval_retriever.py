# evals/eval_retriever.py
"""Retriever evaluation (DeepEval) on the knowledge_base transcripts.

Run from the project root:
    uv run python -m evals.eval_retriever
"""
from dotenv import load_dotenv
import os, json
from datetime import datetime
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from src.ingestion import load_transcripts
from src.chunking import chunk_text
from src.embedding import get_embedding_model
from src.faiss_index import get_or_create_vector_store
from src.reranker import get_reranking_retriever
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = "goldens/retriever_goldens.json"
KNOWLEDGE_BASE_DIR = "knowledge_base"
KB_INDEX_PATH = "data/knowledge_base_index"

JUDGE_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"   # must match src/embedding.py
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
FETCH_K = 20
TOP_K = 3
THRESHOLD = 0.7
LIMIT = None # set to 3 for a smoke test, None for every golden


def load_kb_chunks():
    docs = load_transcripts(KNOWLEDGE_BASE_DIR)
    return chunk_text(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def run(retriever):
    goldens = load_goldens(GOLDEN_PATH)[:LIMIT]

    test_cases = []
    details = []                                              # NEW
    for golden in goldens:
        retrieved = retriever.invoke(golden["query"])
        retrieval_context = [doc.page_content for doc in retrieved]

        details.append({                                      # NEW
            "id": golden.get("id"),
            "query": golden["query"],
            "source": golden.get("source"),
            "retrieved": [
                {
                    "rank": i + 1,
                    "session": doc.metadata.get("session"),
                    "text": doc.page_content,
                }
                for i, doc in enumerate(retrieved)
            ],
        })

        test_cases.append(
            LLMTestCase(
                input=golden["query"],
                expected_output=golden["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": "faiss_flashrank",
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k": TOP_K,
            "fetch_k": FETCH_K,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH,
            "num_goldens": len(goldens),
        },
    )

    # NEW: attach each metric's score and reason to its question (matched by input text)
    by_input = {tr.input: tr for tr in result.test_results}
    for d in details:
        tr = by_input.get(d["query"])
        d["metrics"] = {
            m.name: {"score": m.score, "reason": m.reason}
            for m in (getattr(tr, "metrics_data", None) or [])
        }

    return summarize_by_metric(result), details               # CHANGED: returns two things


def run_local():
    store = get_or_create_vector_store(
        get_embedding_model(), load_kb_chunks, KB_INDEX_PATH
    )
    return run(get_reranking_retriever(store, k=TOP_K, fetch_k=FETCH_K))                 # returns (summary, details)


if __name__ == "__main__":
    summary, details = run_local()                            # CHANGED
    print_summary("retriever", summary)

    if LIMIT is not None:
        print("\nSmoke test (LIMIT is set): not saving a result file.")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")      # NEW: no more overwriting
        out_dir = "result/retriever_evaluation_result"
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"retriever_rerank_{CHUNK_SIZE}_k{TOP_K}_{stamp}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "config": {
                    "embedding_model": EMBEDDING_MODEL, 
                    "chunk_size": CHUNK_SIZE,
                    "fetch_k": FETCH_K,
                    "reranker": "flashrank-MiniLM-L-12",
                    "chunk_overlap": CHUNK_OVERLAP, 
                    "top_k": TOP_K,
                    "judge_model": JUDGE_MODEL, 
                    "golden_set": GOLDEN_PATH
                    },
                "summary": summary,
                "details": details,                           # NEW
            }, f, indent=4, ensure_ascii=False)
        print(f"Saved to {path}")