# evals/eval_retriever.py
"""Retriever evaluation (DeepEval) that saves every run and flags incomplete runs.

Run from the project root:
    uv run python -m evals.eval_retriever
    uv run python -m evals.eval_retriever 512 4      # CHANGED: optional <chunk_size> <top_k>
Needs OPENAI_API_KEY in .env (the judge is an OpenAI model).
"""
import hashlib
import json
import os
import sys  # CHANGED: needed to read the optional chunk size and top_k from the command line
import time
from datetime import datetime

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, ErrorConfig
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

from src.retriever import Retrieval
from src.config import DEFAULT_CHUNK_SIZE, chunks_path  # CHANGED: chunk size and chunk file path come from config
from evals.harness import load_goldens, print_summary, summarize_by_metric

load_dotenv()

GOLDEN_PATH = "golden/retriever_golden.json"
# CHANGED: removed CHUNKS_PATH and CHUNK_SIZE. Both are now derived from the retriever's
# chunk size (see run() and save_run()), so they can never disagree with what was evaluated.
RUNS_DIR = "result/retriever_result"

JUDGE_MODEL = "gpt-4o-mini"
TOP_K = 2                     # change to 5 for the new baseline run
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
CHUNK_OVERLAP = 0
THRESHOLD = 0.7
LIMIT = None           # set to 1 for a smoke test, None for every golden

MAX_ATTEMPTS = 3         # tries per question when a metric comes back without a score
RETRY_WAIT_SECONDS = 20  # wait before retrying a question that came back unscored
PAUSE_SECONDS = 1        # pause between questions

PLACEHOLDER_OUTPUT = "(generator not evaluated in this run)"
N_METRICS = 2


def make_judge():
    """DeepEval accepts an OpenAI model name as a string; the key comes from .env."""
    return JUDGE_MODEL


def make_metrics(judge):
    """Fresh metric objects each time, so a retry never reuses stale state."""
    return [
        ContextualRecallMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ]


def file_hash(path):
    """Short fingerprint of a file, so two runs can be checked for the same data."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def describe_retrieved(retrieved):
    """What was retrieved for one question, in rank order."""
    return [
        {
            "rank": i + 1,
            "chunk_id": r["chunk_id"],
            "heading": r.get("heading", ""),
            "pages": [p["page_label"] for p in r.get("metadata", {}).get("pages", [])],
            "preview": r["text"][:80],
            "score": round(r["score"], 4),
        }
        for i, r in enumerate(retrieved)
    ]


def case_is_complete(test_result):
    """True only if every metric produced a score (an API error leaves score None)."""
    metrics = getattr(test_result, "metrics_data", None) or []
    return len(metrics) == N_METRICS and all(
        getattr(m, "score", None) is not None for m in metrics
    )


def evaluate_case(test_case, judge, config):
    """Evaluate one question. Retry the whole question if any metric has no score."""
    test_result = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = evaluate(
            test_cases=[test_case],
            metrics=make_metrics(judge),
            hyperparameters=config,
            async_config=AsyncConfig(run_async=False),
            error_config=ErrorConfig(ignore_errors=True),
        )
        test_result = result.test_results[0]
        if case_is_complete(test_result):
            return test_result
        print(f"  attempt {attempt}/{MAX_ATTEMPTS}: a metric has no score "
              f"(likely an API error or rate limit)")
        if attempt < MAX_ATTEMPTS:
            print(f"  waiting {RETRY_WAIT_SECONDS}s before retrying...")
            time.sleep(RETRY_WAIT_SECONDS)
    return test_result   # still incomplete; flagged when the run is saved


def save_run(test_results, goldens, retrieved_info, summary, config):
    """Write one JSON file per run. Never overwrites an earlier run."""
    os.makedirs(RUNS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    details = []
    for g, info, tr in zip(goldens, retrieved_info, test_results):
        per_metric = {}
        for m in getattr(tr, "metrics_data", None) or []:
            per_metric[m.name] = {
                "score": getattr(m, "score", None),
                "reason": getattr(m, "reason", None),
                "error": getattr(m, "error", None),
            }
        details.append({
            "question": g["question"],
            "source": g.get("source"),
            "retrieved": info,
            "metrics": per_metric,
        })

    errored = sum(
        1 for d in details for m in d["metrics"].values() if m.get("score") is None
    )
    missing = sum(max(0, N_METRICS - len(d["metrics"])) for d in details)
    complete = errored == 0 and missing == 0

    record = {
        "timestamp": stamp,
        "complete": complete,
        "errored_metrics": errored + missing,
        "golden_sha256": file_hash(GOLDEN_PATH),
        "chunks_sha256": file_hash(chunks_path(config["chunk_size"])),  # CHANGED: hash the chunk file that was actually used (was CHUNKS_PATH)
        "config": config,
        "summary": summary,
        "details": details,
    }

    # CHANGED: chunk size and top_k are in the file name, so runs are easy to tell apart.
    prefix = f"retriever_{config['chunk_size']}_k{config['top_k']}_{stamp}.json"
    name = prefix if complete else f"INCOMPLETE_{prefix}"
    path = os.path.join(RUNS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return path, complete, errored + missing


def run(retriever, judge, top_k=TOP_K):  # CHANGED: top_k is now a parameter (default = TOP_K, same as before)
    goldens = load_goldens(GOLDEN_PATH)[:LIMIT]

    # One config dict, used for DeepEval's hyperparameters AND the saved record.
    config = {
        "retriever": type(retriever).__name__,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": retriever.chunk_size,  # CHANGED: read from the retriever (was the CHUNK_SIZE constant)
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": top_k,  # CHANGED: the value actually used (was TOP_K)
        "judge_model": JUDGE_MODEL,
        "golden_set": GOLDEN_PATH,
        "num_goldens": len(goldens),
    }

    test_results = []
    retrieved_info = []
    for i, g in enumerate(goldens, start=1):
        print(f"\n=== [{i}/{len(goldens)}] {g['question']}")

        # RUN THE INJECTED RETRIEVER, then build the test case
        retrieved = retriever.retrieve(g["question"], top_k=top_k)  # CHANGED: uses the top_k parameter
        retrieved_info.append(describe_retrieved(retrieved))

        test_case = LLMTestCase(
            input=g["question"],
            expected_output=g["ideal_answer"],
            retrieval_context=[r["text"] for r in retrieved],   # rank order preserved
            actual_output=PLACEHOLDER_OUTPUT,
        )

        test_results.append(evaluate_case(test_case, judge, config))

        if i < len(goldens):
            time.sleep(PAUSE_SECONDS)

    summary = summarize_by_metric(test_results)
    path, complete, errored = save_run(test_results, goldens, retrieved_info, summary, config)

    print(f"\nRun saved to {path}")
    if not complete:
        print(f"\n!!! INCOMPLETE RUN: {errored} metric value(s) have no score.")
        print("!!! Do not use these numbers or compare this run with others.")
    return summary


# CHANGED: run_local accepts the chunk size and top_k; with no arguments it behaves as before.
def run_local(chunk_size=DEFAULT_CHUNK_SIZE, top_k=TOP_K):
    """Standalone convenience: build the retriever and the judge, then run."""
    return run(Retrieval(chunk_size=chunk_size), make_judge(), top_k)


if __name__ == "__main__":
    # CHANGED: optional command-line arguments: <chunk_size> <top_k>
    size = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CHUNK_SIZE
    k = int(sys.argv[2]) if len(sys.argv) > 2 else TOP_K
    print_summary("retriever", run_local(size, k))