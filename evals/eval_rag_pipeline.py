# evals/eval_rag_pipeline.py
from dotenv import load_dotenv
import os
import sys  
import json
from datetime import datetime  
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

from src.retriever import Retrieval
from src.context_builder import ContextBuilder
from src.generator import Generator
from src.rag_pipeline import RAGPipeline
from src.config import DEFAULT_CHUNK_SIZE  
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = "golden/faithfulness_dataset.json"   # reuse the queries
GEN_MODEL = "openai/gpt-oss-120b"
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7
TOP_K = 2
MAX_CONTEXT_TOKENS = 6000


def run(rag, top_k=TOP_K):  
    # 1. LOAD queries (only the queries are needed; context comes from the pipeline)
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE INJECTED PIPELINE per query, build a test case from LIVE output
    test_cases = []
    for g in goldens:
        result = rag.run(g["query"], top_k=top_k)   # retrieve -> build context -> generate  # CHANGED: uses the top_k parameter

        # Only the chunks the generator actually saw (the token budget may drop some).
        included = set(result.built_context.included_chunk_ids)
        seen_by_generator = [
            c["text"] for c in result.retrieved_chunks
            if c["chunk_id"] in included
        ]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=result.generation.answer,
                retrieval_context=seen_by_generator,
            )
        )

    # 3. THE THREE TRIAD METRICS
    metrics = [
        ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    # 4. EVALUATE
    result = evaluate(test_cases=test_cases, metrics=metrics)
    
    return summarize_by_metric(result)


def run_local(chunk_size=DEFAULT_CHUNK_SIZE, top_k=TOP_K):
    """Standalone convenience: build the pipeline, then run."""
    pipeline = RAGPipeline(
        retriever=Retrieval(chunk_size=chunk_size),  
        context_builder=ContextBuilder(max_context_tokens=MAX_CONTEXT_TOKENS),
        generator=Generator(model=GEN_MODEL),
    )
    return run(pipeline, top_k)  


if __name__ == "__main__":

    size = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CHUNK_SIZE
    k = int(sys.argv[2]) if len(sys.argv) > 2 else TOP_K

    summary = run_local(size, k)

    print_summary("rag_pipeline", summary)

    # Save results to a result/ folder
    os.makedirs("result/pipeline_result", exist_ok=True)  

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"result/pipeline_result/rag_pipeline_{size}_k{k}_{stamp}.json"

    record = {
        "config": {
            "chunk_size": size,
            "top_k": k,
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "generator_model": GEN_MODEL,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH,
        },
        "summary": summary,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=4)

    print(f"\nSaved evaluation summary to {output_path}")