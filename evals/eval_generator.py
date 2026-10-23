import re
import os 
import json
# import time
from types import SimpleNamespace
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

from src.generator import Generator
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

# Updated dataset path for the hard evaluation set
GOLDEN_PATH = "golden/faithfulness_dataset.json"
GEN_MODEL = "openai/gpt-oss-120b"
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7


def build_golden_context(chunks):
    """Turn raw chunk strings into the shape Generator expects."""
    ids = tuple(f"chunk_{i:06d}" for i, _ in enumerate(chunks, start=1))
    text = "\n\n".join(f"Chunk ID: {cid}\n{c}" for cid, c in zip(ids, chunks))
    return SimpleNamespace(context=text, included_chunk_ids=ids)


def run():
    goldens = load_goldens(GOLDEN_PATH)
    generator = Generator(model=GEN_MODEL)

    test_cases = []
    for g in goldens:
        # Extract query/question field from the dataset schema
        query = g["query"]
        chunks = g["ideal_context"]
        
        # Generate response using the ideal context chunks
        result = generator.generate(query, build_golden_context(chunks))
        # time.sleep(3)

        answer = result.answer

        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=answer,
                retrieval_context=chunks,  # raw chunks corresponding to this item
            )
        )

    metrics = [
        FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]
    return summarize_by_metric(
        evaluate(
            test_cases=test_cases, 
            metrics=metrics,
            async_config=AsyncConfig(run_async=False)
            )
        )


if __name__ == "__main__":
    summary = run()
    print_summary("generator", summary)
    
    # Save results to a result/ folder
    os.makedirs("result", exist_ok=True)
    output_path = "result/generator_evalution.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved evaluation summary to {output_path}")