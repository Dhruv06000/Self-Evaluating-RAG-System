# evals/eval_generator.py
"""Component-level evaluation of the GENERATOR, in isolation.

The generator gets the GOLDEN context (known-good chunks), not the retriever's
output, so a low score is the generator's fault alone.

    uv run python -m evals.eval_generator
"""
import os, json
from datetime import datetime

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

from src.generator import get_llm, generate_answer   # match your real module name
from src.context_builder import get_prompt_template
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
GEN_MODEL = "gpt-6-luna"
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7
LIMIT = None   # 3 for a smoke test

llm = get_llm(model_name=GEN_MODEL, temperature=0)
prompt = get_prompt_template()


def answer_from_context(query, context_chunks):
    final_prompt = prompt.invoke({
        "context": "\n\n".join(context_chunks),
        "question": query,
    })
    return generate_answer(llm, final_prompt)


def run():
    goldens = load_goldens(GOLDEN_PATH)[:LIMIT]

    test_cases = []
    for g in goldens:
        context = g["ideal_context"]
        answer = answer_from_context(g["query"], context)
        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=answer,
                retrieval_context=context,
            )
        )

    metrics = [
        FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "generator_model": GEN_MODEL,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH,
            "num_goldens": len(goldens),
        },
    )

    # per-question record: question, context, answer, each metric's score + reason
    by_input = {tr.input: tr for tr in result.test_results}
    details = []
    for g, tc in zip(goldens, test_cases):
        tr = by_input.get(tc.input)
        details.append({
            "query": g["query"],
            "context": g["ideal_context"],
            "answer": tc.actual_output,
            "metrics": {
                m.name: {"score": m.score, "reason": m.reason}
                for m in (getattr(tr, "metrics_data", None) or [])
            },
        })

    return summarize_by_metric(result), details


if __name__ == "__main__":
    summary, details = run()
    print_summary("generator", summary)

    if LIMIT is not None:
        print("\nSmoke test (LIMIT is set): not saving a result file.")
    else:
        n = len(details)
        complete = len(summary) == 2 and all(s["n"] == n for s in summary.values())
        prefix = "" if complete else "INCOMPLETE_"
        out_dir = "result/generator_evaluation_result"
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"{prefix}generator_{GEN_MODEL}_{stamp}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "complete": complete,
                "config": {"generator_model": GEN_MODEL, "judge_model": JUDGE_MODEL,
                           "golden_set": GOLDEN_PATH, "num_goldens": n},
                "summary": summary,
                "details": details,
            }, f, indent=4, ensure_ascii=False)
        print(f"Saved to {path}")
        if not complete:
            print("INCOMPLETE run: some metrics have no score. Don't use it as a baseline.")