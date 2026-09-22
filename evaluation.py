from retrieval import Retrieval
import json
def calculate_recall_at_k(ground_truth , retrieved_chunks, k):
  top_k_chunk = retrieved_chunks[:k]
  ground_truth_set = set(ground_truth)
  if not ground_truth_set:
     return 0.0

  relevant_count = 0

  for chunk in top_k_chunk:
    if chunk in ground_truth_set:
      relevant_count += 1
  total_relevant_chunk = len(ground_truth_set)
  result = relevant_count/total_relevant_chunk
  return result


def calculate_precision_at_k(ground_truth , retrieved_chunks , k):
  top_k_chunk = retrieved_chunks[:k]
  ground_truth_set = set(ground_truth)
  if not top_k_chunk:
     return 0.0
  relevant_count = 0
  for chunk in top_k_chunk:
    if chunk in ground_truth_set:
      relevant_count += 1
  total_retrieved_chunk = len(top_k_chunk)
  result = relevant_count/total_retrieved_chunk
  return result 

def calculate_mrr(ground_truth , retrieved_chunks, k = None ):
   top_k_chunks = retrieved_chunks[:k] if k is not None else retrieved_chunks
   ground_truth_set = set(ground_truth)


   for rank , chunk in enumerate(top_k_chunks,start= 1):
      if chunk in ground_truth_set:
         return 1 / rank
   return 0


def evaluate_question(retriever, evaluation_item, k):
    question = evaluation_item["question"]
    ground_truth_chunks = evaluation_item["ground_truth"]["relevant_chunks"]

    retrieved_results = retriever.retrieve(question, top_k=k)

    retrieved_chunks = []

    for retrieved_result in retrieved_results:
        chunk_id = retrieved_result["chunk_id"]
        if chunk_id not in retrieved_chunks:
           retrieved_chunks.append(chunk_id)

    recall = calculate_recall_at_k(
        ground_truth_chunks,
        retrieved_chunks,
        k
    )

    precision = calculate_precision_at_k(
        ground_truth_chunks,
        retrieved_chunks,
        k
    )
    mrr = calculate_mrr(ground_truth_chunks,retrieved_chunks,k)

    evaluation_result = {
        "question": question,
        "ground_truth_chunks": ground_truth_chunks,
        "retrieved_chunks": retrieved_chunks,
        "recall": recall,
        "precision": precision,
        "mrr":mrr
    }

    return evaluation_result

def calculate_average_metrics(evaluation_results):
    total_recall = 0
    total_precision = 0
    total_mrr = 0

    for result in evaluation_results:
        total_recall += result["recall"]
        total_precision += result["precision"]
        total_mrr += result["mrr"]

    total_questions = len(evaluation_results)
    if total_questions == 0:
       return 0.0,0.0,0.0

    average_recall = total_recall / total_questions
    average_precision = total_precision / total_questions
    average_mrr = total_mrr / total_questions

    return average_recall, average_precision,average_mrr


if __name__ == "__main__":
    K = 5
    with open("data/evaluation_dataset.json", "r", encoding="utf-8") as file:
        evaluation_data = json.load(file)

    retriever = Retrieval()
    evaluation_results = []

    for item in evaluation_data:
        result = evaluate_question(retriever, item, k=K)
        evaluation_results.append(result)


    average_recall, average_precision, average_mrr = calculate_average_metrics(
    evaluation_results
)

    print("\n" + "=" * 60)
    print("Overall Retrieval Baseline")
    print("=" * 60)
    print(f"Average Recall@5: {average_recall:.2f}")
    print(f"Average Precision@5: {average_precision:.2f}")
    print(f"Average MRR@5: {average_mrr:.2f}")

    # ==========================================
    # SAVE BASELINE RESULTS
    # ==========================================
    baseline_payload = {
        "k": K,
        "summary": {
            "average_recall": average_recall,
            "average_precision": average_precision,
            "average_mrr": average_mrr,
        },
        "details": evaluation_results,
    }

    output_path = "data/baseline_results.json"
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(baseline_payload, out_file, indent=4)

    print(f"\nBaseline results successfully saved to '{output_path}'.")
