import faiss
import json
from sentence_transformers import SentenceTransformer

class Retrieval:

  def __init__(self):
    self.faiss_index = faiss.read_index("data/faiss.index")
    with open("data/chunks.json", "r", encoding="utf-8") as f:
      self.chunks = json.load(f)
    self.model = SentenceTransformer("BAAI/bge-base-en-v1.5")
  def retrieve(self , query : str, top_k : int = 5):
    query = query.strip()
    if not query:
      raise ValueError("Query cannot be empty.")
    if not 1 <= top_k <= len(self.chunks):
      raise ValueError(f"top_k must be between 1 and {len(self.chunks)}.")
    query_embedding = self.model.encode([query],
                                        convert_to_numpy= True,
                                        normalize_embeddings=True)
    scores , indices = self.faiss_index.search(
      query_embedding,
      k = top_k
      )
    results = []
    for score , index in zip(scores[0], indices[0]):
      chunk = self.chunks[index]
      results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk.get("text", ""),
                "heading": chunk.get("heading", ""),
                "score": float(score),
                "metadata": chunk.get("metadata", {})
            })
    return results
if __name__ == "__main__":
    retriever = Retrieval()

    results = retriever.retrieve(
        "What is Artificial Intelligence ?",
        top_k=5
    )

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. "
            f"score={result['score']:.4f} | "
            f"heading={result['heading']}| " 
            f"chunk_id={result['chunk_id']} | "
        )
