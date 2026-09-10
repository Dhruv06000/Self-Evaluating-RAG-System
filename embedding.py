from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

with open("data/chunks.json", "r", encoding = "utf-8") as f :
  data = json.load(f)

embeddings = model.encode([chunk["text"] for chunk in data], convert_to_tensor=True)

final_embeddings = list()
for chunk,embedding in zip(data , embeddings):
  final_embeddings.append({
    "chunk" : chunk,
    "embedding" : embedding
  })
print("Chunk Size:", len(data))
print("Embeddings generated for all chunks.")
print(f"Total number of embeddings: {len(final_embeddings)}")

expected_dim = final_embeddings[0]["embedding"].shape[0]
print("Embedding dimension:", expected_dim)

assert len(data) == len(final_embeddings)
assert all(item["embedding"].shape[0] == expected_dim for item in final_embeddings), \
    "Dimension mismatch across embeddings"

print("All checks passed.")

