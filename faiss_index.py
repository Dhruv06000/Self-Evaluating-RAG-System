import faiss

from embedding import generate_embeddings

data , embeddings = generate_embeddings()

faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
faiss_index.add(embeddings)

print("FAISS index created and embeddings added.")

assert faiss_index.ntotal == len(embeddings)
assert faiss_index.d == embeddings.shape[1]

faiss.write_index(faiss_index, "data/faiss.index")
print("FAISS index saved to 'data/faiss.index'.")