import faiss

from embedding import generate_embeddings


def build_index(index_path: str = "data/faiss.index") -> None:
    _, embeddings = generate_embeddings()

    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)
    print("FAISS index created and embeddings added.")

    assert faiss_index.ntotal == len(embeddings)
    assert faiss_index.d == embeddings.shape[1]

    faiss.write_index(faiss_index, index_path)
    print(f"FAISS index saved to '{index_path}'.")


if __name__ == "__main__":
    build_index()