"""Step 4 - Vector store: build, save, load and reuse a FAISS index."""

import os

from langchain_community.vectorstores import FAISS


def build_vector_store(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)


def save_vector_store(vector_store, path: str = "data/faiss_index"):
    vector_store.save_local(path)


def load_vector_store(embeddings, path: str = "data/faiss_index"):
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def get_or_create_vector_store(embeddings, load_chunks, path: str = "data/faiss_index"):
    """Load the saved index if it exists, otherwise build it from chunks and save it."""
    if os.path.exists(os.path.join(path, "index.faiss")):
        print(f"Loading saved index from {path}")
        return load_vector_store(embeddings, path)

    print(f"No index found, building it at {path}")
    chunks = load_chunks()  # only runs when the index is missing
    vector_store = build_vector_store(chunks, embeddings)
    save_vector_store(vector_store, path)
    return vector_store
