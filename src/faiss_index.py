import sys  
import faiss

from src.embedding import generate_embeddings
from src.config import DEFAULT_CHUNK_SIZE, index_path  

def build_index(chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
    _, embeddings = generate_embeddings(chunk_size)  

    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)
    print("FAISS index created and embeddings added.")

    assert faiss_index.ntotal == len(embeddings)
    assert faiss_index.d == embeddings.shape[1]

    path = index_path(chunk_size)  
    faiss.write_index(faiss_index, path)
    print(f"FAISS index saved to '{path}'.")


if __name__ == "__main__":

    size = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CHUNK_SIZE
    build_index(size)