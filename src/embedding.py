from sentence_transformers import SentenceTransformer
import json
from dotenv import load_dotenv
from src.config import DEFAULT_CHUNK_SIZE, chunks_path  
load_dotenv()

def generate_embeddings(chunk_size=DEFAULT_CHUNK_SIZE):

    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    with open(chunks_path(chunk_size), "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = model.encode(
        [chunk["text"] for chunk in data],
        batch_size=8,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    expected_dim = embeddings.shape[1]

    assert len(data) == len(embeddings), \
        "Number of chunks and embeddings do not match"

    assert all(embedding.shape[0] == expected_dim for embedding in embeddings), \
        "Dimension mismatch across embeddings"

    return data, embeddings


if __name__ == "__main__":
    data, embeddings = generate_embeddings()

    print("Chunk Size:", len(data))
    print("Total number of embeddings:", len(embeddings))
    print("Embedding dimension:", embeddings.shape[1])
    print("All checks passed.")