from sentence_transformers import SentenceTransformer
import json


def generate_embeddings():

    model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    with open("data/chunks.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = model.encode(
        [chunk["text"] for chunk in data],
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