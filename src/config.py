# src/config.py
"""Shared settings. Values and path rules only: no logic, no heavy imports."""

DEFAULT_CHUNK_SIZE = 1024
DATA_DIR = "data"


def chunks_path(chunk_size: int) -> str:
    """Chunk file for a given chunk size, e.g. data/1024_chunks.json."""
    return f"{DATA_DIR}/{chunk_size}_chunks.json"


def index_path(chunk_size: int) -> str:
    """FAISS index for a given chunk size, e.g. data/faiss_1024.index."""
    return f"{DATA_DIR}/faiss_{chunk_size}.index"