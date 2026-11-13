"""Step 3 - Embedding: model used for both chunks and user queries."""

from langchain_openai import OpenAIEmbeddings


def get_embedding_model(model_name: str = "text-embedding-3-small"):
    return OpenAIEmbeddings(model=model_name)