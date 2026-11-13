"""Step 5 - Retrieval: fetch the top-k most relevant chunks."""


def get_retriever(vector_store, k: int = 4, search_type: str = "similarity"):
    return vector_store.as_retriever(
        search_type=search_type, search_kwargs={"k": k}
    )
