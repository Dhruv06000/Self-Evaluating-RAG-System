"""Step 5 - Retrieval: fetch the top-k most relevant chunks."""

import os

from flashrank import Ranker, RerankRequest
from langchain_core.runnables import RunnableLambda


def get_retriever(vector_store, k: int = 4, search_type: str = "similarity"):
    return vector_store.as_retriever(
        search_type=search_type, search_kwargs={"k": k}
    )


def get_reranking_retriever(
    vector_store,
    k: int = 4,
    fetch_k: int = 20,
    model_name: str = "ms-marco-MiniLM-L-12-v2",
    cache_dir: str = "data/flashrank_cache",
):
    """Fetch fetch_k candidates by embedding similarity, re-score them against the
    question with a cross-encoder, keep the best k. Same .invoke(question) -> docs
    interface as get_retriever, so run_rag and the evals need no other change."""
    os.makedirs(cache_dir, exist_ok=True)
    base = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": fetch_k}
    )
    ranker = Ranker(model_name=model_name, cache_dir=cache_dir)

    def retrieve(question: str):
        docs = base.invoke(question)
        passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
        ranked = ranker.rerank(RerankRequest(query=question, passages=passages))
        return [docs[p["id"]] for p in ranked[:k]]

    return RunnableLambda(retrieve)