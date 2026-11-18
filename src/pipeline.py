"""The one RAG path: retrieve -> build context -> generate.
Both main.py and the pipeline eval call this, so they can never drift apart."""

from src.context_builder import format_docs
from src.generator import generate_answer


def run_rag(retriever, prompt, llm, question: str) -> dict:
    """Returns the answer AND the retrieved chunks (the eval and source citations need both)."""
    docs = retriever.invoke(question)
    final_prompt = prompt.invoke({"context": format_docs(docs), "question": question})
    return {"answer": generate_answer(llm, final_prompt), "docs": docs}