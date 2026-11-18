"""The one RAG path: retrieve -> build context -> generate.
Both main.py and the pipeline eval call this, so they can never drift apart."""

from src.context_builder import format_docs
from src.generator import generate_answer


def run_rag(retriever, prompt, llm, question: str) -> dict:
    """Returns the answer AND the retrieved chunks (the eval and source citations need both)."""
    docs = retriever.invoke(question)
    final_prompt = prompt.invoke({"context": format_docs(docs), "question": question})
    return {"answer": generate_answer(llm, final_prompt), "docs": docs}


def fmt_time(t: int) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def format_sources(docs) -> str:
    """One line per distinct timestamp, in relevance order."""
    seen, lines = set(), []
    for d in docs:
        t = d.metadata.get("start_time")
        if t is None or t in seen:
            continue
        seen.add(t)
        lines.append(f"  [{fmt_time(t)}] {d.metadata['url']}")
    return "\n".join(lines)