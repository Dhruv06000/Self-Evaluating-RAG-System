"""Entry point: index a YouTube video (once), then chat with it."""

import os
import logging
from dotenv import load_dotenv

from src.ingestion import ingest_transcript
from src.chunking import chunk_text
from src.embedding import get_embedding_model
from src.faiss_index import get_or_create_vector_store
from src.reranker import get_reranking_retriever
from src.context_builder import get_prompt_template
from src.generator import get_llm
from src.pipeline import run_rag
from src.summarizer import needs_summary, build_summary_chain

load_dotenv()  # loads OPENAI_API_KEY from .env

VIDEO_ID = "foGklduxhM0"  # only the ID, not the full URL
INDEX_PATH = f"data/{VIDEO_ID}"  # one saved index per video
TRANSCRIPT_PATH = f"data/{VIDEO_ID}_transcript.txt"  # cached full transcript


def get_transcript():
    """Fetch the transcript once and cache it, so summaries don't re-hit YouTube."""
    if os.path.exists(TRANSCRIPT_PATH):
        with open(TRANSCRIPT_PATH, encoding="utf-8") as f:
            return f.read()
    text = ingest_transcript(VIDEO_ID)
    os.makedirs("data", exist_ok=True)
    with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def load_chunks():
    return chunk_text(get_transcript())


def main():
    embeddings = get_embedding_model()

    # Indexing: loads from data/ if it exists, otherwise builds and saves it
    vector_store = get_or_create_vector_store(embeddings, load_chunks, INDEX_PATH)

    # Build the pieces once
    retriever = get_reranking_retriever(vector_store, k=4, fetch_k=20)
    logging.getLogger().setLevel(logging.WARNING)   # hide library INFO logs after the reranker loads
    prompt = get_prompt_template()
    llm = get_llm()
    summary_chain = build_summary_chain(llm)      # whole-video questions: no retrieval

    # Chat loop
    print("\nAsk questions about the video (type 'exit' to quit).")
    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        if needs_summary(question):
            answer = summary_chain.invoke(
                {"transcript": get_transcript(), "question": question}
            )
        else:
            # normal RAG: retrieve -> build context -> generate (same path the eval uses)
            answer = run_rag(retriever, prompt, llm, question)["answer"]
        print(f"\nBot: {answer}")


if __name__ == "__main__":
    main()