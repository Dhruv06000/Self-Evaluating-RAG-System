"""Entry point: index a YouTube video (once), then chat with it, with cited sources."""

import json
import os

from dotenv import load_dotenv

from src.ingestion import ingest_transcript_timed
from src.chunking import chunk_transcript
from src.embedding import get_embedding_model
from src.faiss_index import get_or_create_vector_store
from src.reranker import get_reranking_retriever
from src.context_builder import get_prompt_template
from src.generator import get_llm
from src.pipeline import run_rag, format_sources
from src.summarizer import needs_summary, build_summary_chain

load_dotenv()  # loads OPENAI_API_KEY from .env

VIDEO_ID = "foGklduxhM0"  # only the ID, not the full URL
INDEX_PATH = f"data/{VIDEO_ID}"  # one saved index per video
TRANSCRIPT_PATH = f"data/{VIDEO_ID}_transcript.json"  # cached text + caption timings


def get_transcript_data():
    """Fetch once, cache on disk. Returns (text, cues)."""
    if os.path.exists(TRANSCRIPT_PATH):
        with open(TRANSCRIPT_PATH, encoding="utf-8") as f:
            d = json.load(f)
        return d["text"], d["cues"]
    text, cues = ingest_transcript_timed(VIDEO_ID)
    os.makedirs("data", exist_ok=True)
    with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        json.dump({"text": text, "cues": cues}, f, ensure_ascii=False)
    return text, cues


def load_chunks():
    text, cues = get_transcript_data()
    return chunk_transcript(text, cues, VIDEO_ID)


def is_refusal(answer: str) -> bool:
    return answer.replace("’", "'").lower().startswith("i don't know")


def main():
    embeddings = get_embedding_model()
    vector_store = get_or_create_vector_store(embeddings, load_chunks, INDEX_PATH)

    retriever = get_reranking_retriever(vector_store, k=4, fetch_k=20)
    prompt = get_prompt_template()
    llm = get_llm()
    summary_chain = build_summary_chain(llm)

    print("\nAsk questions about the video (type 'exit' to quit).")
    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        if needs_summary(question):
            text, _ = get_transcript_data()
            answer = summary_chain.invoke({"transcript": text, "question": question})
        else:
            result = run_rag(retriever, prompt, llm, question)
            answer = result["answer"]
            if not is_refusal(answer):
                sources = format_sources(result["docs"][:3])   # show only the top 3
                if sources:
                    answer += "\n\nSources (approximate times):\n" + sources
        print(f"\nBot: {answer}")


if __name__ == "__main__":
    main()