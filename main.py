"""Entry point: paste a YouTube link, index the video (once), then chat with it, with cited sources."""

from dotenv import load_dotenv

from src.ingestion import parse_video_id, get_transcript_data, VideoError
from src.chunking import chunk_transcript
from src.embedding import get_embedding_model
from src.faiss_index import get_or_create_vector_store
from src.reranker import get_reranking_retriever
from src.context_builder import get_prompt_template
from src.generator import get_llm
from src.pipeline import run_rag, format_sources
from src.summarizer import needs_summary, build_summary_chain

load_dotenv()  # loads OPENAI_API_KEY from .env


def is_refusal(answer: str) -> bool:
    return answer.replace("’", "'").lower().startswith("i don't know")


def ask_for_video():
    """Keep asking until we get a usable video. Returns (video_id, text, cues)."""
    while True:
        link = input("Paste a YouTube link (or 'exit'): ").strip()
        if link.lower() in ("exit", "quit", "q"):
            raise SystemExit
        try:
            video_id = parse_video_id(link)
            print("Fetching transcript...")
            text, cues = get_transcript_data(video_id)
            return video_id, text, cues
        except VideoError as e:
            print(f"  {e}\n")


def main():
    video_id, text, cues = ask_for_video()

    embeddings = get_embedding_model()
    vector_store = get_or_create_vector_store(
        embeddings,
        lambda: chunk_transcript(text, cues, video_id),  # only runs if no saved index
        f"data/{video_id}",                              # one saved index per video
    )

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