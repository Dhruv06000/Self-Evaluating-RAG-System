"""Entry point: index a YouTube video (once), then chat with it."""

from dotenv import load_dotenv


from src.ingestion import ingest_transcript
from src.chunking import chunk_text
from src.embedding import get_embedding_model
from src.faiss_index import get_or_create_vector_store
from src.retriever import get_retriever
from src.context_builder import get_prompt_template
from src.generation import get_llm
from src.chains import build_chain

load_dotenv()  # loads OPENAI_API_KEY from .env

VIDEO_ID = "LPZh9BOjkQs"  # only the ID, not the full URL
INDEX_PATH = f"data/{VIDEO_ID}"  # one saved index per video


def load_chunks():
    transcript = ingest_transcript(VIDEO_ID)
    return chunk_text(transcript)


def main():
    embeddings = get_embedding_model()

    # Indexing: loads from data/ if it exists, otherwise builds and saves it
    vector_store = get_or_create_vector_store(embeddings, load_chunks, INDEX_PATH)

    # Build the RAG chain once: retriever -> prompt -> llm -> parser
    retriever = get_retriever(vector_store, k=4)
    prompt = get_prompt_template()
    llm = get_llm()
    chain = build_chain(retriever, prompt, llm)

    # Chat loop
    print("\nAsk questions about the video (type 'exit' to quit).")
    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        answer = chain.invoke(question)
        print(f"\nBot: {answer}")


if __name__ == "__main__":
    main()