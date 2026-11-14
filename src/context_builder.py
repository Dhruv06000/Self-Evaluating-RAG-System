"""Step 6 - Augmentation: build the context and the final prompt."""

from langchain_core.prompts import PromptTemplate


def format_docs(retrieved_docs) -> str:
    """Join retrieved chunks into one context string."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def get_prompt_template() -> PromptTemplate:
    return PromptTemplate(
        template="""You are an assistant that answers questions about a YouTube video using its transcript.

Rules:
1. Use ONLY the transcript excerpts below. Do not add outside knowledge, even if you know the topic well.
2. Answer the question directly in the first sentence, then add only the supporting detail the excerpts give. Keep it concise.
3. The excerpts are auto-generated speech, so they contain filler words, repetition and misspelled terms (for example "MM LU" for "MMLU"). Clean this up and write in clear, natural language, without quoting the speaker word for word.
4. If the excerpts answer only part of the question, answer that part and say plainly what the video does not cover.
5. If the excerpts do not contain the answer, reply exactly: "I don't know based on this video."
6. Reply in the same language the user asked in.

Transcript excerpts:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"],
    )


def build_final_prompt(prompt, retriever, question: str):
    """Retrieve -> build context -> fill the prompt (manual, step-by-step)."""
    retrieved_docs = retriever.invoke(question)
    context_text = format_docs(retrieved_docs)
    return prompt.invoke({"context": context_text, "question": question})

