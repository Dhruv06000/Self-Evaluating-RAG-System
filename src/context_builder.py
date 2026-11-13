"""Step 6 - Augmentation: build the context and the final prompt."""

from langchain_core.prompts import PromptTemplate


def format_docs(retrieved_docs) -> str:
    """Join retrieved chunks into one context string."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def get_prompt_template() -> PromptTemplate:
    return PromptTemplate(
        template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
        input_variables=["context", "question"],
    )


def build_final_prompt(prompt, retriever, question: str):
    """Retrieve -> build context -> fill the prompt (manual, step-by-step)."""
    retrieved_docs = retriever.invoke(question)
    context_text = format_docs(retrieved_docs)
    return prompt.invoke({"context": context_text, "question": question})

