"""Step 8 - Chain: retrieval + augmentation + generation in one pipeline."""

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from src.context_builder import format_docs


def build_chain(retriever, prompt, llm):
    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )
    return parallel_chain | prompt | llm | StrOutputParser()