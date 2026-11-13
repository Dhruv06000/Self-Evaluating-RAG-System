"""Step 7 - Generation: the LLM answers from the augmented prompt."""

from langchain_openai import ChatOpenAI


def get_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.2):
    return ChatOpenAI(model=model_name, temperature=temperature)


def generate_answer(llm, final_prompt) -> str:
    return llm.invoke(final_prompt).content