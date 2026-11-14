"""Step 7 - Generation: the LLM answers from the augmented prompt."""

from langchain_openai import ChatOpenAI


def get_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.2):
    kwargs = {"model": model_name}
    # reasoning models (gpt-5.x / gpt-6.x / o-series) only accept the default temperature
    if not model_name.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")):
        kwargs["temperature"] = temperature
    return ChatOpenAI(**kwargs)


def generate_answer(llm, final_prompt) -> str:
    return llm.invoke(final_prompt).content