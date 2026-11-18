"""Whole-video questions (summaries) can't be answered from top-k chunks, so they skip retrieval."""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

SUMMARY_KEYWORDS = (
    "summar", "overview", "main points", "key points", "key takeaways",
    "takeaway", "tl;dr", "whole video", "entire video", "what is this video about",
    "what is the video about",
)

SUMMARY_TEMPLATE = """You are summarizing a YouTube video using its full transcript.
The transcript is auto-generated speech, so it contains typos and misheard words. Fix them silently.
Use ONLY the transcript. Follow the request exactly, including any requested number of points.
If the request is about a topic the video does not cover, say "The video doesn't cover that." instead of summarizing.

Transcript:
{transcript}

Request: {question}

Answer:"""


def needs_summary(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in SUMMARY_KEYWORDS)


def build_summary_chain(llm):
    return PromptTemplate.from_template(SUMMARY_TEMPLATE) | llm | StrOutputParser()