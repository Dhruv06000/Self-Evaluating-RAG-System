from types import SimpleNamespace

import pytest

from context_builder import BuildingContext
from generator import (
    REFUSAL_SENTINEL,
    SYSTEM_PROMPT,
    Generator,
    GenerationResult,
    build_user_message,
)

CONTEXT_TEXT = "[CHUNK]\n\nChunk ID: chunk_000001\n\nContent:\nSome fact.\n\n[END CHUNK]"


@pytest.fixture
def generator(monkeypatch):
    # A dummy key lets the client be constructed; no request is ever sent.
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    return Generator(model="test-model")


def make_context(ids=("chunk_000001",), context=CONTEXT_TEXT):
    return BuildingContext(context=context, included_chunk_ids=tuple(ids))


def fake_llm(generator, reply):
    """Replace the network call with a fake that records messages and returns `reply`."""
    received = []

    def _fake(user_message):
        received.append(user_message)
        return reply

    generator._call_llm = _fake
    return received


# ---------- build_user_message ----------

def test_build_user_message_has_both_sections_and_question_last():
    message = build_user_message("CTX", "What is X?")

    assert "<RETRIEVED_CONTEXT>\nCTX\n</RETRIEVED_CONTEXT>" in message
    assert "<USER_QUESTION>\nWhat is X?\n</USER_QUESTION>" in message
    assert message.index("</RETRIEVED_CONTEXT>") < message.index("<USER_QUESTION>")


def test_system_prompt_contains_sentinel():
    assert REFUSAL_SENTINEL in SYSTEM_PROMPT


# ---------- input validation and shortcuts ----------

@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_question_raises(generator, question):
    with pytest.raises(ValueError):
        generator.generate(question, make_context())


def test_empty_context_refuses_without_calling_model(generator):
    received = fake_llm(generator, "should never be used")

    result = generator.generate("What is X?", make_context(ids=(), context=""))

    assert result.answer == REFUSAL_SENTINEL
    assert result.refused is True
    assert result.cited_chunk_ids == ()
    assert received == []


def test_question_is_stripped_before_building_message(generator):
    received = fake_llm(generator, "Fact. [chunk_000001]")

    generator.generate("   What is X?   ", make_context())

    assert "<USER_QUESTION>\nWhat is X?\n</USER_QUESTION>" in received[0]


def test_context_is_passed_to_model(generator):
    received = fake_llm(generator, "Fact. [chunk_000001]")

    generator.generate("What is X?", make_context())

    assert CONTEXT_TEXT in received[0]


# ---------- citation extraction and validation ----------

def test_valid_citation_is_recorded(generator):
    fake_llm(generator, "Fact. [chunk_000001]")

    result = generator.generate("What is X?", make_context())

    assert result.cited_chunk_ids == ("chunk_000001",)
    assert result.invalid_citations == ()
    assert result.refused is False


def test_invented_citation_is_flagged_as_invalid(generator):
    fake_llm(generator, "Fact. [chunk_000999]")

    result = generator.generate("What is X?", make_context())

    assert result.cited_chunk_ids == ()
    assert result.invalid_citations == ("chunk_000999",)


def test_citation_of_chunk_skipped_by_budget_is_invalid(generator):
    # chunk_000002 existed in retrieval but did not fit the context budget.
    fake_llm(generator, "Fact. [chunk_000001] Other. [chunk_000002]")

    result = generator.generate("What is X?", make_context(ids=("chunk_000001",)))

    assert result.cited_chunk_ids == ("chunk_000001",)
    assert result.invalid_citations == ("chunk_000002",)


def test_duplicate_citations_are_listed_once_in_first_appearance_order(generator):
    fake_llm(
        generator,
        "A. [chunk_000002] B. [chunk_000001] C. [chunk_000002] D. [chunk_000001]",
    )

    result = generator.generate(
        "What is X?", make_context(ids=("chunk_000001", "chunk_000002"))
    )

    assert result.cited_chunk_ids == ("chunk_000002", "chunk_000001")


def test_multiple_adjacent_citations_are_all_extracted(generator):
    fake_llm(generator, "Fact. [chunk_000001] [chunk_000002]")

    result = generator.generate(
        "What is X?", make_context(ids=("chunk_000001", "chunk_000002"))
    )

    assert result.cited_chunk_ids == ("chunk_000001", "chunk_000002")


def test_answer_without_citations_has_no_cited_ids(generator):
    fake_llm(generator, "An answer with no citations.")

    result = generator.generate("What is X?", make_context())

    assert result.cited_chunk_ids == ()
    assert result.invalid_citations == ()


def test_comma_separated_ids_in_one_bracket_are_not_extracted(generator):
    # Documents the deliberately strict format: one ID per bracket.
    fake_llm(generator, "Fact. [chunk_000001, chunk_000002]")

    result = generator.generate(
        "What is X?", make_context(ids=("chunk_000001", "chunk_000002"))
    )

    assert result.cited_chunk_ids == ()
    assert result.invalid_citations == ()


# ---------- refusal detection ----------

def test_exact_sentinel_is_a_refusal(generator):
    fake_llm(generator, REFUSAL_SENTINEL)

    result = generator.generate("What is X?", make_context())

    assert result.refused is True
    assert result.cited_chunk_ids == ()


def test_sentinel_with_surrounding_whitespace_is_a_refusal(generator):
    fake_llm(generator, f"  {REFUSAL_SENTINEL}\n")

    assert generator.generate("What is X?", make_context()).refused is True


def test_partial_answer_mentioning_sentinel_is_not_a_refusal(generator):
    fake_llm(
        generator,
        f"Fact. [chunk_000001] Missing information: details. {REFUSAL_SENTINEL}",
    )

    result = generator.generate("What is X?", make_context())

    assert result.refused is False
    assert result.cited_chunk_ids == ("chunk_000001",)


def test_partial_answer_is_not_a_refusal(generator):
    fake_llm(generator, "Fact. [chunk_000001] Missing information: the date.")

    assert generator.generate("What is X?", make_context()).refused is False


# ---------- _call_llm (client replaced by a fake) ----------

def fake_client(text):
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=text)

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return client, calls


@pytest.mark.parametrize("bad_text", [None, "", "   \n"])
def test_empty_model_response_raises(generator, bad_text):
    generator.client, _ = fake_client(bad_text)

    with pytest.raises(RuntimeError):
        generator._call_llm("message")


def test_call_llm_returns_stripped_text(generator):
    generator.client, _ = fake_client("  hello  \n")

    assert generator._call_llm("message") == "hello"


def test_call_llm_sends_system_prompt_model_and_temperature(generator):
    generator.client, calls = fake_client("ok")

    generator._call_llm("the user message")

    call = calls[0]
    assert call["model"] == "test-model"
    assert call["contents"] == "the user message"
    assert call["config"].system_instruction == SYSTEM_PROMPT
    assert call["config"].temperature == 0.0


def test_generation_result_is_immutable():
    result = GenerationResult(answer="x")

    with pytest.raises(Exception):
        result.answer = "y"