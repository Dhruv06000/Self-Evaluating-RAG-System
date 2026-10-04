#Test_context_builder.py
import pytest
from context_builder import ContextBuilder


@pytest.fixture
def builder():
    return ContextBuilder(1000)


def make_chunk(
    chunk_id,
    text,
    heading="Test Heading",
    source="test.pdf",
    pages=None,
):
    if pages is None:
        pages = [{"page_label": "1"}]

    return {
        "chunk_id": chunk_id,
        "text": text,
        "heading": heading,
        "metadata": {
            "source": source,
            "pages": pages,
        },
    }


def test_invalid_max_context_tokens():
    with pytest.raises(ValueError):
        ContextBuilder(0)

    with pytest.raises(ValueError):
        ContextBuilder(-1)


def test_format_chunk(builder):
    chunk = make_chunk(
        chunk_id="chunk_000001",
        text="Machine learning is a subfield of artificial intelligence.",
        heading="Machine Learning",
        source="book.pdf",
        pages=[
            {"page_label": "42"},
            {"page_label": "43"},
        ],
    )

    formatted = builder._format_chunk(chunk)

    assert "[CHUNK]" in formatted
    assert "[END CHUNK]" in formatted
    assert "Chunk ID: chunk_000001" in formatted
    assert "Source: book.pdf" in formatted
    assert "Pages: 42, 43" in formatted
    assert "Heading: Machine Learning" in formatted
    assert "Machine learning is a subfield of artificial intelligence." in formatted


def test_count_tokens(builder):
    text = "Machine learning is useful."

    token_count = builder._count_tokens(text)

    assert isinstance(token_count, int)
    assert token_count > 0


def test_empty_retrieved_chunks(builder):
    result = builder.build([])

    assert result.context == ""
    assert result.included_chunk_ids == () 
    


def test_all_chunks_fit():
    chunks = [
        make_chunk("chunk_A", "Short text A."),
        make_chunk("chunk_B", "Short text B."),
        make_chunk("chunk_C", "Short text C."),
    ]

    builder = ContextBuilder(1000)

    result = builder.build(chunks)

    assert "chunk_A" in result.context
    assert "chunk_B" in result.context
    assert "chunk_C" in result.context


def test_context_does_not_exceed_token_budget():
    chunks = [
        make_chunk(
            "chunk_A",
            "Machine learning is a field of artificial intelligence."
            * 10,
        ),
        make_chunk(
            "chunk_B",
            "Supervised learning uses labelled training data."
            * 10,
        ),
        make_chunk(
            "chunk_C",
            "Unsupervised learning finds patterns in data."
            * 10,
        ),
    ]

    builder = ContextBuilder(500)

    result = builder.build(chunks)

    assert builder._count_tokens(result.context) <= 500


def test_retrieval_order_is_preserved():
    chunks = [
        make_chunk("chunk_A", "First chunk."),
        make_chunk("chunk_B", "Second chunk."),
        make_chunk("chunk_C", "Third chunk."),
    ]

    builder = ContextBuilder(1000)

    result = builder.build(chunks)

    position_A = result.context.index("chunk_A")
    position_B = result.context.index("chunk_B")
    position_C = result.context.index("chunk_C")

    assert position_A < position_B < position_C


def test_chunk_that_does_not_fit_is_skipped_and_later_chunk_is_checked():
    chunks = [
        make_chunk("chunk_A", "A"),
        make_chunk("chunk_B", "B"),
        make_chunk("chunk_C", "C " * 50),  # Significantly larger chunk
        make_chunk("chunk_D", "D"),
    ]

    builder = ContextBuilder(1000)

    formatted_A = builder._format_chunk(chunks[0])
    formatted_B = builder._format_chunk(chunks[1])
    formatted_D = builder._format_chunk(chunks[3])

    # Account for joining separators in the budget calculation
    context_A_B_D = "\n\n".join([formatted_A, formatted_B, formatted_D])
    budget = builder._count_tokens(context_A_B_D)

    builder = ContextBuilder(budget)

    result = builder.build(chunks)

    assert "chunk_A" in result.context
    assert "chunk_B" in result.context
    assert "chunk_C" not in result.context
    assert "chunk_D" in result.context

    assert builder._count_tokens(result.context) <= budget


def test_context_builder_does_not_modify_retrieval_order():
    chunks = [
        make_chunk("chunk_003", "Third."),
        make_chunk("chunk_001", "First."),
        make_chunk("chunk_002", "Second."),
    ]

    builder = ContextBuilder(1000)

    result = builder.build(chunks)

    assert result.context.index("chunk_003") < result.context.index("chunk_001")
    assert result.context.index("chunk_001") < result.context.index("chunk_002")


def test_multi_page_metadata_is_formatted():
    chunk = make_chunk(
        chunk_id="chunk_000123",
        text="Some content.",
        pages=[
            {"page_label": "42"},
            {"page_label": "43"},
        ],
    )

    builder = ContextBuilder(1000)

    formatted = builder._format_chunk(chunk)

    assert "Pages: 42, 43" in formatted



def make_skipped_chunk_scenario():
    """A, B, C where B is too large to fit. The budget fits exactly A + C."""
    chunks = [
        make_chunk("chunk_A", "A"),
        make_chunk("chunk_B", "large chunk of text " * 500),
        make_chunk("chunk_C", "C"),
    ]

    probe = ContextBuilder(1000)
    formatted_A = probe._format_chunk(chunks[0])
    formatted_B = probe._format_chunk(chunks[1])
    formatted_C = probe._format_chunk(chunks[2])

    budget = probe._count_tokens("\n\n".join([formatted_A, formatted_C]))

    # Make the test's premise explicit: B really cannot fit.
    assert probe._count_tokens(formatted_B) > budget

    return ContextBuilder(budget), chunks


def test_skipped_chunk_id_is_not_in_included_chunk_ids():
    builder, chunks = make_skipped_chunk_scenario()

    result = builder.build(chunks)

    assert result.included_chunk_ids == ("chunk_A", "chunk_C")
    assert "chunk_B" not in result.included_chunk_ids


def test_included_chunk_ids_preserve_retrieval_order():
    chunks = [
        make_chunk("chunk_003", "Third."),
        make_chunk("chunk_001", "First."),
        make_chunk("chunk_002", "Second."),
    ]

    result = ContextBuilder(1000).build(chunks)

    assert result.included_chunk_ids == ("chunk_003", "chunk_001", "chunk_002")


def test_included_chunk_ids_match_context_content():
    builder, chunks = make_skipped_chunk_scenario()

    result = builder.build(chunks)

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        if chunk_id in result.included_chunk_ids:
            assert f"Chunk ID: {chunk_id}" in result.context
        else:
            assert f"Chunk ID: {chunk_id}" not in result.context


def test_first_chunk_larger_than_budget_gives_empty_result():
    chunks = [make_chunk("chunk_A", "some text that cannot fit")]

    result = ContextBuilder(5).build(chunks)

    assert result.context == ""
    assert result.included_chunk_ids == ()