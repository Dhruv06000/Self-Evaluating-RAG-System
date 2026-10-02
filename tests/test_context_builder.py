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
    context = builder.build([])

    assert context == ""


def test_all_chunks_fit():
    chunks = [
        make_chunk("chunk_A", "Short text A."),
        make_chunk("chunk_B", "Short text B."),
        make_chunk("chunk_C", "Short text C."),
    ]

    builder = ContextBuilder(1000)

    context = builder.build(chunks)

    assert "chunk_A" in context
    assert "chunk_B" in context
    assert "chunk_C" in context


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

    context = builder.build(chunks)

    assert builder._count_tokens(context) <= 500


def test_retrieval_order_is_preserved():
    chunks = [
        make_chunk("chunk_A", "First chunk."),
        make_chunk("chunk_B", "Second chunk."),
        make_chunk("chunk_C", "Third chunk."),
    ]

    builder = ContextBuilder(1000)

    context = builder.build(chunks)

    position_A = context.index("chunk_A")
    position_B = context.index("chunk_B")
    position_C = context.index("chunk_C")

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

    context = builder.build(chunks)

    assert "chunk_A" in context
    assert "chunk_B" in context
    assert "chunk_C" not in context
    assert "chunk_D" in context

    assert builder._count_tokens(context) <= budget


def test_context_builder_does_not_modify_retrieval_order():
    chunks = [
        make_chunk("chunk_003", "Third."),
        make_chunk("chunk_001", "First."),
        make_chunk("chunk_002", "Second."),
    ]

    builder = ContextBuilder(1000)

    context = builder.build(chunks)

    assert context.index("chunk_003") < context.index("chunk_001")
    assert context.index("chunk_001") < context.index("chunk_002")


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