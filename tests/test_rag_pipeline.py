from context_builder import BuildingContext
from generator import GenerationResult
from rag_pipeline import RAGPipeline, RAGResult
import pytest


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []                      # records (query, top_k)

    def retrieve(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.chunks


class FakeBuilder:
    def __init__(self, built):
        self.built = built
        self.received = None                 # records what it was given

    def build(self, retrieved_chunks):
        self.received = retrieved_chunks
        return self.built


class FakeGenerator:
    def __init__(self, result):
        self.result = result
        self.received = None

    def generate(self, question, built_context):
        self.received = (question, built_context)
        return self.result

class FailingRetriever:
    def retrieve(self, query, top_k=5):
        raise RuntimeError("retrieval failed")
    
class FailingGenerator:
    def generate(self, question, built_context):
        raise RuntimeError("generation failed")


def make_pipeline():
    chunks = [{"chunk_id": "chunk_000001"}, {"chunk_id": "chunk_000002"}]
    built = BuildingContext(context="ctx", included_chunk_ids=("chunk_000001",))
    generation = GenerationResult(answer="ans [chunk_000001]")
    retriever, builder, generator = FakeRetriever(chunks), FakeBuilder(built), FakeGenerator(generation)
    pipeline = RAGPipeline(retriever, builder, generator)
    return pipeline, retriever, builder, generator, chunks, built, generation


def test_result_contains_each_stage_output():
    pipeline, _, _, _, chunks, built, generation = make_pipeline()

    result = pipeline.run("What is AI?")

    assert isinstance(result, RAGResult)
    assert result.question == "What is AI?"
    assert result.retrieved_chunks == tuple(chunks)
    assert result.built_context is built
    assert result.generation is generation

def test_stages_receive_the_right_inputs():
    pipeline, retriever, builder, generator, chunks, built, generation = make_pipeline()

    result = pipeline.run("What is AI?",top_k = 3)

    # Check that each stage received the right inputs
    assert retriever.calls == [("What is AI?", 3)]
    assert builder.received == chunks
    assert generator.received == ("What is AI?", built)


def test_retrieval_error_propagates_and_stops_later_stages():
    builder = FakeBuilder(BuildingContext(context="", included_chunk_ids=()))
    generator = FakeGenerator(GenerationResult(answer="x"))
    pipeline = RAGPipeline(FailingRetriever(), builder, generator)

    with pytest.raises(RuntimeError):
        pipeline.run("What is AI?")

    assert builder.received is None
    assert generator.received is None

def test_generation_error_propagates():
    chunks = [{"chunk_id": "chunk_000001"}]
    built = BuildingContext(context="ctx", included_chunk_ids=("chunk_000001",))
    retriever = FakeRetriever(chunks)
    builder = FakeBuilder(built)
    pipeline = RAGPipeline(retriever, builder, FailingGenerator())

    with pytest.raises(RuntimeError):
        pipeline.run("What is AI?")

    assert retriever.calls == [("What is AI?", 5)]
    assert builder.received == chunks