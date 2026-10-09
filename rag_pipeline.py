from dataclasses import dataclass 

from context_builder import BuildingContext
from generator import GenerationResult

@dataclass(frozen= True)
class RAGResult:
  """ The Result of a RAG pipeline run, including the generated answer and the context used to generate it. """
  question: str
  retrieved_chunks : tuple
  built_context : BuildingContext
  generation : GenerationResult

class RAGPipeline:
  """ A simple RAG pipeline that retrieves context, builds it, and generates an answer. """
  def __init__(self,retriever, context_builder, generator):
    self.retriever = retriever
    self.context_builder = context_builder
    self.generator = generator

  def run(self,question, top_k =5):
    """ Run the RAG pipeline for a single question. Returns a RAGResult."""
    retrieved = self.retriever.retrieve(question, top_k)
    built = self.context_builder.build(retrieved)
    generation = self.generator.generate(question, built)
    return RAGResult(
      question = question,
      retrieved_chunks = tuple(retrieved),
      built_context = built,
      generation = generation
    )