from dataclasses import dataclass 

from src.context_builder import BuildingContext
from src.generator import GenerationResult

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

  def run(self,question, top_k =2):
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
  
if __name__ == "__main__":
    from src.retriever import Retrieval
    from src.context_builder import ContextBuilder
    from src.generator import Generator

    # Load the heavy objects once and reuse them for every question.
    pipeline = RAGPipeline(
        retriever=Retrieval(),
        context_builder=ContextBuilder(max_context_tokens=6000),
        generator=Generator(model="openai/gpt-oss-120b"),
    )

    # A mix of in-scope, partial, and out-of-scope questions.
    test_questions = [
        "What is Machine Learning?",
        "How are machine learning and deep learning related?",
        "What is the capital of France?",  # should refuse
    ]

    def show(result: RAGResult):
        print("=" * 70)
        print("QUESTION   :", result.question)
        print("-" * 70)
        print("RETRIEVED  :")
        for r in result.retrieved_chunks:
            print(f"  {r['chunk_id']}  score={r['score']:.4f}  heading={r['heading']!r}")
        print("IN CONTEXT :", result.built_context.included_chunk_ids)
        print("-" * 70)
        g = result.generation
        print("ANSWER     :", g.answer)
        print("CITED      :", g.cited_chunk_ids)
        print("INVALID    :", g.invalid_citations)
        print("REFUSED    :", g.refused)

    # 1. Run the fixed test questions.
    for q in test_questions:
        try:
            show(pipeline.run(q, top_k=2))
        except Exception as e:
            print(f"[ERROR] {q!r}: {type(e).__name__}: {e}")

    # 2. Interactive mode for your own questions.
    print("\nInteractive mode. Type a question, or 'exit' to quit.")
    while True:
        q = input("\nQuestion> ").strip()
        if q.lower() in {"exit", "quit", ""}:
            break
        try:
            show(pipeline.run(q, top_k=2))
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")