from transformers import AutoTokenizer


class ContextBuilder:
    def __init__(self, max_context_tokens: int):
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be greater than zero.")
        self.max_context_tokens = max_context_tokens
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

    def build(self, retrieved_chunks: list) -> str:
        """Build a context string from retrieved chunks without exceeding max_context_tokens."""
        selected_chunks = []
        separator = "\n\n"

        for chunk in retrieved_chunks:
            formatted_chunk = self._format_chunk(chunk)
            
            # Construct candidate context string including double newlines
            candidate_chunks = selected_chunks + [formatted_chunk]
            candidate_context = separator.join(candidate_chunks)

            # Check exact token size of the combined candidate string
            if self._count_tokens(candidate_context) <= self.max_context_tokens:
                selected_chunks.append(formatted_chunk)

        return separator.join(selected_chunks)

    def _format_chunk(self, chunk: dict) -> str:
        return f"""[CHUNK]\n
Chunk ID: {chunk["chunk_id"]}\n
Source: {chunk["metadata"]["source"]}\n
Pages: {', '.join(str(p["page_label"]) for p in chunk["metadata"]["pages"])}\n
Heading: {chunk["heading"]}\n

Content:\n{chunk["text"]}\n

[END CHUNK]"""

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tokenizer without adding special tokens like [CLS]/[SEP]."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))