# Project State

## Project

Self-Evaluating RAG System

## Current Milestone

### Feature 4 — Embeddings + FAISS

Status: **Completed**

### Completed Work

#### Feature 1 — Knowledge Base Setup

- Created the `knowledge_base/` directory.
- Selected the AI/technology PDF as the initial knowledge source.
- Added `setup_knowledge_base.py`.
- Implemented a check to determine whether the PDF already exists.
- Implemented PDF downloading when the file is missing.
- Added a request timeout for the download request.
- Tested the missing-PDF case successfully.
- Tested the existing-PDF case successfully.

#### Feature 2 — PDF Document Ingestion

- Learned how to open PDFs using PyMuPDF.
- Created `document_ingestion.py` for reusable ingestion logic.
- Implemented page-by-page PDF text extraction using `page.get_text()`.
- Added basic text formatting by replacing newlines with spaces and stripping surrounding whitespace.
- Implemented handling for pages with no extractable text by skipping them.
- Represented each extracted page as a page-level document.
- Added `source` metadata.
- Added `pdf_page` metadata representing the physical PDF page number.
- Added `page_label` metadata using the PDF's own page label through PyMuPDF.
- Verified that the PDF contains 308 pages.
- Verified that all 308 pages produced extracted page-level documents.
- Verified page labels:
  - PDF page 1 → `C1`
  - PDF page 13 → `xiii`
  - PDF page 14 → `1`
  - PDF page 308 → `298`
- Verified the extracted document structure and metadata.

#### Feature 3 — Document Chunking

- Created `document_chunking.py` for the document chunking pipeline.
- Implemented heading detection for chapter and section headings.
- Grouped paragraphs under their corresponding headings.
- Used a hierarchical chunking strategy:

  **Heading → Paragraph → Sentence → Word**

- Used `BAAI/bge-base-en-v1.5` tokenizer for actual token counting.
- Set the maximum chunk size to **350 tokenizer tokens**.
- Chunks do not use overlap in the current version.
- Content from different headings is never intentionally mixed.
- PDF page boundaries do not force a new chunk when the same heading continues across pages.
- Paragraphs are the primary unit for creating chunks.
- If a paragraph is larger than 350 tokens, it is split into sentences.
- If an individual sentence is larger than 350 tokens, word-level splitting is used as a fallback.
- The word-level fallback was retained for robustness even though the current PDF had no sentences larger than 350 tokens.
- Preserved the heading associated with every final chunk.
- Preserved the source and page-label information in final chunk metadata.
- Refactored the chunking pipeline so PDF processing and chunk generation occur through a reusable function.
- Added `if __name__ == "__main__":` to prevent chunk generation from running automatically when the module is imported.
- Added JSON persistence for the final chunks.
- Saved the 467 final chunks to `data/chunks.json`.
- The embedding pipeline will consume the saved chunks instead of re-running the chunking pipeline.
- Final chunk representation:

```python
{
    "text": "...",
    "heading": "...",
    "token_count": 280,
    "metadata": {
        "source": "knowledge_base/artificial_intelligence_technology.pdf",
        "pages": [
            {"page_label": "1"},
            {"page_label": "2"}
        ]
    }
}

### Chunking Validation

The final chunking pipeline was validated using the complete PDF.

Validation results:

- Total final chunks: **467**
- Minimum token count: **7**
- Maximum token count: **350**
- Average token count: **253.57**
- Chunks over 350 tokens: **0**
- Chunks with empty text: **0**
- Chunks without heading: **0**
- Chunks with missing metadata: **0**

Therefore, all final chunks satisfy the configured maximum token limit of 350 tokens.

### Additional Chunking Validation

The following checks were also performed:

- Heading detection was manually tested against real chapter and section headings.
- False heading detection for section number `3.0` was fixed by requiring the first section number after the decimal to begin from `1-9`.
- Heading boundaries were inspected across the generated chunks.
- Multi-page content under the same heading was verified.
- **9 paragraphs** were found to be larger than 350 tokens.
- **0 sentences** were larger than 350 tokens.
- Word-level fallback therefore was not required for the current PDF, but remains implemented as a robustness mechanism.

### PDF Artifacts

Some PDF extraction artifacts were observed, including table-of-contents text and running headers/footers.

These artifacts were reviewed and intentionally left unchanged because they do not currently prevent the chunking pipeline from working correctly.

No additional PDF preprocessing is planned at this stage.

#### Feature 4 — Embeddings + FAISS

- Created `embedding.py` for the embedding generation pipeline.
- Used `BAAI/bge-base-en-v1.5` as the embedding model.
- Loaded the persisted chunks from `data/chunks.json` instead of re-running the chunking pipeline.
- Generated one embedding for each of the 467 final chunks.
- Verified that the number of embeddings matches the number of chunks.
- Verified that all embeddings have the same dimension.
- Embedding dimension: **768**.
- Used normalized embeddings so that inner product can be used as cosine similarity.
- Used `convert_to_numpy=True` so the embeddings are directly compatible with FAISS.
- Created a FAISS `IndexFlatIP` index using the embedding dimension.
- Added all 467 normalized embeddings to the FAISS index.
- Verified that the number of vectors stored in FAISS matches the number of embeddings.
- Saved the FAISS index to `data/faiss.index`.
- Implemented and tested query embedding generation using the same embedding model and normalization configuration.
- Tested semantic search using the query: `What is Artificial Intelligence?`
- Retrieved the top 5 results using FAISS.
- Verified that FAISS returns both similarity scores and vector indices.
- Verified that returned FAISS indices correctly map back to the corresponding chunks in `data/chunks.json`.
- Verified that retrieved chunks contain relevant AI-related content and preserved metadata.

##### Embedding and FAISS Architecture

The indexing pipeline is:

**`chunks.json → embeddings → normalization → FAISS index → faiss.index`**

The query pipeline is:

**`question → query embedding → normalization → FAISS search → indices + scores → chunks.json → retrieved chunks`**

`data/chunks.json` remains the source of truth for chunk text and metadata, while `data/faiss.index` stores the numerical vectors used for semantic search.

The current FAISS index uses `IndexFlatIP` because the project currently contains only 467 chunks. Exact brute-force search is therefore simple and appropriate at this stage.

##### Embedding and FAISS Validation

- Total chunks: **467**
- Total embeddings: **467**
- Embedding dimension: **768**
- FAISS vectors: **467**
- FAISS index dimension: **768**
- Query search: **Successful**
- Top-k retrieval: **Successful**
- Chunk-to-index mapping: **Verified**

## Current Project Structure

```text
self_evaluating_rag/
├── knowledge_base/
│   └── artificial_intelligence_technology.pdf
├── data/
│   ├── chunks.json
│   └── faiss.index
├── document_ingestion.py
├── document_chunking.py
├── setup_knowledge_base.py
├── embedding.py
├── faiss_index.py
├── main.ipynb
├── LEARNING_CONTEXT.md
├── PROJECT_STATE.md
└── .gitignore

## Chunking Strategy

The current chunking strategy is hierarchical:

**Heading → Paragraph → Sentence → Word**

The strategy follows this priority:

1. Keep content under the same heading together.
2. Use paragraphs as the primary chunking unit.
3. If a paragraph exceeds 350 tokens, split it into sentences.
4. If a sentence exceeds 350 tokens, split it into words.
5. Keep every final chunk at or below 350 tokenizer tokens.
6. No overlap is used in the current version.

The current version intentionally uses no overlap so that retrieval quality can be evaluated before introducing additional complexity.

## Status

- Feature 1 — Knowledge Base Setup: **Completed**
- Feature 2 — PDF Document Ingestion: **Completed**
- Feature 3 — Document Chunking: **Completed**
- Feature 4 — Embeddings + FAISS: **Completed**

## Next Milestone

### Feature 5 — Semantic Retrieval