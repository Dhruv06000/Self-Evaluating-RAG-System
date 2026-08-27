# Project State

## Project

Self-Evaluating RAG System

## Current Milestone

### Feature 2 — PDF Document Ingestion

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
- Created `document_ingestion.py` for the reusable ingestion logic.
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

## Current Knowledge Source

```text
knowledge_base/
└── artificial_intelligence_technology.pdf
```

## Current Project Structure

```text
self_evaluating_rag/
├── knowledge_base/
│   └── artificial_intelligence_technology.pdf
├── LEARNING_CONTEXT.md
├── PROJECT_STATE.md
├── main.ipynb
├── setup_knowledge_base.py
└── document_ingestion.py
```

## Document Representation

Each successfully extracted page is currently represented as:

```text
{
    "text": "...",
    "metadata": {
        "source": "...",
        "pdf_page": ...,
        "page_label": "..."
    }
}
```

## Metadata Design

- `source` identifies the original document.
- `pdf_page` identifies the physical page position inside the PDF.
- `page_label` preserves the page label defined by the PDF itself.
- Both `pdf_page` and `page_label` are preserved because PDF page indexing and printed/document page numbering can differ.

## Validation

Current PDF:

- Total PDF pages: **308**
- Extracted page-level documents: **308**
- Empty pages skipped: **0** for the current PDF
- PyMuPDF extraction verified successfully.
- Metadata verified successfully.

## Status

Feature 1 — Knowledge Base Setup: **Completed**

Feature 2 — PDF Document Ingestion: **Completed**

### Next Milestone

Feature 3 — Document Chunking

Next session:

- Learn why chunking is required in RAG.
- Decide the chunking strategy.
- Determine appropriate chunk size and overlap.
- Chunk the page-level documents while preserving metadata.
- Test the resulting chunks.
