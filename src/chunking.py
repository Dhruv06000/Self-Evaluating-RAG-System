"""Step 2 - Chunking: split the transcript into overlapping documents."""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text_or_docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    if isinstance(text_or_docs, str):
        return splitter.create_documents([text_or_docs])
    return splitter.split_documents(text_or_docs)