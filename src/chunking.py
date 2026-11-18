"""Step 2 - Chunking: split the transcript into overlapping documents."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
import bisect

def chunk_text(text_or_docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    if isinstance(text_or_docs, str):
        return splitter.create_documents([text_or_docs])
    return splitter.split_documents(text_or_docs)

def chunk_transcript(text, cues, video_id, chunk_size=1000, chunk_overlap=200):
    """Split a timed transcript and give every chunk citation metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,               # each chunk records its position in `text`
    )
    offsets = [c[0] for c in cues]
    docs = splitter.create_documents([text])
    for i, d in enumerate(docs):
        # last caption that begins at or before this chunk's start
        j = bisect.bisect_right(offsets, d.metadata["start_index"]) - 1
        start = int(cues[max(j, 0)][1])
        d.metadata.update({
            "video_id": video_id,
            "start_time": start,
            "chunk_index": i,
            "url": f"https://youtu.be/{video_id}?t={start}",
        })
    return docs