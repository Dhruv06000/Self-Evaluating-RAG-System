"""Step 1 - Ingestion: fetch the YouTube transcript as plain text."""

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import glob, os, re
from langchain_core.documents import Document

DATA_DIR = "knowledge_base"
ENGLISH_CODES = ("en", "en-US", "en-GB", "en-IN", "en-CA", "en-AU")

def ingest_transcript(video_id: str, languages=ENGLISH_CODES) -> str:
    """Fetch the transcript for a video ID (not the full URL) and flatten it."""
    languages = list(languages)
    try:
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            # youtube-transcript-api < 1.0
            items = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
            return " ".join(chunk["text"] for chunk in items)

        # youtube-transcript-api >= 1.0
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=languages)
        return " ".join(snippet.text for snippet in fetched)
    

    except TranscriptsDisabled:
        raise RuntimeError("No captions available for this video.")
    except NoTranscriptFound:
        raise RuntimeError(
            f"No transcript in {languages} for video {video_id}. "
            "The video may only have captions in another language."
        )

# 1. LOAD ---- read each transcript, throw away the VTT timestamps

def load_transcripts(folder="knowledge_base"):
    docs = []
    for path in sorted(glob.glob(f"{folder}/*.vtt")):
        lines = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line == "WEBVTT" or "-->" in line:
                    continue
                lines.append(line)
        m = re.search(r"Session[ _]*(\d+)", path)
        session = m.group(1) if m else os.path.basename(path)
        docs.append(Document(page_content=" ".join(lines),
                             metadata={"session": session}))
    return docs

if __name__ == "__main__":
    # Run from the project root:
    #   uv run python rag/ingestion.py                 (default video)
    #   uv run python rag/ingestion.py <video_id>      (any other video)
    import sys
    from pathlib import Path

    video_id = sys.argv[1] if len(sys.argv) > 1 else "foGklduxhM0"

    text = ingest_transcript(video_id)
    print(f"Video ID : {video_id}")
    print(f"Length   : {len(text)} characters, about {len(text.split())} words")

    out_file = Path("src/transcript.txt")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(text, encoding="utf-8")
    print(f"Saved to {out_file}")