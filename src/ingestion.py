"""Step 1 - Ingestion: fetch the YouTube transcript as plain text."""

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound ,VideoUnavailable, CouldNotRetrieveTranscript
import youtube_transcript_api as yta
import glob, os, re , json
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

MAX_TRANSCRIPT_CHARS = 150_000
_ID = r"[A-Za-z0-9_-]{11}"
_BLOCKED = tuple(
    c for c in (getattr(yta, "RequestBlocked", None), getattr(yta, "IpBlocked", None)) if c
)


class VideoError(Exception):
    """An error whose message is safe to show to the user."""


def parse_video_id(text: str) -> str:
    """Accept a full YouTube URL (watch, youtu.be, shorts, embed, live) or a bare 11-char ID."""
    text = (text or "").strip()
    if re.fullmatch(_ID, text):
        return text
    m = re.search(
        r"(?:youtube\.com/(?:watch\?(?:[^\s&]*&)*v=|shorts/|embed/|live/)|youtu\.be/)(" + _ID + ")",
        text,
    )
    if m:
        return m.group(1)
    raise VideoError("That doesn't look like a YouTube link. Paste a link like https://www.youtube.com/watch?v=...")


def ingest_transcript_timed(video_id: str, languages=ENGLISH_CODES):
    """Return (text, cues). Raises VideoError with a friendly message on any failure."""
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=list(languages))
    except TranscriptsDisabled:
        raise VideoError("This video has captions turned off, so I can't read it.")
    except NoTranscriptFound:
        raise VideoError("This video has no English captions.")
    except VideoUnavailable:
        raise VideoError("This video is unavailable or private.")
    except _BLOCKED:
        raise VideoError("YouTube is blocking transcript requests right now. Please try again later.")
    except CouldNotRetrieveTranscript:
        raise VideoError("I couldn't get the transcript for this video.")
    except Exception:
        raise VideoError("Something went wrong while fetching the video. Please try again.")

    parts, cues, pos = [], [], 0
    for s in fetched:
        t = " ".join(s.text.split())
        if not t:
            continue
        cues.append((pos, s.start))
        parts.append(t)
        pos += len(t) + 1
    text = " ".join(parts)

    if len(text) > MAX_TRANSCRIPT_CHARS:
        raise VideoError(
            f"This video is too long ({len(text):,} characters). The limit is {MAX_TRANSCRIPT_CHARS:,}."
        )
    return text, cues


def get_transcript_data(video_id: str):
    """Fetch once, cache on disk. Returns (text, cues)."""
    path = f"data/{video_id}_transcript.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d["text"], d["cues"]
    text, cues = ingest_transcript_timed(video_id)
    os.makedirs("data", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"text": text, "cues": cues}, f, ensure_ascii=False)
    return text, cues

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