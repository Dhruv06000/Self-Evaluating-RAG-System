# Mine code 
import re

import io

from contextlib import redirect_stdout

from transformers import AutoTokenizer

from document_ingestion import open_and_read_pdf


def is_heading(text):

    pattern = r"^(Chapter\s+\d+|\d+\.[1-9]\d*(?:\.\d+)?)\s+.+"

    section_numbers = re.findall(
        r"\b\d+\.[1-9]\d*(?:\.\d+)?\b",
        text
    )

    if text.startswith("Chapter"):
        return bool(re.match(pattern, text))

    if re.search(r"\s+\d+$", text):
        return False

    return bool(
        re.match(pattern, text)
        and len(section_numbers) == 1
    )


tokenizer = AutoTokenizer.from_pretrained(
    "BAAI/bge-base-en-v1.5"
)


with redirect_stdout(io.StringIO()):
    pages_and_texts = open_and_read_pdf(
        "knowledge_base/artificial_intelligence_technology.pdf"
    )


def create_chunks(pages_and_texts, chunk_size=350):

    chunks = []

    current_heading = None

    current_paragraphs = []

    for page in pages_and_texts:

        for paragraph in page["paragraphs"]:

            if is_heading(paragraph):

                if current_heading is not None:

                    chunks.append({
                        "heading": current_heading,
                        "paragraphs": current_paragraphs
                    })

                current_heading = paragraph

                current_paragraphs = []

            else:

                current_paragraphs.append({
                    "text": paragraph,
                    "metadata": page["metadata"]
                })

    if current_heading is not None:

        chunks.append({
            "heading": current_heading,
            "paragraphs": current_paragraphs
        })

    return chunks


chunk = create_chunks(pages_and_texts)


def count_tokens(text):
    return len(
        tokenizer.backend_tokenizer.encode(
            text,
            add_special_tokens=False
        ).ids
    )

def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

def split_into_words(text):
    return text.split()

def build_paragraph_chunks(heading_data, chunk_size=350):
    chunks = []
    current_text = []
    current_token_count = 0
    current_pages = []
    source = None

    for paragraph in heading_data["paragraphs"]:
        paragraph_text = paragraph["text"]
        paragraph_metadata = paragraph["metadata"]

        if source is None:
            source = paragraph_metadata["source"]

        page_label = paragraph_metadata["page_label"]

        if page_label not in current_pages:
            current_pages.append(page_label)

        paragraph_token = count_tokens(paragraph_text)

        if paragraph_token <= chunk_size:
            if current_token_count + paragraph_token <= chunk_size:
                current_text.append(paragraph_text)
                current_token_count += paragraph_token

            else:
                if current_text:
                    chunks.append({
                        "text": "\n\n".join(current_text),
                        "heading": heading_data["heading"],
                        "token_count": current_token_count,
                        "metadata": {
                            "source": source,
                            "pages": [
                                {"page_label": page}
                                for page in current_pages
                            ]
                        }
                    })

                current_text = [paragraph_text]
                current_token_count = paragraph_token
                current_pages = [page_label]

        else:
            sentences = split_into_sentences(paragraph_text)

            for sentence in sentences:
                sentence_token = count_tokens(sentence)

                if sentence_token <= chunk_size:
                    if current_token_count + sentence_token <= chunk_size:
                        current_text.append(sentence)
                        current_token_count += sentence_token

                    else:
                        if current_text:
                            chunks.append({
                                "text": "\n\n".join(current_text),
                                "heading": heading_data["heading"],
                                "token_count": current_token_count,
                                "metadata": {
                                    "source": source,
                                    "pages": [
                                        {"page_label": page}
                                        for page in current_pages
                                    ]
                                }
                            })

                        current_text = [sentence]
                        current_token_count = sentence_token
                        current_pages = [page_label]

                else:
                    words = split_into_words(sentence)

                    for word in words:
                        word_token = count_tokens(word)

                        if current_token_count + word_token <= chunk_size:
                            current_text.append(word)
                            current_token_count += word_token

                        else:
                            if current_text:
                                chunks.append({
                                    "text": " ".join(current_text),
                                    "heading": heading_data["heading"],
                                    "token_count": current_token_count,
                                    "metadata": {
                                        "source": source,
                                        "pages": [
                                            {"page_label": page}
                                            for page in current_pages
                                        ]
                                    }
                                })

                            current_text = [word]
                            current_token_count = word_token
                            current_pages = [page_label]

    if current_text:
        chunks.append({
            "text": "\n\n".join(current_text),
            "heading": heading_data["heading"],
            "token_count": current_token_count,
            "metadata": {
                "source": source,
                "pages": [
                    {"page_label": page}
                    for page in current_pages
                ]
            }
        })

    return chunks


all_final_chunks = []

for heading_data in chunk:

    result = build_paragraph_chunks(heading_data)

    all_final_chunks.extend(result)

print("\nFINAL CHUNK VALIDATION")
print("======================")

# 1. Total chunks
print("Total chunks:", len(all_final_chunks))

# 2. Token statistics
token_counts = [chunk["token_count"] for chunk in all_final_chunks]

print("Minimum tokens:", min(token_counts))
print("Maximum tokens:", max(token_counts))
print("Average tokens:", round(sum(token_counts) / len(token_counts), 2))

# 3. Chunks over the limit
over_limit = [
    chunk for chunk in all_final_chunks
    if chunk["token_count"] > 350
]

print("Chunks over 350 tokens:", len(over_limit))

# 4. Empty text
empty_text = [
    chunk for chunk in all_final_chunks
    if not chunk["text"].strip()
]

print("Chunks with empty text:", len(empty_text))

# 5. Missing heading
missing_heading = [
    chunk for chunk in all_final_chunks
    if not chunk["heading"]
]

print("Chunks without heading:", len(missing_heading))

# 6. Missing metadata
missing_metadata = [
    chunk for chunk in all_final_chunks
    if (
        not chunk.get("metadata")
        or not chunk["metadata"].get("source")
        or not chunk["metadata"].get("pages")
    )
]

print("Chunks with missing metadata:", len(missing_metadata))