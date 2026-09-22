import re
import io
import json

from contextlib import redirect_stdout
from transformers import AutoTokenizer

from document_ingestion import open_and_read_pdf


def is_heading(text):

    pattern = r"^(Chapter\s+\d+|\d+\.[1-9]\d*(?:\.\d+)?)\s+.+"

    section_numbers = re.findall(
        r"\b\d+\.[1-9]\d*(?:\.\d+)?\b",
        text
    )

    if text.startswith("Appendix"):
        return bool(re.match(r"^Appendix\s+\d+:\s+.+", text))

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


def build_paragraph_chunks(heading_data, chunk_size=350, start_index = 0):

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
                        "chunk_id": f"chunk_{start_index + len(chunks) +1 :06d}",
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
                                "chunk_id": f"chunk_{start_index + len(chunks) +1 :06d}",
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
                                    "chunk_id": f"chunk_{start_index + len(chunks) +1 :06d}",
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
            "chunk_id": f"chunk_{start_index + len(chunks) +1 :06d}",
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


def generate_chunks():

    with redirect_stdout(io.StringIO()):

        pages_and_texts = open_and_read_pdf(
            "knowledge_base/artificial_intelligence_technology.pdf"
        )

    chunk = create_chunks(pages_and_texts)

    all_final_chunks = []

    for heading_data in chunk:
        # Skip Appendix 2 because it contains answer keys for the exercise questions
        # used in evaluation. Including it in the retrieval corpus would cause data leakage.
        if heading_data["heading"].startswith("Appendix 2: Key to Exercises"):
            continue

        result = build_paragraph_chunks(
            heading_data,
            start_index = len(all_final_chunks))

        all_final_chunks.extend(result)

    return all_final_chunks


if __name__ == "__main__":

    all_final_chunks = generate_chunks()

    with open("data/chunks.json", "w", encoding="utf-8") as file:

        json.dump(
            all_final_chunks,
            file,
            ensure_ascii=False,
            indent=4
        )

    print(
        f"Chunks generated and saved successfully. "
        f"Total chunks: {len(all_final_chunks)}"
    )
    print(all_final_chunks[0]["chunk_id"])
    print(all_final_chunks[-1]["chunk_id"])
    print(len(all_final_chunks))
    ids = [chunk["chunk_id"] for chunk in all_final_chunks]
    print(len(ids) == len(set(ids)))