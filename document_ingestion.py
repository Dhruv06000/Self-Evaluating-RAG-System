import pymupdf
from tqdm.auto import tqdm


def text_format(text: str) -> str:
    """Clean extracted PDF text."""
    clear_text = text.replace("\n", " ").strip()
    return clear_text


def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """
    Extract PDF content page by page.

    Each PDF page is converted into a list of text blocks/paragraphs,
    while page information is stored as metadata.
    """

    doc = pymupdf.open(pdf_path)

    pages_and_texts = []

    for page_number, page in tqdm(
        enumerate(doc, start=1),
        total=len(doc)
    ):
        blocks = page.get_text("blocks")

        paragraphs = []

        for block in blocks:

            # A block contains:
            # x0, y0, x1, y1, text, block_number, block_type
            x0 = block[0]
            y0 = block[1]
            x1 = block[2]
            y1 = block[3]
            text = block[4]

            text = text_format(text)

            if not text:
                continue

            # Remove page numbers located at the bottom of the page
            if (
                text.isdigit()
                and y1 > page.rect.height - 50
            ):
                continue
            

            paragraphs.append(text)

        if not paragraphs:
            continue

        pages_and_texts.append(
            {
                "paragraphs": paragraphs,
                "metadata": {
                    "source": pdf_path,
                    "pdf_page": page_number,
                    "page_label": page.get_label(),
                },
            }
        )

    doc.close()

    return pages_and_texts



