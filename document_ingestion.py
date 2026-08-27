import pymupdf
from tqdm.auto import tqdm

def text_format(text:str)->str:
  """Performs minor formatting on extracted PDF text."""
  clear_text = text.replace("\n"," ").strip()
  return clear_text

def open_and_read_pdf(pdf_path: str)-> list[dict]:
  """
      Opens a PDF file, reads its text content page by page, and collects statistics.
  
      Parameters:
          pdf_path (str): The file path to the PDF document to be opened and read.
  
      Returns:
          list[dict]: A list of dictionaries, each containing the page number
          (adjusted), character count, word count, sentence count, token count, and the extracted text
          for each page.
      """
  doc = pymupdf.open(pdf_path)
  pages_and_texts = []
  for page_number,page in tqdm(enumerate(doc,start=1)):
    text = page.get_text()
    text = text_format(text)
    if not text:
      continue
    pages_and_texts.append({
                            "text": text,
                            "metadata":{
                              "source":pdf_path,
                              "pdf_page": page_number,
                              "page_label":page.get_label()
                            }
    })
  return pages_and_texts
