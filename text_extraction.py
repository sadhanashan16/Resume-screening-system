"""
text_extraction.py
-------------------
Reads resume files (.pdf, .docx, .doc, .txt) and returns their raw text.

Beginner note: PDFs and Word docs don't store text the way a .txt file does,
so we need special libraries to pull the words out:
    - PyPDF2      -> reads .pdf files
    - python-docx -> reads .docx files
"""

import os
from PyPDF2 import PdfReader
import docx


def extract_text_from_pdf(file_path: str) -> str:
    """Pull all text out of a PDF file, page by page."""
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def extract_text_from_docx(file_path: str) -> str:
    """Pull all text out of a .docx file, paragraph by paragraph."""
    document = docx.Document(file_path)
    return "\n".join(p.text for p in document.paragraphs)


def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text(file_path: str) -> str:
    """
    Main entry point: figures out the file type from its extension
    and calls the right extractor.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in (".docx",):
        return extract_text_from_docx(file_path)
    elif ext == ".doc":
        # Old binary .doc format needs extra tools (e.g. antiword/libreoffice)
        # not available in a plain Python install. We raise a clear error
        # instead of failing silently.
        raise ValueError(
            f"Legacy .doc format not supported for '{file_path}'. "
            "Please convert it to .docx or .pdf first."
        )
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def load_resumes_from_folder(folder_path: str) -> dict:
    """
    Reads every supported resume file in a folder.
    Returns a dict: { filename: extracted_text }
    """
    resumes = {}
    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if not os.path.isfile(full_path):
            continue
        try:
            resumes[filename] = extract_text(full_path)
        except ValueError as e:
            print(f"Skipping {filename}: {e}")
    return resumes
