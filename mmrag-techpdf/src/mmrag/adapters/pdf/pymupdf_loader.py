import fitz  # PyMuPDF
from pathlib import Path
from typing import List

from ...domain.document import Document, Page, Block
from ...ports.pdf_loader import PDFLoader


class PyMuPDFLoader(PDFLoader):
    """PDF loader implementation using PyMuPDF."""

    def load(self, file_path: str) -> Document:
        """Load a PDF document and extract basic text blocks."""
        path = Path(file_path)
        doc = fitz.open(path)

        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_blocks = page.get_text("dict")["blocks"]

            blocks = []
            for block in text_blocks:
                if "lines" in block:
                    text = "\n".join(
                        " ".join(span["text"] for span in line["spans"])
                        for line in block["lines"]
                    )
                    bbox = tuple(block["bbox"])
                    blocks.append(Block(
                        text=text,
                        bbox=bbox,
                        block_type="text"
                    ))

            pages.append(Page(
                number=page_num + 1,
                blocks=blocks
            ))

        doc.close()

        return Document(
            title=path.stem,
            pages=pages,
            file_path=str(path)
        )