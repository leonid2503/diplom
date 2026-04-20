from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from ...domain.document import Block, Page
from ...ports.layout_analyzer import LayoutAnalyzer


class HeuristicLayoutAnalyzer(LayoutAnalyzer):
    """
    Layout analyzer that uses PyMuPDF's dict output to classify blocks.

    Block type heuristics
    ---------------------
    * ``type == 1`` in PyMuPDF → image block → "figure"
    * Text block whose text contains many ``\\t`` characters or whose lines
      are short and column-aligned → "table"
    * Everything else → "text"

    This is intentionally simple and works well on single-column papers.
    For complex multi-column layouts consider an ML-based detector
    (e.g. LayoutParser / DocLayNet).
    """

    def analyze_page(self, page_image_path: str, page_number: int) -> List[Block]:
        raise NotImplementedError(
            "HeuristicLayoutAnalyzer operates on PDF paths, not rasterized images. "
            "Use analyze_document() instead."
        )

    def analyze_document(self, document_path: str) -> List[Page]:
        doc = fitz.open(document_path)
        pages: List[Page] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            raw = page.get_text("dict")
            blocks: List[Block] = []

            for block in raw.get("blocks", []):
                b_type = block.get("type", 0)
                bbox = tuple(block["bbox"])  # (x0, y0, x1, y1)

                if b_type == 1:
                    # Image XObject – already extracted by BasicFigureExtractor
                    blocks.append(Block(text="", bbox=bbox, block_type="figure"))
                    continue

                # Text block
                lines = block.get("lines", [])
                text = "\n".join(
                    " ".join(span["text"] for span in line.get("spans", []))
                    for line in lines
                )
                if not text.strip():
                    continue

                block_type = self._classify_text_block(text, lines)
                blocks.append(Block(text=text, bbox=bbox, block_type=block_type))

            pages.append(Page(number=page_num + 1, blocks=blocks))

        doc.close()
        return pages

    def _classify_text_block(self, text: str, lines: list) -> str:
        """Simple heuristic: many tab chars or very short uniform lines → table."""
        if text.count("\t") > 4:
            return "table"
        if len(lines) > 2:
            lengths = [len(l.get("spans", [])) for l in lines]
            avg = sum(lengths) / len(lengths)
            # Many single-word cells laid out uniformly → table
            if avg <= 2 and len(lines) >= 4:
                return "table"
        return "text"
