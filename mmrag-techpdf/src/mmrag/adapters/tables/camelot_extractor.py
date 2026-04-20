import uuid
from typing import List

import fitz  # PyMuPDF

from ...domain.artifact import Artifact, Provenance
from ...ports.table_extractor import TableExtractor


class PyMuPDFTableExtractor(TableExtractor):
    """
    Extracts tables from a PDF using PyMuPDF's built-in table finder
    (available since PyMuPDF ≥ 1.23).

    Each detected table is serialised to Markdown and stored as an
    Artifact of type ``"table"``.  The Markdown representation is
    intentionally verbose so that downstream embedding captures all
    column/row relationships.

    Falls back gracefully to an empty list if no tables are found on a
    given page or if ``find_tables()`` is unavailable.
    """

    def extract(self, file_path: str) -> List[Artifact]:
        doc = fitz.open(file_path)
        artifacts: List[Artifact] = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            try:
                tabs = page.find_tables()
            except AttributeError:
                # PyMuPDF version too old
                break

            for tab in tabs:
                try:
                    markdown = self._table_to_markdown(tab)
                except Exception:
                    continue

                if not markdown.strip():
                    continue

                bbox = (tab.bbox.x0, tab.bbox.y0, tab.bbox.x1, tab.bbox.y1)
                artifacts.append(
                    Artifact(
                        id=str(uuid.uuid4()),
                        artifact_type="table",
                        content=markdown,
                        provenance=Provenance(
                            document_path=file_path,
                            page_number=page_num + 1,
                            bbox=bbox,
                        ),
                    )
                )

        doc.close()
        return artifacts

    def _table_to_markdown(self, tab) -> str:
        """Convert a PyMuPDF Table object to a Markdown table string."""
        rows = tab.extract()  # list[list[str | None]]
        if not rows:
            return ""

        def cell(v) -> str:
            return str(v).strip().replace("\n", " ") if v is not None else ""

        header = rows[0]
        lines = []
        lines.append("| " + " | ".join(cell(c) for c in header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for row in rows[1:]:
            lines.append("| " + " | ".join(cell(c) for c in row) + " |")

        return "\n".join(lines)
