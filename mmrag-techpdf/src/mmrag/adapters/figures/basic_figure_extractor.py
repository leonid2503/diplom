import uuid
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from ...domain.artifact import Artifact, Provenance
from ...domain.document import Document
from ...ports.figure_extractor import FigureExtractor


class BasicFigureExtractor(FigureExtractor):
    """
    Extracts raster images embedded inside a PDF using PyMuPDF.

    For each page, iterates over the image XObjects referenced by that page.
    Images smaller than min_size (px in either dimension) are skipped to
    avoid icons, decorators, and single-pixel separators.
    """

    def __init__(self, min_size: int = 80, dpi: int = 150):
        self.min_size = min_size
        self.dpi = dpi

    def extract(self, document: Document, output_dir: str) -> List[Artifact]:
        figures_dir = Path(output_dir) / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(document.file_path)
        artifacts: List[Artifact] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                except Exception:
                    continue

                width = base_image["width"]
                height = base_image["height"]
                if width < self.min_size or height < self.min_size:
                    continue

                ext = base_image["ext"]
                img_bytes = base_image["image"]
                artifact_id = str(uuid.uuid4())
                img_filename = f"{artifact_id}.{ext}"
                img_path = figures_dir / img_filename

                img_path.write_bytes(img_bytes)

                # Attempt to find the bounding box of the image on the page
                bbox = self._find_image_bbox(page, xref)

                artifacts.append(
                    Artifact(
                        id=artifact_id,
                        artifact_type="figure",
                        content="",  # filled in by captioner
                        image_path=str(img_path),
                        provenance=Provenance(
                            document_path=document.file_path,
                            page_number=page_num + 1,
                            bbox=bbox,
                        ),
                    )
                )

        doc.close()
        return artifacts

    def _find_image_bbox(
        self, page: fitz.Page, xref: int
    ) -> tuple[float, float, float, float]:
        """Return the bounding box of an image on the page, or full-page bbox."""
        for item in page.get_image_rects(xref):
            r = item[0] if isinstance(item, tuple) else item
            return (r.x0, r.y0, r.x1, r.y1)
        r = page.rect
        return (r.x0, r.y0, r.x1, r.y1)
