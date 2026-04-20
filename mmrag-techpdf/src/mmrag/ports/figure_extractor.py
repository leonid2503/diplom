from abc import ABC, abstractmethod
from typing import List

from ..domain.artifact import Artifact
from ..domain.document import Document


class FigureExtractor(ABC):
    """Interface for extracting figures from a PDF document."""

    @abstractmethod
    def extract(self, document: Document, output_dir: str) -> List[Artifact]:
        """Extract all figures from the document, saving cropped images to output_dir."""
        pass
