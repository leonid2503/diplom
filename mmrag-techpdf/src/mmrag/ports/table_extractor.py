from abc import ABC, abstractmethod
from typing import List

from ..domain.artifact import Artifact


class TableExtractor(ABC):
    """Interface for extracting tables from a PDF file."""

    @abstractmethod
    def extract(self, file_path: str) -> List[Artifact]:
        """Extract all tables from the PDF at file_path as Artifacts."""
        pass
