from abc import ABC, abstractmethod
from typing import Protocol

from ..domain.document import Document


class PDFLoader(ABC):
    """Interface for loading and parsing PDF documents."""

    @abstractmethod
    def load(self, file_path: str) -> Document:
        """Load a PDF document and return a Document object."""
        pass