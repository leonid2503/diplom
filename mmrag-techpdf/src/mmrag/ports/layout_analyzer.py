from abc import ABC, abstractmethod
from typing import List

from ..domain.document import Page, Block


class LayoutAnalyzer(ABC):
    """Interface for analyzing document layout and extracting blocks."""

    @abstractmethod
    def analyze_page(self, page_image_path: str, page_number: int) -> List[Block]:
        """Analyze a page image and return extracted blocks."""
        pass

    @abstractmethod
    def analyze_document(self, document_path: str) -> List[Page]:
        """Analyze entire document and return pages with blocks."""
        pass