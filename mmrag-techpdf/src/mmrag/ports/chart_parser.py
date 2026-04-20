from abc import ABC, abstractmethod


class ChartParser(ABC):
    """Interface for parsing structured data out of chart images."""

    @abstractmethod
    def parse(self, image_path: str) -> str:
        """
        Parse a chart image and return a textual description of its data,
        e.g. axis labels, series names, key values.
        """
        pass
