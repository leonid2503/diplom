from abc import ABC, abstractmethod
from typing import List

from ..domain.artifact import Artifact
from ..domain.query import Query, Answer


class Generator(ABC):
    """Interface for generating answers from retrieved artifacts."""

    @abstractmethod
    def generate(self, query: Query, artifacts: List[Artifact]) -> Answer:
        """Generate an answer for the query using the provided artifacts."""
        pass