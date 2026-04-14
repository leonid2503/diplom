from abc import ABC, abstractmethod
from typing import List

from ..domain.artifact import Artifact
from ..domain.query import Query


class Retriever(ABC):
    """Interface for retrieving relevant artifacts for a query."""

    @abstractmethod
    def retrieve(self, query: Query, top_k: int = 10) -> List[Artifact]:
        """Retrieve relevant artifacts for the query."""
        pass