from abc import ABC, abstractmethod
from typing import List

from ..domain.artifact import Artifact
from ..domain.query import Query


class Reranker(ABC):
    """Interface for reranking retrieved artifacts by relevance to a query."""

    @abstractmethod
    def rerank(self, query: Query, artifacts: List[Artifact], top_k: int = 5) -> List[Artifact]:
        """Return the top_k most relevant artifacts, ordered by score descending."""
        pass
