from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class IndexStore(ABC):
    """Interface for storing and searching document embeddings."""

    @abstractmethod
    def add(self, embeddings: np.ndarray, ids: List[str]) -> None:
        """Add embeddings with their IDs to the index."""
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar embeddings, return (id, score) pairs."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the index to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the index from disk."""
        pass