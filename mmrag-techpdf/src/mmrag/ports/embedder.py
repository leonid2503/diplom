from abc import ABC, abstractmethod
from typing import List
import numpy as np


class Embedder(ABC):
    """Interface for generating embeddings from text."""

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into embeddings."""
        pass

    @abstractmethod
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        pass