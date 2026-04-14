from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from ...ports.embedder import Embedder


class SentenceTransformerEmbedder(Embedder):
    """Embedder implementation using Sentence Transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into embeddings."""
        return self.model.encode(texts, convert_to_numpy=True)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding."""
        return self.model.encode([text], convert_to_numpy=True)[0]

    @property
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        return self.model.get_sentence_embedding_dimension()