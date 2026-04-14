from typing import List, Tuple
import numpy as np
import faiss
import pickle
from pathlib import Path

from ...ports.index_store import IndexStore


class FAISSStore(IndexStore):
    """FAISS-based index store for embeddings."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.ids = []

    def add(self, embeddings: np.ndarray, ids: List[str]) -> None:
        """Add embeddings with their IDs to the index."""
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms

        self.index.add(normalized_embeddings.astype(np.float32))
        self.ids.extend(ids)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar embeddings."""
        # Normalize query
        norm = np.linalg.norm(query_embedding)
        normalized_query = query_embedding / norm

        scores, indices = self.index.search(
            normalized_query.reshape(1, -1).astype(np.float32),
            min(top_k, len(self.ids))
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.ids):
                results.append((self.ids[idx], float(score)))

        return results

    def save(self, path: str) -> None:
        """Save the index to disk."""
        path = Path(path)
        faiss.write_index(self.index, str(path / "faiss.index"))
        with open(path / "ids.pkl", "wb") as f:
            pickle.dump(self.ids, f)

    def load(self, path: str) -> None:
        """Load the index from disk."""
        path = Path(path)
        self.index = faiss.read_index(str(path / "faiss.index"))
        with open(path / "ids.pkl", "rb") as f:
            self.ids = pickle.load(f)