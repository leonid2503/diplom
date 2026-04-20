import pickle
import re
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi

from ...domain.artifact import Artifact
from ...domain.query import Query
from ...ports.retriever import Retriever


def _tokenize(text: str) -> List[str]:
    """Lowercase, split on non-word characters."""
    return re.findall(r"\w+", text.lower())


class BM25Retriever(Retriever):
    """
    Sparse retriever backed by BM25Okapi.

    The index is built from a list of Artifact objects.  The corpus is
    the concatenation of each artifact's content and caption (if present).
    """

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._artifacts: List[Artifact] = []

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build(self, artifacts: List[Artifact]) -> None:
        """Build a BM25 index from a list of artifacts."""
        self._artifacts = artifacts
        corpus = [self._artifact_text(a) for a in artifacts]
        tokenized = [_tokenize(text) for text in corpus]
        self._bm25 = BM25Okapi(tokenized)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: Query, top_k: int = 10) -> List[Artifact]:
        if self._bm25 is None or not self._artifacts:
            return []

        tokens = _tokenize(query.text)
        scores = self._bm25.get_scores(tokens)

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._artifacts[i] for i in ranked_indices[:top_k]]

    def get_scores(self, query_text: str) -> List[float]:
        """Return raw BM25 scores for all artifacts (used in hybrid fusion)."""
        if self._bm25 is None:
            return []
        tokens = _tokenize(query_text)
        return list(self._bm25.get_scores(tokens))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({"bm25": self._bm25, "artifacts": self._artifacts}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._bm25 = state["bm25"]
        self._artifacts = state["artifacts"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _artifact_text(artifact: Artifact) -> str:
        parts = [artifact.content]
        if artifact.caption:
            parts.append(artifact.caption)
        return " ".join(parts)
