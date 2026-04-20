from typing import List

from ...domain.artifact import Artifact
from ...domain.query import Query
from ...ports.reranker import Reranker


class CrossEncoderReranker(Reranker):
    """
    Reranker backed by a sentence-transformers CrossEncoder model.

    Default model: ``cross-encoder/ms-marco-MiniLM-L-6-v2``
    Alternatives  : ``cross-encoder/ms-marco-MiniLM-L-12-v2`` (slower, better)

    The CrossEncoder is loaded lazily on first call.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None  # lazy

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: Query, artifacts: List[Artifact], top_k: int = 5) -> List[Artifact]:
        if not artifacts:
            return []

        model = self._get_model()

        # Build (query, passage) pairs – CrossEncoder expects string pairs
        pairs = [(query.text, self._artifact_text(a)) for a in artifacts]
        scores = model.predict(pairs)

        ranked = sorted(zip(scores, artifacts), key=lambda x: x[0], reverse=True)
        return [artifact for _score, artifact in ranked[:top_k]]

    @staticmethod
    def _artifact_text(artifact: Artifact) -> str:
        parts = [artifact.content]
        if artifact.caption:
            parts.append(artifact.caption)
        # Truncate to avoid exceeding typical max token limit (~512 tokens)
        combined = " ".join(parts)
        return combined[:1500]
