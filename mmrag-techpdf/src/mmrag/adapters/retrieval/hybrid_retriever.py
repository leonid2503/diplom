from typing import List

import numpy as np

from ...domain.artifact import Artifact
from ...domain.query import Query
from ...ports.embedder import Embedder
from ...ports.index_store import IndexStore
from ...ports.retriever import Retriever
from ..storage.local_docstore import LocalDocStore
from .bm25_retriever import BM25Retriever


class HybridRetriever(Retriever):
    """
    Hybrid retriever combining dense (FAISS) and sparse (BM25) signals.

    Fusion strategy: Reciprocal Rank Fusion (RRF).

      RRF score = Σ  weight_i / (rank_i + k)

    where k=60 (standard RRF constant) dampens the impact of very high
    rank differences.  Dense and sparse scores are then combined with
    configurable weights.
    """

    RRF_K = 60

    def __init__(
        self,
        embedder: Embedder,
        index_store: IndexStore,
        doc_store: LocalDocStore,
        bm25_retriever: BM25Retriever,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ):
        self.embedder = embedder
        self.index_store = index_store
        self.doc_store = doc_store
        self.bm25_retriever = bm25_retriever
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def retrieve(self, query: Query, top_k: int = 10) -> List[Artifact]:
        # ----------------------------------------------------------------
        # 1. Dense retrieval
        # ----------------------------------------------------------------
        query_emb = self.embedder.encode_single(query.text)
        # Retrieve 3× top_k candidates from each leg for broader recall
        candidate_k = min(top_k * 3, 100)
        dense_hits = self.index_store.search(query_emb, top_k=candidate_k)
        # dense_hits: List[(artifact_id, score)]

        # ----------------------------------------------------------------
        # 2. Sparse (BM25) retrieval
        # ----------------------------------------------------------------
        bm25_hits = self.bm25_retriever.retrieve(query, top_k=candidate_k)
        # bm25_hits: List[Artifact]

        # ----------------------------------------------------------------
        # 3. Reciprocal Rank Fusion
        # ----------------------------------------------------------------
        rrf_scores: dict[str, float] = {}

        for rank, (artifact_id, _score) in enumerate(dense_hits):
            rrf_scores[artifact_id] = rrf_scores.get(artifact_id, 0.0) + (
                self.dense_weight / (rank + self.RRF_K)
            )

        for rank, artifact in enumerate(bm25_hits):
            rrf_scores[artifact.id] = rrf_scores.get(artifact.id, 0.0) + (
                self.bm25_weight / (rank + self.RRF_K)
            )

        # ----------------------------------------------------------------
        # 4. Sort and resolve artifacts
        # ----------------------------------------------------------------
        sorted_ids = sorted(rrf_scores, key=lambda aid: rrf_scores[aid], reverse=True)
        result_artifacts: List[Artifact] = []
        for aid in sorted_ids[:top_k]:
            artifact = self.doc_store.get(aid)
            if artifact is not None:
                result_artifacts.append(artifact)

        return result_artifacts
