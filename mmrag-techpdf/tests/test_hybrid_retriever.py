"""Tests for HybridRetriever: RRF fusion, weights, and edge cases."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from mmrag.adapters.retrieval.hybrid_retriever import HybridRetriever
from mmrag.domain.query import Query

from conftest import make_artifact


def _make_retriever(
    dense_hits=None,
    bm25_hits=None,
    docstore_map=None,
    dense_weight=0.6,
    bm25_weight=0.4,
):
    """Build a HybridRetriever with all dependencies mocked."""
    embedder = MagicMock()
    embedder.encode_single.return_value = np.ones(32, dtype=np.float32)

    index_store = MagicMock()
    index_store.search.return_value = dense_hits or []

    bm25_retriever = MagicMock()
    bm25_retriever.retrieve.return_value = bm25_hits or []

    doc_store = MagicMock()
    docstore_map = docstore_map or {}
    doc_store.get.side_effect = lambda aid: docstore_map.get(aid)

    return HybridRetriever(
        embedder=embedder,
        index_store=index_store,
        doc_store=doc_store,
        bm25_retriever=bm25_retriever,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )


@pytest.fixture
def arts():
    return [make_artifact(id=f"art-{i}", content=f"content {i}") for i in range(5)]


class TestRetrieve:
    def test_empty_both_legs_returns_empty(self):
        r = _make_retriever()
        assert r.retrieve(Query(text="x")) == []

    def test_dense_only_returns_top_k(self, arts):
        dense_hits = [(a.id, 0.9 - i * 0.1) for i, a in enumerate(arts)]
        docstore = {a.id: a for a in arts}
        r = _make_retriever(dense_hits=dense_hits, docstore_map=docstore)
        results = r.retrieve(Query(text="x"), top_k=3)
        assert len(results) == 3

    def test_bm25_only_returns_top_k(self, arts):
        bm25_hits = arts[:4]
        docstore = {a.id: a for a in arts}
        r = _make_retriever(bm25_hits=bm25_hits, docstore_map=docstore)
        results = r.retrieve(Query(text="x"), top_k=2)
        assert len(results) == 2

    def test_rrf_prefers_artifact_present_in_both_legs(self, arts):
        # art-0 appears in both dense and BM25 → should score highest
        shared = arts[0]
        other = arts[1]
        dense_hits = [(shared.id, 0.5), (other.id, 0.4)]
        bm25_hits = [shared, other]
        docstore = {a.id: a for a in arts}
        r = _make_retriever(dense_hits=dense_hits, bm25_hits=bm25_hits, docstore_map=docstore)
        results = r.retrieve(Query(text="x"), top_k=2)
        assert results[0].id == shared.id

    def test_artifact_missing_from_docstore_skipped(self, arts):
        # Only art-1 is in docstore; art-0 is not
        dense_hits = [("art-0", 0.9), ("art-1", 0.5)]
        docstore = {"art-1": arts[1]}
        r = _make_retriever(dense_hits=dense_hits, docstore_map=docstore)
        results = r.retrieve(Query(text="x"), top_k=5)
        assert all(res.id != "art-0" for res in results)
        assert any(res.id == "art-1" for res in results)

    def test_top_k_zero_returns_empty(self, arts):
        dense_hits = [(a.id, 0.9) for a in arts]
        docstore = {a.id: a for a in arts}
        r = _make_retriever(dense_hits=dense_hits, docstore_map=docstore)
        results = r.retrieve(Query(text="x"), top_k=0)
        assert results == []

    def test_candidate_k_capped_at_100(self, arts):
        """Verify index_store.search is called with min(top_k*3, 100)."""
        index_store = MagicMock()
        index_store.search.return_value = []
        embedder = MagicMock()
        embedder.encode_single.return_value = np.ones(32)
        bm25 = MagicMock()
        bm25.retrieve.return_value = []
        doc_store = MagicMock()
        doc_store.get.return_value = None

        r = HybridRetriever(embedder, index_store, doc_store, bm25)
        r.retrieve(Query(text="x"), top_k=50)
        # min(50*3, 100) = 100
        _, kwargs = index_store.search.call_args
        assert kwargs["top_k"] == 100


class TestRRFConstants:
    def test_rrf_k_is_60(self):
        assert HybridRetriever.RRF_K == 60

    def test_custom_weights_stored(self):
        r = _make_retriever(dense_weight=0.3, bm25_weight=0.7)
        assert r.dense_weight == 0.3
        assert r.bm25_weight == 0.7
