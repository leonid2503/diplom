"""Tests for FAISSStore: add, search, normalization, and persistence."""
from pathlib import Path

import numpy as np
import pytest

from mmrag.adapters.storage.faiss_store import FAISSStore


@pytest.fixture
def store(dim):
    return FAISSStore(dimension=dim)


@pytest.fixture
def populated_store(store, random_embeddings):
    ids = ["id-0", "id-1", "id-2"]
    store.add(random_embeddings, ids)
    return store, ids


# ---------------------------------------------------------------------------
# Add
# ---------------------------------------------------------------------------

class TestAdd:
    def test_add_extends_ids(self, store, random_embeddings):
        ids = ["a", "b", "c"]
        store.add(random_embeddings, ids)
        assert store.ids == ids

    def test_add_twice_accumulates(self, store, dim):
        rng = np.random.default_rng(0)
        emb1 = rng.random((2, dim)).astype(np.float32)
        emb2 = rng.random((2, dim)).astype(np.float32)
        store.add(emb1, ["a", "b"])
        store.add(emb2, ["c", "d"])
        assert store.ids == ["a", "b", "c", "d"]

    def test_add_empty_does_nothing(self, store, dim):
        store.add(np.empty((0, dim), dtype=np.float32), [])
        assert store.ids == []


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_returns_list_of_tuples(self, populated_store, dim):
        store, _ = populated_store
        rng = np.random.default_rng(99)
        q = rng.random((dim,)).astype(np.float32)
        results = store.search(q, top_k=2)
        assert isinstance(results, list)
        assert len(results) == 2
        for item in results:
            assert len(item) == 2  # (id, score)
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)

    def test_exact_match_scores_highest(self, store, dim):
        # Use a single-entry store so the query IS the vector
        vec = np.ones((1, dim), dtype=np.float32)
        store.add(vec, ["exact"])
        query = np.ones(dim, dtype=np.float32)
        results = store.search(query, top_k=1)
        assert results[0][0] == "exact"
        assert abs(results[0][1] - 1.0) < 1e-5  # cosine of identical vectors = 1

    def test_top_k_capped_by_index_size(self, populated_store, dim):
        store, _ = populated_store
        q = np.ones(dim, dtype=np.float32)
        results = store.search(q, top_k=100)
        assert len(results) <= 3  # only 3 items indexed

    def test_search_empty_store_returns_empty(self, store, dim):
        q = np.ones(dim, dtype=np.float32)
        results = store.search(q, top_k=5)
        assert results == []

    def test_ids_in_results_are_valid(self, populated_store, dim):
        store, ids = populated_store
        q = np.ones(dim, dtype=np.float32)
        results = store.search(q, top_k=3)
        returned_ids = [r[0] for r in results]
        assert all(rid in ids for rid in returned_ids)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_roundtrip(self, populated_store, tmp_path, dim):
        store, ids = populated_store
        store.save(str(tmp_path))

        new_store = FAISSStore(dimension=dim)
        new_store.load(str(tmp_path))
        assert new_store.ids == ids

    def test_save_creates_faiss_index_file(self, populated_store, tmp_path):
        store, _ = populated_store
        store.save(str(tmp_path))
        assert (tmp_path / "faiss.index").exists()
        assert (tmp_path / "ids.pkl").exists()

    def test_loaded_store_gives_same_results(self, populated_store, tmp_path, dim):
        store, ids = populated_store
        q = np.ones(dim, dtype=np.float32)
        original_results = store.search(q, top_k=3)

        store.save(str(tmp_path))
        new_store = FAISSStore(dimension=dim)
        new_store.load(str(tmp_path))
        loaded_results = new_store.search(q, top_k=3)

        assert [r[0] for r in original_results] == [r[0] for r in loaded_results]
