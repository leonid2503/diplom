"""Tests for LocalDocStore: CRUD operations and pickle persistence."""
import pickle
from pathlib import Path

import pytest

from mmrag.adapters.storage.local_docstore import LocalDocStore

from conftest import make_artifact


@pytest.fixture
def store():
    return LocalDocStore()


@pytest.fixture
def populated_store(sample_artifacts):
    s = LocalDocStore()
    for a in sample_artifacts:
        s.put(a)
    return s


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

class TestCRUD:
    def test_put_and_get(self, store, sample_artifact):
        store.put(sample_artifact)
        result = store.get(sample_artifact.id)
        assert result is not None
        assert result.id == sample_artifact.id

    def test_get_missing_returns_none(self, store):
        assert store.get("nonexistent") is None

    def test_put_overwrites(self, store):
        a = make_artifact(id="dup", content="original")
        store.put(a)
        a2 = make_artifact(id="dup", content="updated")
        store.put(a2)
        assert store.get("dup").content == "updated"

    def test_get_many(self, populated_store, sample_artifacts):
        ids = [a.id for a in sample_artifacts[:2]]
        results = populated_store.get_many(ids)
        assert len(results) == 2
        assert {r.id for r in results} == set(ids)

    def test_get_many_skips_missing(self, populated_store):
        results = populated_store.get_many(["art-1", "ghost"])
        assert len(results) == 1
        assert results[0].id == "art-1"

    def test_get_all(self, populated_store, sample_artifacts):
        all_arts = populated_store.get_all()
        assert len(all_arts) == len(sample_artifacts)

    def test_len(self, store, sample_artifacts):
        assert len(store) == 0
        for a in sample_artifacts:
            store.put(a)
        assert len(store) == len(sample_artifacts)

    def test_empty_store_get_all(self, store):
        assert store.get_all() == []


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_roundtrip(self, populated_store, tmp_path, sample_artifacts):
        path = str(tmp_path / "docstore.pkl")
        populated_store.save(path)

        new_store = LocalDocStore()
        new_store.load(path)
        assert len(new_store) == len(sample_artifacts)
        for a in sample_artifacts:
            loaded = new_store.get(a.id)
            assert loaded is not None
            assert loaded.content == a.content

    def test_save_creates_parent_dirs(self, store, sample_artifact, tmp_path):
        path = str(tmp_path / "nested" / "deep" / "store.pkl")
        store.put(sample_artifact)
        store.save(path)
        assert Path(path).exists()

    def test_save_produces_valid_pickle(self, populated_store, tmp_path):
        path = tmp_path / "store.pkl"
        populated_store.save(str(path))
        with open(path, "rb") as f:
            data = pickle.load(f)
        assert isinstance(data, dict)
        assert "art-1" in data
