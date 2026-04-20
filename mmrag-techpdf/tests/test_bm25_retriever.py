"""Tests for BM25Retriever: indexing, retrieval, scoring, and persistence."""
import pickle
from pathlib import Path

import pytest

from mmrag.adapters.retrieval.bm25_retriever import BM25Retriever, _tokenize
from mmrag.domain.query import Query

from conftest import make_artifact


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_punctuation_stripped(self):
        tokens = _tokenize("neural-network, deep.learning!")
        assert "neural" in tokens
        assert "network" in tokens
        assert "deep" in tokens
        assert "learning" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_numbers_kept(self):
        tokens = _tokenize("model v2 with 512 dimensions")
        assert "512" in tokens
        assert "v2" in tokens


# ---------------------------------------------------------------------------
# Build / retrieve
# ---------------------------------------------------------------------------

@pytest.fixture
def retriever(sample_artifacts):
    r = BM25Retriever()
    r.build(sample_artifacts)
    return r


class TestBuild:
    def test_empty_before_build(self):
        r = BM25Retriever()
        q = Query(text="anything")
        assert r.retrieve(q) == []

    def test_build_stores_artifacts(self, retriever, sample_artifacts):
        assert len(retriever._artifacts) == len(sample_artifacts)

    def test_build_creates_bm25(self, retriever):
        assert retriever._bm25 is not None


class TestRetrieve:
    def test_returns_list(self, retriever, sample_query):
        results = retriever.retrieve(sample_query)
        assert isinstance(results, list)

    def test_top_k_respected(self, retriever, sample_query):
        results = retriever.retrieve(sample_query, top_k=2)
        assert len(results) <= 2

    def test_relevant_artifact_ranked_first(self, retriever):
        q = Query(text="transformer attention")
        results = retriever.retrieve(q, top_k=3)
        assert results[0].id == "art-2"

    def test_retrieve_returns_artifacts(self, retriever, sample_query):
        from mmrag.domain.artifact import Artifact
        results = retriever.retrieve(sample_query, top_k=1)
        assert all(isinstance(r, Artifact) for r in results)

    def test_retrieve_on_empty_retriever(self):
        r = BM25Retriever()
        assert r.retrieve(Query(text="test")) == []


# ---------------------------------------------------------------------------
# get_scores
# ---------------------------------------------------------------------------

class TestGetScores:
    def test_scores_length_matches_artifacts(self, retriever, sample_artifacts):
        scores = retriever.get_scores("neural")
        assert len(scores) == len(sample_artifacts)

    def test_scores_are_floats(self, retriever):
        scores = retriever.get_scores("deep learning")
        assert all(isinstance(s, float) for s in scores)

    def test_scores_on_empty_retriever(self):
        r = BM25Retriever()
        assert r.get_scores("anything") == []

    def test_relevant_score_is_higher(self, retriever):
        scores = retriever.get_scores("convolutional image classification")
        # art-3 is about CNN image classification → index 2
        assert scores[2] > scores[1]


# ---------------------------------------------------------------------------
# _artifact_text helper
# ---------------------------------------------------------------------------

class TestArtifactText:
    def test_content_only(self):
        a = make_artifact(content="some text", caption=None)
        assert BM25Retriever._artifact_text(a) == "some text"

    def test_content_and_caption(self):
        a = make_artifact(content="figure content", caption="a bar chart")
        text = BM25Retriever._artifact_text(a)
        assert "figure content" in text
        assert "a bar chart" in text


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, retriever, sample_artifacts, tmp_path):
        path = str(tmp_path / "bm25.pkl")
        retriever.save(path)

        r2 = BM25Retriever()
        r2.load(path)
        assert len(r2._artifacts) == len(sample_artifacts)
        # Should still retrieve correctly after reload
        results = r2.retrieve(Query(text="transformer attention"), top_k=1)
        assert results[0].id == "art-2"

    def test_save_creates_parent_dirs(self, retriever, tmp_path):
        path = str(tmp_path / "nested" / "bm25.pkl")
        retriever.save(path)
        assert Path(path).exists()

    def test_save_produces_valid_pickle(self, retriever, tmp_path):
        path = tmp_path / "bm25.pkl"
        retriever.save(str(path))
        with open(path, "rb") as f:
            data = pickle.load(f)
        assert "bm25" in data
        assert "artifacts" in data
