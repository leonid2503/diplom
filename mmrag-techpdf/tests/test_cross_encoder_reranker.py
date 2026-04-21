"""Tests for CrossEncoderReranker: reranking, truncation, lazy loading."""
from unittest.mock import MagicMock, patch

import pytest

from mmrag.adapters.rerank.cross_encoder_reranker import CrossEncoderReranker
from mmrag.domain.query import Query

from conftest import make_artifact


@pytest.fixture
def reranker():
    return CrossEncoderReranker()


@pytest.fixture
def artifacts():
    return [
        make_artifact(id="a1", content="deep learning basics"),
        make_artifact(id="a2", content="transformer model architecture"),
        make_artifact(id="a3", content="convolutional networks for vision"),
    ]


@pytest.fixture
def mock_model():
    """Return a mock CrossEncoder whose predict is controllable."""
    m = MagicMock()
    return m


# ---------------------------------------------------------------------------
# _artifact_text
# ---------------------------------------------------------------------------

class TestArtifactText:
    def test_content_only(self):
        a = make_artifact(content="hello world", caption=None)
        assert CrossEncoderReranker._artifact_text(a) == "hello world"

    def test_content_and_caption_joined(self):
        a = make_artifact(content="figure", caption="a line chart")
        text = CrossEncoderReranker._artifact_text(a)
        assert "figure" in text
        assert "a line chart" in text

    def test_truncated_at_1500_chars(self):
        long_content = "x" * 2000
        a = make_artifact(content=long_content)
        text = CrossEncoderReranker._artifact_text(a)
        assert len(text) == 1500

    def test_short_content_not_truncated(self):
        a = make_artifact(content="short", caption="cap")
        text = CrossEncoderReranker._artifact_text(a)
        assert len(text) == len("short cap")


# ---------------------------------------------------------------------------
# rerank
# ---------------------------------------------------------------------------

class TestRerank:
    def test_empty_artifacts_returns_empty(self, reranker):
        result = reranker.rerank(Query(text="x"), [])
        assert result == []

    def test_order_follows_scores(self, reranker, artifacts, mock_model):
        # Scores: a3=0.9, a1=0.5, a2=0.1  → expected order a3, a1, a2
        mock_model.predict.return_value = [0.5, 0.1, 0.9]
        reranker._model = mock_model

        results = reranker.rerank(Query(text="vision"), artifacts, top_k=3)
        assert [r.id for r in results] == ["a3", "a1", "a2"]

    def test_top_k_limits_results(self, reranker, artifacts, mock_model):
        mock_model.predict.return_value = [0.9, 0.8, 0.7]
        reranker._model = mock_model

        results = reranker.rerank(Query(text="x"), artifacts, top_k=2)
        assert len(results) == 2

    def test_top_k_larger_than_artifacts(self, reranker, artifacts, mock_model):
        mock_model.predict.return_value = [0.5, 0.6, 0.7]
        reranker._model = mock_model

        results = reranker.rerank(Query(text="x"), artifacts, top_k=100)
        assert len(results) == len(artifacts)

    def test_predict_called_with_pairs(self, reranker, artifacts, mock_model):
        mock_model.predict.return_value = [0.1, 0.2, 0.3]
        reranker._model = mock_model

        q = Query(text="my question")
        reranker.rerank(q, artifacts, top_k=3)

        call_args = mock_model.predict.call_args[0][0]
        assert len(call_args) == 3
        for query_text, passage in call_args:
            assert query_text == "my question"
            assert isinstance(passage, str)


# ---------------------------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------------------------

class TestLazyLoading:
    def test_model_is_none_initially(self):
        r = CrossEncoderReranker()
        assert r._model is None

    def test_model_loaded_on_first_rerank(self, artifacts):
        r = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        fake_model = MagicMock()
        fake_model.predict.return_value = [0.1, 0.2, 0.3]

        # CrossEncoder is a local import inside _get_model, so patch at source package level
        with patch("sentence_transformers.CrossEncoder", return_value=fake_model) as mock_ce:
            r.rerank(Query(text="x"), artifacts, top_k=1)
            mock_ce.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")
            assert r._model is fake_model

    def test_model_loaded_only_once(self, artifacts):
        r = CrossEncoderReranker()
        fake_model = MagicMock()
        fake_model.predict.return_value = [0.1, 0.2, 0.3]

        with patch("sentence_transformers.CrossEncoder", return_value=fake_model) as mock_ce:
            r.rerank(Query(text="x"), artifacts)
            r.rerank(Query(text="y"), artifacts)
            assert mock_ce.call_count == 1
