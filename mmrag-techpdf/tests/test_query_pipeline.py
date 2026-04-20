"""Tests for QueryPipeline: retrieve → rerank → generate flow."""
from unittest.mock import MagicMock

import pytest

from mmrag.application.query_pipeline import QueryPipeline
from mmrag.domain.artifact import Evidence
from mmrag.domain.query import Answer, Query, QueryIntent

from conftest import make_artifact


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def artifacts():
    return [make_artifact(id=f"a{i}", content=f"content {i}") for i in range(5)]


def _make_pipeline(retrieved=None, reranked=None, answer=None):
    retrieved = retrieved or []
    reranked = reranked or []
    answer = answer or Answer(text="The answer.", confidence=0.9)

    retriever = MagicMock()
    retriever.retrieve.return_value = retrieved

    reranker = MagicMock()
    reranker.rerank.return_value = reranked

    generator = MagicMock()
    generator.generate.return_value = answer

    return QueryPipeline(retriever=retriever, reranker=reranker, generator=generator), retriever, reranker, generator


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestRun:
    def test_returns_answer(self, artifacts):
        expected = Answer(text="42", confidence=0.99)
        pipeline, *_ = _make_pipeline(retrieved=artifacts, reranked=artifacts[:3], answer=expected)
        result = pipeline.run("What is the answer?")
        assert result.text == "42"

    def test_retriever_called_with_query(self, artifacts):
        pipeline, retriever, *_ = _make_pipeline(retrieved=artifacts)
        pipeline.run("my question")
        retriever.retrieve.assert_called_once()
        call_query = retriever.retrieve.call_args[0][0]
        assert call_query.text == "my question"

    def test_retriever_top_k_passed(self, artifacts):
        pipeline, retriever, *_ = _make_pipeline(retrieved=artifacts)
        pipeline.retrieval_top_k = 15
        pipeline.run("q")
        _, kwargs = retriever.retrieve.call_args
        assert kwargs.get("top_k", retriever.retrieve.call_args[0][1] if len(retriever.retrieve.call_args[0]) > 1 else None) == 15

    def test_reranker_receives_retrieved_artifacts(self, artifacts):
        pipeline, _, reranker, _ = _make_pipeline(retrieved=artifacts)
        pipeline.run("x")
        call_arts = reranker.rerank.call_args[0][1]
        assert call_arts == artifacts

    def test_reranker_top_k_passed(self, artifacts):
        pipeline, _, reranker, _ = _make_pipeline(retrieved=artifacts)
        pipeline.rerank_top_k = 7
        pipeline.run("q")
        _, kwargs = reranker.rerank.call_args
        assert kwargs.get("top_k", reranker.rerank.call_args[0][2] if len(reranker.rerank.call_args[0]) > 2 else None) == 7

    def test_generator_receives_reranked_artifacts(self, artifacts):
        top3 = artifacts[:3]
        pipeline, _, _, generator = _make_pipeline(retrieved=artifacts, reranked=top3)
        pipeline.run("x")
        call_arts = generator.generate.call_args[0][1]
        assert call_arts == top3

    def test_query_intent_forwarded(self, artifacts):
        pipeline, retriever, *_ = _make_pipeline(retrieved=artifacts)
        pipeline.run("summarize this", intent=QueryIntent.SUMMARIZE)
        q = retriever.retrieve.call_args[0][0]
        assert q.intent == QueryIntent.SUMMARIZE

    def test_filters_forwarded(self, artifacts):
        pipeline, retriever, *_ = _make_pipeline(retrieved=artifacts)
        pipeline.run("x", filters={"document": "doc.pdf"})
        q = retriever.retrieve.call_args[0][0]
        assert q.filters == {"document": "doc.pdf"}

    def test_default_filters_empty(self, artifacts):
        pipeline, retriever, *_ = _make_pipeline(retrieved=artifacts)
        pipeline.run("x")
        q = retriever.retrieve.call_args[0][0]
        assert q.filters == {}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_retrieval_still_calls_generator(self):
        pipeline, _, _, generator = _make_pipeline(retrieved=[], reranked=[])
        pipeline.run("x")
        generator.generate.assert_called_once()

    def test_empty_reranking_still_calls_generator(self, artifacts):
        pipeline, _, _, generator = _make_pipeline(retrieved=artifacts, reranked=[])
        pipeline.run("x")
        generator.generate.assert_called_once()

    def test_answer_with_sources_returned_as_is(self):
        ev = Evidence(artifact_id="a1", text="evidence", confidence=0.8)
        ans = Answer(text="ans", confidence=0.7, sources=[ev])
        pipeline, *_ = _make_pipeline(answer=ans)
        result = pipeline.run("q")
        assert len(result.sources) == 1

    def test_default_intent_is_qa(self, artifacts):
        pipeline, retriever, *_ = _make_pipeline(retrieved=artifacts)
        pipeline.run("question")
        q = retriever.retrieve.call_args[0][0]
        assert q.intent == QueryIntent.QA
