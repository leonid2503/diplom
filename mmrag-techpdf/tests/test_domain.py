"""Tests for domain models: Block, Page, Document, Artifact, Evidence, Query, Answer."""
import pytest
from pydantic import ValidationError

from mmrag.domain.artifact import Artifact, Evidence, Provenance
from mmrag.domain.document import Block, Document, Page
from mmrag.domain.query import Answer, Query, QueryIntent

from conftest import make_artifact, make_block, make_provenance


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

class TestBlock:
    def test_defaults(self):
        b = Block(bbox=(0, 0, 1, 1), block_type="text")
        assert b.text == ""
        assert b.confidence is None

    def test_custom_values(self):
        b = make_block(text="hello", block_type="figure")
        assert b.text == "hello"
        assert b.block_type == "figure"

    def test_bbox_stored_correctly(self):
        b = Block(bbox=(1.5, 2.5, 3.5, 4.5), block_type="text")
        assert b.bbox == (1.5, 2.5, 3.5, 4.5)

    def test_missing_bbox_raises(self):
        with pytest.raises(ValidationError):
            Block(block_type="text")

    def test_missing_block_type_raises(self):
        with pytest.raises(ValidationError):
            Block(bbox=(0, 0, 1, 1))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

class TestPage:
    def test_page_number_must_be_positive(self):
        with pytest.raises(ValidationError):
            Page(number=0)

    def test_valid_page(self):
        p = Page(number=1)
        assert p.blocks == []
        assert p.image_path is None

    def test_page_with_blocks(self):
        p = Page(number=2, blocks=[make_block(), make_block(text="second")])
        assert len(p.blocks) == 2


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

class TestDocument:
    def test_empty_document(self):
        doc = Document(title="Empty", file_path="empty.pdf")
        assert doc.pages == []
        assert doc.metadata == {}

    def test_document_with_pages(self):
        page = Page(number=1, blocks=[make_block()])
        doc = Document(title="Doc", file_path="doc.pdf", pages=[page])
        assert len(doc.pages) == 1

    def test_metadata_stored(self):
        doc = Document(title="Doc", file_path="doc.pdf", metadata={"author": "Alice"})
        assert doc.metadata["author"] == "Alice"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_valid(self):
        p = make_provenance(page=3)
        assert p.page_number == 3

    def test_page_number_must_be_positive(self):
        with pytest.raises(ValidationError):
            Provenance(document_path="x.pdf", page_number=0, bbox=(0, 0, 1, 1))

    def test_bbox_stored(self):
        p = make_provenance(bbox=(10.0, 20.0, 30.0, 40.0))
        assert p.bbox == (10.0, 20.0, 30.0, 40.0)


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------

class TestArtifact:
    def test_defaults(self):
        a = make_artifact()
        assert a.image_path is None
        assert a.caption is None
        assert a.confidence is None

    def test_all_fields(self):
        a = make_artifact(
            id="x",
            artifact_type="figure",
            content="a chart",
            caption="Figure 1",
            image_path="/tmp/fig.png",
            confidence=0.9,
        )
        assert a.id == "x"
        assert a.artifact_type == "figure"
        assert a.caption == "Figure 1"
        assert a.image_path == "/tmp/fig.png"
        assert a.confidence == 0.9

    def test_missing_required_fields_raises(self):
        with pytest.raises(ValidationError):
            Artifact(artifact_type="text", content="x")  # missing id and provenance


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

class TestEvidence:
    def test_valid(self):
        e = Evidence(artifact_id="art-1", text="relevant text", confidence=0.85)
        assert e.artifact_id == "art-1"
        assert e.confidence == 0.85

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            Evidence(artifact_id="a", text="t", confidence=1.5)
        with pytest.raises(ValidationError):
            Evidence(artifact_id="a", text="t", confidence=-0.1)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class TestQuery:
    def test_default_intent(self):
        q = Query(text="what is AI?")
        assert q.intent == QueryIntent.QA
        assert q.filters == {}

    def test_explicit_intent(self):
        q = Query(text="summarize this", intent=QueryIntent.SUMMARIZE)
        assert q.intent == QueryIntent.SUMMARIZE

    def test_filters(self):
        q = Query(text="find X", filters={"document": "doc.pdf"})
        assert q.filters["document"] == "doc.pdf"

    def test_all_intents_valid(self):
        for intent in QueryIntent:
            q = Query(text="test", intent=intent)
            assert q.intent == intent


# ---------------------------------------------------------------------------
# Answer
# ---------------------------------------------------------------------------

class TestAnswer:
    def test_valid_answer(self):
        a = Answer(text="The answer is 42.", confidence=0.95)
        assert a.text == "The answer is 42."
        assert a.sources == []
        assert a.artifacts == []

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            Answer(text="x", confidence=1.1)
        with pytest.raises(ValidationError):
            Answer(text="x", confidence=-0.01)

    def test_with_sources_and_artifacts(self):
        ev = Evidence(artifact_id="a1", text="evidence", confidence=0.7)
        ans = Answer(text="answer", confidence=0.8, sources=[ev], artifacts=["a1"])
        assert len(ans.sources) == 1
        assert "a1" in ans.artifacts
