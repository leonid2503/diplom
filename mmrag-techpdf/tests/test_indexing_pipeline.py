"""Tests for IndexingPipeline: text chunking, artifact extraction, and full run."""
from unittest.mock import MagicMock, call, patch
from pathlib import Path

import numpy as np
import pytest

from mmrag.application.indexing_pipeline import IndexingPipeline
from mmrag.domain.artifact import Artifact
from mmrag.domain.document import Block, Document, Page

from conftest import make_artifact, make_block


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(**kwargs):
    defaults = dict(
        pdf_loader=MagicMock(),
        figure_extractor=MagicMock(return_value=[]),
        table_extractor=MagicMock(return_value=[]),
        captioner=MagicMock(return_value="a caption"),
        embedder=MagicMock(),
        index_store=MagicMock(),
        doc_store=MagicMock(),
        bm25_retriever=MagicMock(),
    )
    # figure_extractor.extract() and table_extractor.extract() need to be on the mock itself
    for key in ("figure_extractor", "table_extractor"):
        if key not in kwargs:
            m = MagicMock()
            m.extract.return_value = []
            defaults[key] = m
    defaults.update(kwargs)
    return IndexingPipeline(**defaults)


def _make_document_with_text(pages_text):
    """Build a Document where each element of pages_text is a page's text content."""
    pages = []
    for i, text in enumerate(pages_text, start=1):
        block = Block(text=text, bbox=(0, 0, 100, 100), block_type="text")
        pages.append(Page(number=i, blocks=[block]))
    return Document(title="Test", file_path="test.pdf", pages=pages)


# ---------------------------------------------------------------------------
# _chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    @pytest.fixture
    def pipeline(self):
        return _make_pipeline()

    def test_short_text_single_chunk(self, pipeline):
        words = ["word"] * 10
        chunks = pipeline._chunk_text(" ".join(words))
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self, pipeline):
        words = ["word"] * 500
        chunks = pipeline._chunk_text(" ".join(words))
        assert len(chunks) > 1

    def test_chunk_size_respected(self, pipeline):
        words = ["w"] * 1000
        chunks = pipeline._chunk_text(" ".join(words))
        for chunk in chunks:
            assert len(chunk.split()) <= pipeline.chunk_size

    def test_overlap_present(self):
        p = _make_pipeline()
        p.chunk_size = 10
        p.chunk_overlap = 3
        words = [f"w{i}" for i in range(20)]
        chunks = p._chunk_text(" ".join(words))
        # chunk[0] ends at w9; chunk[1] starts at w7 (10-3=7)
        assert chunks[1].split()[0] == "w7"

    def test_empty_text_no_chunks(self, pipeline):
        assert pipeline._chunk_text("") == []

    def test_exact_chunk_size_no_overlap_needed(self, pipeline):
        words = ["w"] * pipeline.chunk_size
        chunks = pipeline._chunk_text(" ".join(words))
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# _extract_text_artifacts
# ---------------------------------------------------------------------------

class TestExtractTextArtifacts:
    @pytest.fixture
    def pipeline(self):
        return _make_pipeline()

    def test_produces_artifacts_for_text_blocks(self, pipeline):
        doc = _make_document_with_text(["word " * 10])
        artifacts = pipeline._extract_text_artifacts(doc)
        assert len(artifacts) >= 1
        assert all(a.artifact_type == "text" for a in artifacts)

    def test_skips_non_text_blocks(self, pipeline):
        page = Page(number=1, blocks=[
            Block(text="", bbox=(0, 0, 100, 50), block_type="figure"),
            Block(text="real text here", bbox=(0, 50, 100, 100), block_type="text"),
        ])
        doc = Document(title="T", file_path="f.pdf", pages=[page])
        artifacts = pipeline._extract_text_artifacts(doc)
        assert len(artifacts) == 1

    def test_skips_empty_text_blocks(self, pipeline):
        page = Page(number=1, blocks=[
            Block(text="   ", bbox=(0, 0, 100, 50), block_type="text"),
        ])
        doc = Document(title="T", file_path="f.pdf", pages=[page])
        artifacts = pipeline._extract_text_artifacts(doc)
        assert len(artifacts) == 0

    def test_provenance_page_number(self, pipeline):
        doc = _make_document_with_text(["first page content"])
        artifacts = pipeline._extract_text_artifacts(doc)
        assert artifacts[0].provenance.page_number == 1

    def test_provenance_document_path(self, pipeline):
        doc = _make_document_with_text(["some text"])
        doc.file_path = "my/doc.pdf"
        artifacts = pipeline._extract_text_artifacts(doc)
        assert artifacts[0].provenance.document_path == "my/doc.pdf"

    def test_each_artifact_has_unique_id(self, pipeline):
        # Two pages, each producing at least one chunk
        doc = _make_document_with_text(["page one " * 5, "page two " * 5])
        artifacts = pipeline._extract_text_artifacts(doc)
        ids = [a.id for a in artifacts]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# run (integration with mocked dependencies)
# ---------------------------------------------------------------------------

class TestRun:
    def _make_full_pipeline(self, doc, text_artifacts=None, figure_artifacts=None, table_artifacts=None):
        """Build a pipeline with all heavy deps mocked.

        If text_artifacts is provided, _extract_text_artifacts is patched to
        return that list instead of parsing the document (which lets tests
        control exactly which text artifacts flow through run()).
        """
        figure_artifacts = figure_artifacts or []
        table_artifacts = table_artifacts or []

        pdf_loader = MagicMock()
        pdf_loader.load.return_value = doc

        figure_extractor = MagicMock()
        figure_extractor.extract.return_value = figure_artifacts

        table_extractor = MagicMock()
        table_extractor.extract.return_value = table_artifacts

        captioner = MagicMock()
        captioner.caption.return_value = "auto caption"

        # Dynamic embedding size so it matches however many artifacts run() produces
        embedder = MagicMock()
        embedder.encode.side_effect = lambda texts: np.ones((len(texts), 32), dtype=np.float32)

        index_store = MagicMock()
        doc_store = MagicMock()
        bm25_retriever = MagicMock()

        pipeline = IndexingPipeline(
            pdf_loader=pdf_loader,
            figure_extractor=figure_extractor,
            table_extractor=table_extractor,
            captioner=captioner,
            embedder=embedder,
            index_store=index_store,
            doc_store=doc_store,
            bm25_retriever=bm25_retriever,
        )

        # Patch text extraction when caller wants to inject specific artifacts
        if text_artifacts is not None:
            pipeline._extract_text_artifacts = MagicMock(return_value=text_artifacts)

        return pipeline, doc_store, index_store, bm25_retriever

    def test_run_returns_all_artifacts(self, tmp_path):
        doc = _make_document_with_text(["hello world"] * 2)
        t_art = make_artifact(id="t1", content="text")
        f_art = make_artifact(id="f1", artifact_type="figure", content="fig", image_path="/img.png")
        tbl_art = make_artifact(id="tbl1", artifact_type="table", content="table md")

        pipeline, doc_store, index_store, bm25 = self._make_full_pipeline(
            doc, text_artifacts=[t_art], figure_artifacts=[f_art], table_artifacts=[tbl_art]
        )
        result = pipeline.run("test.pdf", str(tmp_path))
        assert {a.id for a in result} >= {"f1", "tbl1"}

    def test_run_calls_doc_store_put_for_each_artifact(self, tmp_path):
        doc = _make_document_with_text([])
        t_art = make_artifact(id="t1")
        pipeline, doc_store, *_ = self._make_full_pipeline(doc, text_artifacts=[t_art])
        pipeline.run("test.pdf", str(tmp_path))
        doc_store.put.assert_called()

    def test_run_calls_bm25_build(self, tmp_path):
        doc = _make_document_with_text([])
        pipeline, _, _, bm25 = self._make_full_pipeline(doc)
        pipeline.run("test.pdf", str(tmp_path))
        bm25.build.assert_called_once()

    def test_run_calls_index_store_add(self, tmp_path):
        doc = _make_document_with_text([])
        pipeline, _, index_store, _ = self._make_full_pipeline(doc)
        pipeline.run("test.pdf", str(tmp_path))
        index_store.add.assert_called_once()

    def test_run_saves_artifacts_jsonl(self, tmp_path):
        doc = _make_document_with_text([])
        t_art = make_artifact(id="t1")
        pipeline, *_ = self._make_full_pipeline(doc, text_artifacts=[t_art])
        pipeline.run("test.pdf", str(tmp_path))
        jsonl = tmp_path / "artifacts.jsonl"
        assert jsonl.exists()
        lines = jsonl.read_text().strip().splitlines()
        assert len(lines) >= 1

    def test_figure_captioner_called_for_figures_with_image(self, tmp_path):
        doc = _make_document_with_text([])
        fig = make_artifact(id="f1", artifact_type="figure", content="", image_path="/img.png")
        pipeline, _, _, _ = self._make_full_pipeline(doc, figure_artifacts=[fig])
        pipeline.captioner.caption.return_value = "auto caption"
        pipeline.run("test.pdf", str(tmp_path))
        pipeline.captioner.caption.assert_called_once_with("/img.png")

    def test_figure_without_image_path_not_captioned(self, tmp_path):
        doc = _make_document_with_text([])
        fig = make_artifact(id="f1", artifact_type="figure", content="fig", image_path=None)
        pipeline, _, _, _ = self._make_full_pipeline(doc, figure_artifacts=[fig])
        pipeline.run("test.pdf", str(tmp_path))
        pipeline.captioner.caption.assert_not_called()

    def test_run_creates_output_dir(self, tmp_path):
        out = tmp_path / "nested" / "output"
        doc = _make_document_with_text([])
        pipeline, *_ = self._make_full_pipeline(doc)
        pipeline.run("test.pdf", str(out))
        assert out.exists()
