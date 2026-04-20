"""Shared pytest fixtures for mmrag-techpdf tests."""
import numpy as np
import pytest

from mmrag.domain.artifact import Artifact, Evidence, Provenance
from mmrag.domain.document import Block, Document, Page
from mmrag.domain.query import Answer, Query, QueryIntent


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_provenance(doc_path="doc.pdf", page=1, bbox=(0.0, 0.0, 100.0, 50.0)):
    return Provenance(document_path=doc_path, page_number=page, bbox=bbox)


def make_artifact(
    id="art-1",
    artifact_type="text",
    content="sample content",
    caption=None,
    image_path=None,
    confidence=None,
    page=1,
):
    return Artifact(
        id=id,
        artifact_type=artifact_type,
        content=content,
        caption=caption,
        image_path=image_path,
        confidence=confidence,
        provenance=make_provenance(page=page),
    )


def make_block(text="hello world", block_type="text", bbox=(0.0, 0.0, 100.0, 20.0)):
    return Block(text=text, bbox=bbox, block_type=block_type)


def make_document(title="Test Doc", pages=None, file_path="test.pdf"):
    pages = pages or [Page(number=1, blocks=[make_block()])]
    return Document(title=title, pages=pages, file_path=file_path)


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_artifact():
    return make_artifact()


@pytest.fixture
def sample_artifacts():
    return [
        make_artifact(id="art-1", content="neural networks and deep learning"),
        make_artifact(id="art-2", content="transformer architecture attention mechanism"),
        make_artifact(id="art-3", content="convolutional neural network image classification"),
    ]


@pytest.fixture
def sample_query():
    return Query(text="what is a neural network?")


@pytest.fixture
def dim():
    return 32


@pytest.fixture
def random_embeddings(dim):
    rng = np.random.default_rng(42)
    return rng.random((3, dim)).astype(np.float32)
