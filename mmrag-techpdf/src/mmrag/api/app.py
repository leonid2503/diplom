"""
FastAPI application exposing the mmrag system over HTTP.

Endpoints
---------
POST /index          – Index a PDF file.
POST /query          – Ask a question against indexed documents.
GET  /artifacts      – List all indexed artifacts (id, type, page).
GET  /health         – Health check.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..application.indexing_pipeline import IndexingPipeline
from ..application.query_pipeline import QueryPipeline
from ..config.logging import setup_logging
from ..config.settings import settings

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multimodal RAG for Technical PDFs",
    description="Index technical papers and ask questions over text, figures, and tables.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Dependency injection – assembled once at startup
# ---------------------------------------------------------------------------

_indexing_pipeline: Optional[IndexingPipeline] = None
_query_pipeline: Optional[QueryPipeline] = None


def _build_pipelines() -> tuple[IndexingPipeline, QueryPipeline]:
    from ..adapters.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
    from ..adapters.figures.basic_figure_extractor import BasicFigureExtractor
    from ..adapters.figures.blip2_captioner import BLIPCaptioner
    from ..adapters.llm.local_generator import LocalGenerator
    from ..adapters.llm.openai_generator import OpenAIGenerator
    from ..adapters.pdf.pymupdf_loader import PyMuPDFLoader
    from ..adapters.rerank.cross_encoder_reranker import CrossEncoderReranker
    from ..adapters.retrieval.bm25_retriever import BM25Retriever
    from ..adapters.retrieval.hybrid_retriever import HybridRetriever
    from ..adapters.storage.faiss_store import FAISSStore
    from ..adapters.storage.local_docstore import LocalDocStore
    from ..adapters.tables.camelot_extractor import PyMuPDFTableExtractor

    embedder = SentenceTransformerEmbedder(model_name=settings.embedder_model)
    index_store = FAISSStore(dimension=embedder.dimension)
    doc_store = LocalDocStore()
    bm25 = BM25Retriever()

    # Try to load existing indexes
    idx_dir = settings.index_dir
    docstore_path = idx_dir / "docstore.pkl"
    bm25_path = idx_dir / "bm25.pkl"
    if (idx_dir / "faiss.index").exists():
        logger.info("Loading existing FAISS index from %s", idx_dir)
        index_store.load(str(idx_dir))
    if docstore_path.exists():
        doc_store.load(str(docstore_path))
    if bm25_path.exists():
        bm25.load(str(bm25_path))

    captioner = BLIPCaptioner(model_name=settings.captioner_model)
    table_extractor = PyMuPDFTableExtractor()
    figure_extractor = BasicFigureExtractor()

    indexing = IndexingPipeline(
        pdf_loader=PyMuPDFLoader(),
        figure_extractor=figure_extractor,
        table_extractor=table_extractor,
        captioner=captioner,
        embedder=embedder,
        index_store=index_store,
        doc_store=doc_store,
        bm25_retriever=bm25,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    retriever = HybridRetriever(
        embedder=embedder,
        index_store=index_store,
        doc_store=doc_store,
        bm25_retriever=bm25,
        dense_weight=settings.dense_weight,
        bm25_weight=settings.bm25_weight,
    )
    reranker = CrossEncoderReranker(model_name=settings.reranker_model)

    if settings.generator_backend == "openai":
        generator = OpenAIGenerator(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
        )
    else:
        generator = LocalGenerator(
            model_name=settings.local_llm_model,
            max_new_tokens=settings.local_llm_max_new_tokens,
        )

    querying = QueryPipeline(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        retrieval_top_k=settings.retrieval_top_k,
        rerank_top_k=settings.rerank_top_k,
    )

    return indexing, querying


@app.on_event("startup")
async def startup():
    global _indexing_pipeline, _query_pipeline
    _indexing_pipeline, _query_pipeline = _build_pipelines()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class ArtifactSummary(BaseModel):
    id: str
    artifact_type: str
    page_number: int
    content_preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[ArtifactSummary]


class IndexResponse(BaseModel):
    indexed_artifacts: int
    index_dir: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/index", response_model=IndexResponse)
async def index_pdf(
    file: UploadFile = File(...),
    output_dir: str = Form(default=str(settings.index_dir)),
):
    """Upload and index a PDF document."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save upload to temp path
    tmp_path = Path(output_dir) / "uploads" / file.filename
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(await file.read())

    try:
        artifacts = _indexing_pipeline.run(str(tmp_path), output_dir)
    except Exception as exc:
        logger.exception("Indexing failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return IndexResponse(indexed_artifacts=len(artifacts), index_dir=output_dir)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Ask a question against indexed documents."""
    try:
        answer = _query_pipeline.run(req.question)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))

    sources = [
        ArtifactSummary(
            id=s.artifact_id,
            artifact_type="unknown",
            page_number=0,
            content_preview=s.text[:200],
        )
        for s in answer.sources
    ]

    return QueryResponse(answer=answer.text, sources=sources)


@app.get("/artifacts", response_model=List[ArtifactSummary])
def list_artifacts():
    """List all artifacts in the docstore."""
    all_arts = _indexing_pipeline.doc_store.get_all()
    return [
        ArtifactSummary(
            id=a.id,
            artifact_type=a.artifact_type,
            page_number=a.provenance.page_number,
            content_preview=a.content[:200],
        )
        for a in all_arts
    ]
