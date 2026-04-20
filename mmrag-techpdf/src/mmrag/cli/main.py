"""
mmrag CLI  –  entry point: ``mmrag <command> [options]``

Commands
--------
index    Index a PDF document.
query    Ask a question against indexed documents.
evaluate Run evaluation against a golden JSONL dataset.
serve    Start the FastAPI server.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

from ..config.logging import setup_logging
from ..config.settings import settings

app = typer.Typer(
    name="mmrag",
    help="Multimodal RAG for technical PDFs.",
    no_args_is_help=True,
)


def _build_components():
    """Assemble all adapters from settings. Shared by index / query / evaluate."""
    from ..adapters.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
    from ..adapters.figures.basic_figure_extractor import BasicFigureExtractor
    from ..adapters.figures.blip2_captioner import BLIPCaptioner
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

    return embedder, index_store, doc_store, bm25


def _load_indexes(
    index_store,
    doc_store,
    bm25,
    index_dir: str,
) -> None:
    idx = Path(index_dir)
    faiss_path = idx / "faiss.index"
    docstore_path = idx / "docstore.pkl"
    bm25_path = idx / "bm25.pkl"

    if not faiss_path.exists():
        typer.echo(
            f"[error] No FAISS index found at {faiss_path}. "
            "Run `mmrag index <pdf>` first.",
            err=True,
        )
        raise typer.Exit(code=1)

    index_store.load(str(idx))
    if docstore_path.exists():
        doc_store.load(str(docstore_path))
    if bm25_path.exists():
        bm25.load(str(bm25_path))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def index(
    pdf_path: str = typer.Argument(..., help="Path to the PDF file to index."),
    index_dir: str = typer.Option(
        str(settings.index_dir), "--index-dir", "-o", help="Directory to write indexes."
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level."),
):
    """Index a PDF document (extract text, figures, tables; embed; persist)."""
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    from ..adapters.figures.basic_figure_extractor import BasicFigureExtractor
    from ..adapters.figures.blip2_captioner import BLIPCaptioner
    from ..adapters.pdf.pymupdf_loader import PyMuPDFLoader
    from ..adapters.tables.camelot_extractor import PyMuPDFTableExtractor
    from ..application.indexing_pipeline import IndexingPipeline

    embedder, index_store, doc_store, bm25 = _build_components()

    pipeline = IndexingPipeline(
        pdf_loader=PyMuPDFLoader(),
        figure_extractor=BasicFigureExtractor(),
        table_extractor=PyMuPDFTableExtractor(),
        captioner=BLIPCaptioner(model_name=settings.captioner_model),
        embedder=embedder,
        index_store=index_store,
        doc_store=doc_store,
        bm25_retriever=bm25,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    artifacts = pipeline.run(pdf_path, index_dir)
    typer.echo(f"Indexed {len(artifacts)} artifacts → {index_dir}")


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask."),
    index_dir: str = typer.Option(
        str(settings.index_dir), "--index-dir", "-i", help="Directory with saved indexes."
    ),
    top_k: int = typer.Option(settings.rerank_top_k, "--top-k", help="Number of artifacts to use."),
    backend: str = typer.Option(
        settings.generator_backend, "--backend", help="Generator backend: openai | local"
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """Ask a question against indexed documents."""
    setup_logging(log_level)

    from ..adapters.rerank.cross_encoder_reranker import CrossEncoderReranker
    from ..adapters.retrieval.hybrid_retriever import HybridRetriever
    from ..application.query_pipeline import QueryPipeline

    embedder, index_store, doc_store, bm25 = _build_components()
    _load_indexes(index_store, doc_store, bm25, index_dir)

    retriever = HybridRetriever(
        embedder=embedder,
        index_store=index_store,
        doc_store=doc_store,
        bm25_retriever=bm25,
        dense_weight=settings.dense_weight,
        bm25_weight=settings.bm25_weight,
    )
    reranker = CrossEncoderReranker(model_name=settings.reranker_model)

    if backend == "openai":
        from ..adapters.llm.openai_generator import OpenAIGenerator
        generator = OpenAIGenerator(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
        )
    else:
        from ..adapters.llm.local_generator import LocalGenerator
        generator = LocalGenerator(
            model_name=settings.local_llm_model,
            max_new_tokens=settings.local_llm_max_new_tokens,
        )

    qp = QueryPipeline(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        retrieval_top_k=settings.retrieval_top_k,
        rerank_top_k=top_k,
    )

    answer = qp.run(question)
    typer.echo("\n" + "=" * 60)
    typer.echo(answer.text)
    typer.echo("=" * 60)
    typer.echo(f"\nSources ({len(answer.sources)}):")
    for src in answer.sources:
        typer.echo(f"  [{src.artifact_id[:8]}] {src.text[:120]}")


@app.command()
def evaluate(
    golden_path: str = typer.Argument(..., help="Path to golden JSONL dataset."),
    index_dir: str = typer.Option(str(settings.index_dir), "--index-dir", "-i"),
    output_path: Optional[str] = typer.Option(None, "--output", "-o", help="Path to write per-question results."),
    backend: str = typer.Option(settings.generator_backend, "--backend"),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """Evaluate the system against a golden question/answer dataset."""
    setup_logging(log_level)

    from ..adapters.rerank.cross_encoder_reranker import CrossEncoderReranker
    from ..adapters.retrieval.hybrid_retriever import HybridRetriever
    from ..application.evaluation_pipeline import EvaluationPipeline
    from ..application.query_pipeline import QueryPipeline

    embedder, index_store, doc_store, bm25 = _build_components()
    _load_indexes(index_store, doc_store, bm25, index_dir)

    retriever = HybridRetriever(
        embedder=embedder,
        index_store=index_store,
        doc_store=doc_store,
        bm25_retriever=bm25,
    )
    reranker = CrossEncoderReranker(model_name=settings.reranker_model)

    if backend == "openai":
        from ..adapters.llm.openai_generator import OpenAIGenerator
        generator = OpenAIGenerator(model=settings.openai_model, api_key=settings.openai_api_key)
    else:
        from ..adapters.llm.local_generator import LocalGenerator
        generator = LocalGenerator(model_name=settings.local_llm_model)

    qp = QueryPipeline(retriever=retriever, reranker=reranker, generator=generator)
    ep = EvaluationPipeline(query_pipeline=qp)

    metrics = ep.run(golden_path, output_path=output_path)
    typer.echo("\nEvaluation Results")
    typer.echo("=" * 40)
    for k, v in metrics.items():
        typer.echo(f"  {k:30s}: {v:.4f}" if isinstance(v, float) else f"  {k:30s}: {v}")


@app.command()
def serve(
    host: str = typer.Option(settings.api_host, "--host"),
    port: int = typer.Option(settings.api_port, "--port"),
    log_level: str = typer.Option("info", "--log-level"),
):
    """Start the FastAPI server."""
    import uvicorn
    uvicorn.run("mmrag.api.app:app", host=host, port=port, log_level=log_level, reload=False)


if __name__ == "__main__":
    app()
