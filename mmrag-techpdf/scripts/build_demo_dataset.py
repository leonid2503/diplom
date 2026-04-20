"""
build_demo_dataset.py
~~~~~~~~~~~~~~~~~~~~~
Downloads a small set of open-access technical PDFs from arXiv and
indexes them, producing a ready-to-use demo dataset.

Usage
-----
    python scripts/build_demo_dataset.py [--output-dir data/indexes]

The script will:
  1. Download up to MAX_PAPERS arXiv papers (PDF).
  2. Run the full indexing pipeline on each.
  3. Write a sample golden QA JSONL to tests/golden/questions_ua.jsonl.

Requirements: requests, tqdm  (pip install requests tqdm)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Papers to download  (title, arXiv PDF URL, sample QA pairs)
# ---------------------------------------------------------------------------

PAPERS = [
    {
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/pdf/1706.03762",
        "filename": "attention_is_all_you_need.pdf",
        "qa": [
            {
                "question": "What is the main contribution of the Transformer architecture?",
                "expected_answer": "The Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolutions.",
            },
            {
                "question": "How many attention heads do the encoder and decoder use in the base model?",
                "expected_answer": "8",
            },
            {
                "question": "What optimizer is used to train the Transformer?",
                "expected_answer": "Adam optimizer",
            },
        ],
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "url": "https://arxiv.org/pdf/1810.04805",
        "filename": "bert.pdf",
        "qa": [
            {
                "question": "What does BERT stand for?",
                "expected_answer": "Bidirectional Encoder Representations from Transformers",
            },
            {
                "question": "What are the two pre-training tasks used in BERT?",
                "expected_answer": "Masked Language Model (MLM) and Next Sentence Prediction (NSP)",
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_pdf(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  [skip] {dest.name} already exists.")
        return
    print(f"  Downloading {dest.name} …")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved to {dest}")
    except Exception as exc:
        print(f"  [warn] Download failed: {exc}", file=sys.stderr)


def write_golden(qa_pairs: list[dict], golden_path: Path) -> None:
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if golden_path.exists():
        with open(golden_path, encoding="utf-8") as f:
            existing = [json.loads(l) for l in f if l.strip()]

    existing_qs = {r["question"] for r in existing}
    new_pairs = [qa for qa in qa_pairs if qa["question"] not in existing_qs]
    if new_pairs:
        with open(golden_path, "a", encoding="utf-8") as f:
            for qa in new_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
        print(f"  Added {len(new_pairs)} QA pairs to {golden_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build demo dataset for mmrag.")
    parser.add_argument(
        "--output-dir",
        default="data/indexes",
        help="Directory where indexes will be written.",
    )
    parser.add_argument(
        "--pdf-dir",
        default="data/pdfs",
        help="Directory to save downloaded PDFs.",
    )
    parser.add_argument(
        "--golden",
        default="tests/golden/questions_ua.jsonl",
        help="Path to the golden QA JSONL file.",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Only download PDFs and write golden data, skip indexing.",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    golden_path = Path(args.golden)

    all_qa: list[dict] = []

    for paper in PAPERS:
        print(f"\n── {paper['title']} ──")
        pdf_dest = pdf_dir / paper["filename"]
        download_pdf(paper["url"], pdf_dest)
        all_qa.extend(paper["qa"])

        if not args.skip_index and pdf_dest.exists():
            print(f"  Indexing …")
            _run_indexing(str(pdf_dest), args.output_dir)

    write_golden(all_qa, golden_path)
    print("\nDone.")


def _run_indexing(pdf_path: str, output_dir: str) -> None:
    """Run the indexing pipeline programmatically."""
    try:
        from mmrag.adapters.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
        from mmrag.adapters.figures.basic_figure_extractor import BasicFigureExtractor
        from mmrag.adapters.figures.blip2_captioner import BLIPCaptioner
        from mmrag.adapters.pdf.pymupdf_loader import PyMuPDFLoader
        from mmrag.adapters.retrieval.bm25_retriever import BM25Retriever
        from mmrag.adapters.storage.faiss_store import FAISSStore
        from mmrag.adapters.storage.local_docstore import LocalDocStore
        from mmrag.adapters.tables.camelot_extractor import PyMuPDFTableExtractor
        from mmrag.application.indexing_pipeline import IndexingPipeline
        from mmrag.config.settings import settings

        embedder = SentenceTransformerEmbedder(model_name=settings.embedder_model)
        index_store = FAISSStore(dimension=embedder.dimension)
        doc_store = LocalDocStore()
        bm25 = BM25Retriever()

        pipeline = IndexingPipeline(
            pdf_loader=PyMuPDFLoader(),
            figure_extractor=BasicFigureExtractor(),
            table_extractor=PyMuPDFTableExtractor(),
            captioner=BLIPCaptioner(model_name=settings.captioner_model),
            embedder=embedder,
            index_store=index_store,
            doc_store=doc_store,
            bm25_retriever=bm25,
        )
        artifacts = pipeline.run(pdf_path, output_dir)
        print(f"  Indexed {len(artifacts)} artifacts.")
    except ImportError as exc:
        print(f"  [error] mmrag not installed: {exc}", file=sys.stderr)
        print("  Run: pip install -e mmrag-techpdf/", file=sys.stderr)


if __name__ == "__main__":
    main()
