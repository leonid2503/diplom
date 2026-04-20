"""
Indexing pipeline: PDF → artifacts → embeddings → persisted indexes.

Flow
----
1. Load PDF with PyMuPDFLoader.
2. Extract text blocks and chunk them.
3. Extract embedded images (figures) and save as PNG crops.
4. Caption each figure with BLIP.
5. Extract tables as Markdown strings.
6. Embed all artifact text representations with SentenceTransformer.
7. Add embeddings to FAISS index.
8. Build BM25 index from the same artifact texts.
9. Persist FAISS index, BM25 index, docstore, and artifacts.jsonl.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import List

from ..adapters.retrieval.bm25_retriever import BM25Retriever
from ..adapters.storage.local_docstore import LocalDocStore
from ..domain.artifact import Artifact, Provenance
from ..domain.document import Document
from ..ports.captioner import Captioner
from ..ports.embedder import Embedder
from ..ports.figure_extractor import FigureExtractor
from ..ports.index_store import IndexStore
from ..ports.pdf_loader import PDFLoader
from ..ports.table_extractor import TableExtractor

logger = logging.getLogger(__name__)


class IndexingPipeline:
    def __init__(
        self,
        pdf_loader: PDFLoader,
        figure_extractor: FigureExtractor,
        table_extractor: TableExtractor,
        captioner: Captioner,
        embedder: Embedder,
        index_store: IndexStore,
        doc_store: LocalDocStore,
        bm25_retriever: BM25Retriever,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
    ):
        self.pdf_loader = pdf_loader
        self.figure_extractor = figure_extractor
        self.table_extractor = table_extractor
        self.captioner = captioner
        self.embedder = embedder
        self.index_store = index_store
        self.doc_store = doc_store
        self.bm25_retriever = bm25_retriever
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, pdf_path: str, output_dir: str) -> List[Artifact]:
        """
        Index *pdf_path* and write all artefacts to *output_dir*.

        Returns the list of all indexed Artifact objects.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        logger.info("Loading PDF: %s", pdf_path)
        document = self.pdf_loader.load(pdf_path)

        # 1. Text
        logger.info("Extracting text chunks …")
        text_artifacts = self._extract_text_artifacts(document)
        logger.info("  → %d text chunks", len(text_artifacts))

        # 2. Figures
        logger.info("Extracting figures …")
        figure_artifacts = self.figure_extractor.extract(document, str(out))
        logger.info("  → %d figures", len(figure_artifacts))

        # 3. Caption figures
        logger.info("Captioning figures …")
        for art in figure_artifacts:
            if art.image_path:
                caption = self.captioner.caption(art.image_path)
                art.caption = caption
                art.content = caption  # embed the caption text
                logger.debug("  figure %s: %s", art.id[:8], caption[:80])

        # 4. Tables
        logger.info("Extracting tables …")
        table_artifacts = self.table_extractor.extract(pdf_path)
        logger.info("  → %d tables", len(table_artifacts))

        all_artifacts = text_artifacts + figure_artifacts + table_artifacts
        logger.info("Total artifacts: %d", len(all_artifacts))

        # 5. Populate docstore
        for art in all_artifacts:
            self.doc_store.put(art)

        # 6. Embed
        logger.info("Embedding artifacts …")
        texts = [art.content for art in all_artifacts]
        ids = [art.id for art in all_artifacts]
        embeddings = self.embedder.encode(texts)

        # 7. FAISS
        self.index_store.add(embeddings, ids)

        # 8. BM25
        logger.info("Building BM25 index …")
        self.bm25_retriever.build(all_artifacts)

        # 9. Persist
        logger.info("Saving indexes to %s …", out)
        self.index_store.save(str(out))
        self.doc_store.save(str(out / "docstore.pkl"))
        self.bm25_retriever.save(str(out / "bm25.pkl"))
        self._save_artifacts_jsonl(all_artifacts, str(out / "artifacts.jsonl"))

        logger.info("Indexing complete.")
        return all_artifacts

    # ------------------------------------------------------------------
    # Text chunking
    # ------------------------------------------------------------------

    def _extract_text_artifacts(self, document: Document) -> List[Artifact]:
        artifacts: List[Artifact] = []
        for page in document.pages:
            page_text = "\n".join(
                b.text for b in page.blocks if b.block_type == "text" and b.text.strip()
            )
            if not page_text.strip():
                continue

            chunks = self._chunk_text(page_text)
            for idx, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                artifacts.append(
                    Artifact(
                        id=str(uuid.uuid4()),
                        artifact_type="text",
                        content=chunk,
                        provenance=Provenance(
                            document_path=document.file_path,
                            page_number=page.number,
                            bbox=(0.0, 0.0, 0.0, 0.0),
                        ),
                    )
                )
        return artifacts

    def _chunk_text(self, text: str) -> List[str]:
        """Split *text* into overlapping word-count chunks."""
        words = text.split()
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += self.chunk_size - self.chunk_overlap
        return chunks

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_artifacts_jsonl(artifacts: List[Artifact], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for art in artifacts:
                f.write(art.model_dump_json() + "\n")
