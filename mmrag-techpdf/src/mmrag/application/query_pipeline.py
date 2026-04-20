"""
Query pipeline: question → retrieved context → generated answer.

Flow
----
1. Embed the query.
2. Hybrid retrieve (FAISS + BM25) a large candidate pool.
3. Cross-encoder rerank down to top_k artifacts.
4. Generate an answer (OpenAI vision or local LLM).
"""

import logging
from typing import List, Optional

from ..domain.artifact import Artifact
from ..domain.query import Answer, Query, QueryIntent
from ..ports.generator import Generator
from ..ports.reranker import Reranker
from ..ports.retriever import Retriever

logger = logging.getLogger(__name__)


class QueryPipeline:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        generator: Generator,
        retrieval_top_k: int = 20,
        rerank_top_k: int = 5,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k

    def run(
        self,
        question: str,
        intent: QueryIntent = QueryIntent.QA,
        filters: Optional[dict] = None,
    ) -> Answer:
        """
        End-to-end query: retrieve, rerank, generate.

        Parameters
        ----------
        question:  Natural language question.
        intent:    QueryIntent (QA, SEARCH, SUMMARIZE, EXTRACT).
        filters:   Optional metadata filters forwarded to the retriever.

        Returns
        -------
        Answer object with text, sources, and artifact IDs.
        """
        query = Query(text=question, intent=intent, filters=filters or {})

        logger.info("Retrieving candidates for: %s", question[:80])
        candidates: List[Artifact] = self.retriever.retrieve(
            query, top_k=self.retrieval_top_k
        )
        logger.info("  retrieved %d candidates", len(candidates))

        logger.info("Reranking …")
        top_artifacts = self.reranker.rerank(
            query, candidates, top_k=self.rerank_top_k
        )
        logger.info("  top %d artifacts after reranking", len(top_artifacts))

        for art in top_artifacts:
            logger.debug(
                "  [%s] page=%d  %.60s",
                art.artifact_type,
                art.provenance.page_number,
                art.content,
            )

        logger.info("Generating answer …")
        answer = self.generator.generate(query, top_artifacts)

        logger.info("Done.  Answer length: %d chars", len(answer.text))
        return answer
