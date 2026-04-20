import logging
from typing import List, Optional

from ...domain.artifact import Artifact, Evidence
from ...domain.query import Answer, Query
from ...ports.generator import Generator

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are an expert assistant for technical and scientific papers.
Answer the question based ONLY on the context provided below.
If the answer is not in the context, say "I don't know".

### Context
{context}

### Question
{question}

### Answer
"""


class LocalGenerator(Generator):
    """
    Answer generator backed by a local HuggingFace causal language model.

    The model is loaded lazily on first call via the ``transformers``
    text-generation pipeline.

    Recommended models (CPU-friendly)
    ----------------------------------
    * ``microsoft/phi-2``              – 2.7 B params, strong reasoning
    * ``Qwen/Qwen2.5-0.5B-Instruct``  – 0.5 B params, very fast on CPU
    * ``TinyLlama/TinyLlama-1.1B-Chat-v1.0``

    For GPU use any 7 B instruction model (mistral, llama3, etc.).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device  # None → auto-detect
        self._pipe = None  # lazy

    def _get_pipe(self):
        if self._pipe is None:
            from transformers import pipeline

            kwargs: dict = dict(
                task="text-generation",
                model=self.model_name,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                return_full_text=False,
            )
            if self.device is not None:
                kwargs["device"] = self.device

            self._pipe = pipeline(**kwargs)
        return self._pipe

    def generate(self, query: Query, artifacts: List[Artifact]) -> Answer:
        context = self._build_context(artifacts)
        prompt = _PROMPT_TEMPLATE.format(context=context, question=query.text)

        try:
            pipe = self._get_pipe()
            result = pipe(prompt)
            if result and isinstance(result, list):
                answer_text = result[0].get("generated_text", "").strip()
            else:
                answer_text = str(result).strip()
        except Exception as exc:
            logger.error("Local generator error: %s", exc)
            answer_text = f"[Generation failed: {exc}]"

        sources = [
            Evidence(artifact_id=a.id, text=a.content[:300], confidence=1.0)
            for a in artifacts[:5]
        ]

        return Answer(
            text=answer_text,
            confidence=1.0,
            sources=sources,
            artifacts=[a.id for a in artifacts],
        )

    @staticmethod
    def _build_context(artifacts: List[Artifact]) -> str:
        parts: List[str] = []
        for a in artifacts:
            page = a.provenance.page_number
            if a.artifact_type == "text":
                parts.append(f"[Page {page}]\n{a.content}")
            elif a.artifact_type in ("figure", "chart"):
                caption = a.caption or a.content or "No caption."
                parts.append(f"[Page {page} – figure] {caption}")
            elif a.artifact_type == "table":
                parts.append(f"[Page {page} – table]\n{a.content}")
        return "\n\n---\n\n".join(parts)
