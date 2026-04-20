import base64
import logging
from pathlib import Path
from typing import List, Optional

from ...domain.artifact import Artifact, Evidence
from ...domain.query import Answer, Query
from ...ports.generator import Generator

logger = logging.getLogger(__name__)


class OpenAIGenerator(Generator):
    """
    Answer generator backed by the OpenAI Chat Completions API.

    Multimodal support
    ------------------
    When the retrieved artifacts include figures with saved image paths,
    those images are base64-encoded and appended to the user message so
    that vision-capable models (gpt-4o, gpt-4o-mini) can reason over them.

    Model recommendations
    ---------------------
    * ``gpt-4o-mini``  – fast, cheap, supports vision
    * ``gpt-4o``       – highest quality, also supports vision
    """

    SYSTEM_PROMPT = (
        "You are an expert assistant for technical and scientific papers. "
        "Answer the user's question solely based on the provided context. "
        "If the context does not contain enough information, say so explicitly. "
        "Be concise, precise, and cite the relevant page numbers when possible."
    )

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        from openai import OpenAI

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key)

    def generate(self, query: Query, artifacts: List[Artifact]) -> Answer:
        context_text, image_blocks = self._build_context(artifacts)

        user_content: list = [
            {
                "type": "text",
                "text": (
                    f"## Context\n\n{context_text}\n\n"
                    f"## Question\n\n{query.text}\n\n"
                    "Answer based solely on the context above."
                ),
            }
        ]
        user_content.extend(image_blocks)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            answer_text = response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("OpenAI API error: %s", exc)
            answer_text = f"[Generation failed: {exc}]"

        sources = [
            Evidence(
                artifact_id=a.id,
                text=a.content[:300],
                confidence=1.0,
            )
            for a in artifacts[:5]
        ]

        return Answer(
            text=answer_text.strip(),
            confidence=1.0,
            sources=sources,
            artifacts=[a.id for a in artifacts],
        )

    def _build_context(self, artifacts: List[Artifact]):
        """Build a text context string and a list of OpenAI image content blocks."""
        text_parts: List[str] = []
        image_blocks: List[dict] = []

        for a in artifacts:
            page = a.provenance.page_number
            if a.artifact_type == "text":
                text_parts.append(f"[Page {page} – text]\n{a.content}")

            elif a.artifact_type in ("figure", "chart"):
                caption = a.caption or a.content or "No caption available."
                text_parts.append(f"[Page {page} – figure] Caption: {caption}")
                if a.image_path and Path(a.image_path).exists():
                    image_blocks.append(self._encode_image_block(a.image_path))

            elif a.artifact_type == "table":
                text_parts.append(f"[Page {page} – table]\n{a.content}")

        return "\n\n".join(text_parts), image_blocks

    @staticmethod
    def _encode_image_block(image_path: str) -> dict:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        suffix = Path(image_path).suffix.lstrip(".").lower()
        mime = "image/png" if suffix == "png" else f"image/{suffix}"
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "auto"},
        }
