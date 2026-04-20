from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths ---
    data_dir: Path = Path("data")
    index_dir: Path = Path("data/indexes")
    figures_dir: Path = Path("data/figures")
    processed_dir: Path = Path("data/processed")

    # --- Embedding ---
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Captioning ---
    # Use "Salesforce/blip-image-captioning-base" for lighter GPU/CPU usage
    # Use "Salesforce/blip2-opt-2.7b" for higher quality (requires GPU + ~8 GB VRAM)
    captioner_model: str = "Salesforce/blip-image-captioning-base"

    # --- Retrieval ---
    retrieval_top_k: int = 20          # candidates before reranking
    rerank_top_k: int = 5              # final artifacts sent to generator
    bm25_weight: float = 0.4           # weight for BM25 in hybrid fusion
    dense_weight: float = 0.6          # weight for dense (FAISS) in hybrid fusion

    # --- Cross-encoder reranker ---
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- LLM generator ---
    generator_backend: str = "openai"  # "openai" | "local"
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = "gpt-4o-mini"

    # Local HuggingFace generator (used when generator_backend="local")
    local_llm_model: str = "microsoft/phi-2"
    local_llm_max_new_tokens: int = 512

    # --- Table extraction ---
    table_backend: str = "pymupdf"     # "pymupdf" | "camelot"

    # --- Chunking ---
    chunk_size: int = 400              # approximate word count per text chunk
    chunk_overlap: int = 50            # word overlap between consecutive chunks

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
