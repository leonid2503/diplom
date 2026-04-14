from typing import Optional
from pydantic import BaseModel, Field


class Block(BaseModel):
    """A content block extracted from a PDF page."""
    text: str = ""
    bbox: tuple[float, float, float, float] = Field(..., description="Bounding box as (x0, y0, x1, y1)")
    block_type: str = Field(..., description="Type of block: 'text', 'image', 'table', etc.")
    confidence: Optional[float] = None


class Page(BaseModel):
    """A single page from a PDF document."""
    number: int = Field(..., ge=1)
    blocks: list[Block] = Field(default_factory=list)
    image_path: Optional[str] = None  # Path to rendered page image


class Document(BaseModel):
    """A PDF document with extracted content."""
    title: str
    pages: list[Page] = Field(default_factory=list)
    file_path: str
    metadata: dict = Field(default_factory=dict)