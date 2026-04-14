from typing import Optional
from pydantic import BaseModel, Field


class Provenance(BaseModel):
    """Provenance information for an extracted artifact."""
    document_path: str
    page_number: int = Field(..., ge=1)
    bbox: tuple[float, float, float, float] = Field(..., description="Bounding box as (x0, y0, x1, y1)")


class Artifact(BaseModel):
    """An extracted artifact from a document (figure, table, chart, etc.)."""
    id: str
    artifact_type: str = Field(..., description="Type: 'figure', 'table', 'chart', etc.")
    content: str = Field(..., description="Extracted text or description")
    image_path: Optional[str] = None  # Path to cropped image
    caption: Optional[str] = None
    provenance: Provenance
    confidence: Optional[float] = None


class Evidence(BaseModel):
    """Supporting evidence for an artifact or answer."""
    artifact_id: str
    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)