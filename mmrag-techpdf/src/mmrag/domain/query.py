from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field

from .artifact import Evidence


class QueryIntent(Enum):
    """Intent of the user query."""
    SEARCH = "search"
    QA = "qa"  # Question Answering
    SUMMARIZE = "summarize"
    EXTRACT = "extract"


class Query(BaseModel):
    """A user query."""
    text: str
    intent: QueryIntent = QueryIntent.QA
    filters: dict = Field(default_factory=dict)  # e.g., {"document": "path"}


class Answer(BaseModel):
    """An answer to a query."""
    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: List[Evidence] = Field(default_factory=list)
    artifacts: List[str] = Field(default_factory=list)  # Artifact IDs