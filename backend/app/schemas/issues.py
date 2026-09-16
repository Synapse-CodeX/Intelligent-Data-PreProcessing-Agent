from typing import Any

from pydantic import BaseModel, Field


class DataQualityIssue(BaseModel):
    """Structured representation of a detected data-quality issue."""

    issue_type: str
    severity: str
    description: str
    column: str | None = None
    affected_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
