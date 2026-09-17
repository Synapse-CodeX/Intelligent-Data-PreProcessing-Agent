from typing import Any

from pydantic import BaseModel, Field

from app.schemas.issues import DataQualityIssue


class DatasetContext(BaseModel):
    """Contextual information about the dataset."""

    name: str | None = None
    rows: int = 0
    columns: int = 0
    column_names: list[str] = Field(default_factory=list)


class QualityContext(BaseModel):
    """Dataset quality assessment produced by the quality layer."""

    score: int = 100
    max_score: int = 100
    quality_level: str = "excellent"
    issue_count: int = 0
    severity_counts: dict[str, int] = Field(
        default_factory=lambda: {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }
    )


class AgentAction(BaseModel):
    """A preprocessing action proposed by the agent."""

    action_type: str
    column: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    approved: bool = False


class AgentState(BaseModel):
    """Shared state carried through the intelligent preprocessing workflow."""

    dataset: DatasetContext = Field(default_factory=DatasetContext)

    issues: list[DataQualityIssue] = Field(default_factory=list)

    quality: QualityContext = Field(default_factory=QualityContext)

    actions: list[AgentAction] = Field(default_factory=list)

    validation_result: dict[str, Any] = Field(default_factory=dict)

    execution_result: dict[str, Any] = Field(default_factory=dict)

    messages: list[str] = Field(default_factory=list)

    status: str = "initialized"
