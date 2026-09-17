from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from app.detection.detector_engine import DetectorEngine
from app.services.quality_service import QualityService


class AgentTool(ABC):
    """Base interface for tools available to the agent."""

    name: str
    description: str

    @abstractmethod
    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool."""
        raise NotImplementedError


class DatasetInspectionTool(AgentTool):
    """Inspect the basic structure of a dataset."""

    name = "inspect_dataset"

    description = (
        "Inspect dataset dimensions, columns, data types, "
        "missing values, and duplicate rows."
    )

    def run(
        self,
        df: pd.DataFrame,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return structured information about the dataset."""
        if df.empty:
            return {
                "success": False,
                "error": "Dataset is empty.",
            }

        return {
            "success": True,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
            "missing_values": {
                column: int(count)
                for column, count in df.isna().sum().items()
                if count > 0
            },
            "duplicate_rows": int(df.duplicated().sum()),
        }


class DetectionTool(AgentTool):
    """Run the project's deterministic detection engine."""

    name = "detect_issues"

    description = (
        "Detect data-quality issues using the registered data-quality detectors."
    )

    def __init__(
        self,
        detector_engine: DetectorEngine | None = None,
    ) -> None:
        self.detector_engine = detector_engine or DetectorEngine()

    def run(
        self,
        df: pd.DataFrame,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run all registered detectors."""
        if df.empty:
            return {
                "success": False,
                "error": "Dataset is empty.",
                "issues": [],
            }

        issues = self.detector_engine.detect(df)

        return {
            "success": True,
            "issue_count": len(issues),
            "issues": [issue.model_dump() for issue in issues],
        }


class QualityAnalysisTool(AgentTool):
    """Analyze detected issues and calculate dataset quality."""

    name = "analyze_quality"

    description = "Calculate the dataset quality score and prioritize detected issues."

    def __init__(
        self,
        quality_service: QualityService | None = None,
    ) -> None:
        self.quality_service = quality_service or QualityService()

    def run(
        self,
        issues: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Calculate quality information from detected issues."""
        result = self.quality_service.analyze(issues)

        return {
            "success": True,
            "quality": result["quality"],
            "issues": result["issues"],
        }
