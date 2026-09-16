import pandas as pd

from app.detection.base_detector import BaseDetector
from app.schemas.issues import DataQualityIssue


class LeakageDetector(BaseDetector):
    """Detect potential target leakage in dataset features."""

    name = "leakage"

    def __init__(self, target_column: str | None = None) -> None:
        self.target_column = target_column

    def detect(self, df: pd.DataFrame) -> list[DataQualityIssue]:
        """Detect features identical to the target column."""
        if self.target_column is None:
            return []

        if self.target_column not in df.columns:
            return []

        issues = []

        target = df[self.target_column]

        for column in df.columns:
            if column == self.target_column:
                continue

            if df[column].equals(target):
                issues.append(
                    DataQualityIssue(
                        issue_type="potential_target_leakage",
                        severity="critical",
                        description=(
                            f"Column '{column}' is identical to the target "
                            f"column '{self.target_column}'."
                        ),
                        column=str(column),
                        affected_count=len(df),
                        metadata={
                            "target_column": self.target_column,
                            "reason": "feature_identical_to_target",
                        },
                    )
                )

        return issues
