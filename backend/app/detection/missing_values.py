import pandas as pd

from app.detection.base_detector import BaseDetector
from app.schemas.issues import DataQualityIssue


class MissingValueDetector(BaseDetector):
    """Detect missing values in dataset columns."""

    name = "missing_values"

    def detect(self, df: pd.DataFrame) -> list[DataQualityIssue]:
        """Detect columns containing missing values."""
        issues = []

        for column in df.columns:
            missing_count = int(df[column].isna().sum())

            if missing_count == 0:
                continue

            missing_percentage = float(df[column].isna().mean() * 100)

            if missing_percentage >= 50:
                severity = "high"
            elif missing_percentage >= 20:
                severity = "medium"
            else:
                severity = "low"

            issues.append(
                DataQualityIssue(
                    issue_type="missing_values",
                    severity=severity,
                    description=(
                        f"Column '{column}' contains "
                        f"{missing_count} missing values "
                        f"({missing_percentage:.2f}%)."
                    ),
                    column=str(column),
                    affected_count=missing_count,
                    metadata={
                        "missing_percentage": missing_percentage,
                    },
                )
            )

        return issues
