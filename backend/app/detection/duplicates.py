import pandas as pd

from app.detection.base_detector import BaseDetector
from app.schemas.issues import DataQualityIssue


class DuplicateDetector(BaseDetector):
    """Detect duplicate rows in a dataset."""

    name = "duplicates"

    def detect(self, df: pd.DataFrame) -> list[DataQualityIssue]:
        """Detect duplicate rows."""
        duplicate_count = int(df.duplicated().sum())

        if duplicate_count == 0:
            return []

        duplicate_percentage = float(duplicate_count / len(df) * 100)

        if duplicate_percentage >= 20:
            severity = "high"
        elif duplicate_percentage >= 5:
            severity = "medium"
        else:
            severity = "low"

        return [
            DataQualityIssue(
                issue_type="duplicate_rows",
                severity=severity,
                description=(
                    f"Dataset contains {duplicate_count} duplicate rows "
                    f"({duplicate_percentage:.2f}%)."
                ),
                affected_count=duplicate_count,
                metadata={
                    "duplicate_percentage": duplicate_percentage,
                },
            )
        ]
