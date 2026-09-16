import pandas as pd

from app.detection.base_detector import BaseDetector
from app.schemas.issues import DataQualityIssue


class CardinalityDetector(BaseDetector):
    """Detect categorical columns with high cardinality."""

    name = "high_cardinality"

    def detect(self, df: pd.DataFrame) -> list[DataQualityIssue]:
        """Detect categorical columns with a high uniqueness ratio."""
        issues = []

        categorical_columns = df.select_dtypes(include=["object", "category"]).columns

        for column in categorical_columns:
            series = df[column].dropna()

            if series.empty:
                continue

            unique_count = int(series.nunique())
            total_count = len(series)

            uniqueness_ratio = unique_count / total_count

            if uniqueness_ratio < 0.5:
                continue

            if unique_count >= 100:
                severity = "high"
            elif uniqueness_ratio >= 0.8:
                severity = "medium"
            else:
                severity = "low"

            issues.append(
                DataQualityIssue(
                    issue_type="high_cardinality",
                    severity=severity,
                    description=(
                        f"Column '{column}' has high cardinality: "
                        f"{unique_count} unique values out of "
                        f"{total_count} non-null values."
                    ),
                    column=str(column),
                    affected_count=unique_count,
                    metadata={
                        "unique_count": unique_count,
                        "uniqueness_ratio": float(uniqueness_ratio),
                        "non_null_count": total_count,
                    },
                )
            )

        return issues
