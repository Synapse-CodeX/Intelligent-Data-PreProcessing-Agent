import pandas as pd

from app.detection.base_detector import BaseDetector
from app.schemas.issues import DataQualityIssue


class SkewnessDetector(BaseDetector):
    """Detect highly skewed numerical features."""

    name = "skewness"

    def detect(self, df: pd.DataFrame) -> list[DataQualityIssue]:
        """Detect numerical columns with high absolute skewness."""
        issues = []

        numeric_columns = df.select_dtypes(include="number").columns

        for column in numeric_columns:
            series = df[column].dropna()

            if len(series) < 3:
                continue

            skewness = float(series.skew())
            absolute_skewness = abs(skewness)

            if absolute_skewness < 1:
                continue

            if absolute_skewness >= 2:
                severity = "high"
            else:
                severity = "medium"

            issues.append(
                DataQualityIssue(
                    issue_type="skewness",
                    severity=severity,
                    description=(
                        f"Column '{column}' has high skewness ({skewness:.2f})."
                    ),
                    column=str(column),
                    metadata={
                        "skewness": skewness,
                        "absolute_skewness": absolute_skewness,
                    },
                )
            )

        return issues
