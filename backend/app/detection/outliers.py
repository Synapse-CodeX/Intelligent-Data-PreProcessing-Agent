import pandas as pd

from app.detection.base_detector import BaseDetector
from app.schemas.issues import DataQualityIssue


class OutlierDetector(BaseDetector):
    """Detect statistical outliers in numerical columns."""

    name = "outliers"

    def detect(self, df: pd.DataFrame) -> list[DataQualityIssue]:
        """Detect outliers using the IQR method."""
        issues = []

        numeric_columns = df.select_dtypes(include="number").columns

        for column in numeric_columns:
            series = df[column].dropna()

            if series.empty:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_count = int(outlier_mask.sum())

            if outlier_count == 0:
                continue

            outlier_percentage = float(outlier_count / len(series) * 100)

            if outlier_percentage >= 20:
                severity = "high"
            elif outlier_percentage >= 5:
                severity = "medium"
            else:
                severity = "low"

            issues.append(
                DataQualityIssue(
                    issue_type="outliers",
                    severity=severity,
                    description=(
                        f"Column '{column}' contains "
                        f"{outlier_count} potential outliers "
                        f"using the IQR method."
                    ),
                    column=str(column),
                    affected_count=outlier_count,
                    metadata={
                        "method": "IQR",
                        "q1": float(q1),
                        "q3": float(q3),
                        "iqr": float(iqr),
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound),
                        "outlier_percentage": outlier_percentage,
                    },
                )
            )

        return issues
