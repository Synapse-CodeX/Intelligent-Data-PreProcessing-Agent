import pandas as pd

from app.detection.base_detector import BaseDetector
from app.schemas.issues import DataQualityIssue


class DateTimeDetector(BaseDetector):
    """Detect columns that appear to contain datetime values."""

    name = "datetime"

    def detect(self, df: pd.DataFrame) -> list[DataQualityIssue]:
        """Detect string columns that can potentially be parsed as datetimes."""
        issues = []

        object_columns = df.select_dtypes(include="object").columns

        for column in object_columns:
            series = df[column].dropna()

            if series.empty:
                continue

            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            parse_ratio = float(parsed.notna().mean())

            if parse_ratio < 0.8:
                continue

            issues.append(
                DataQualityIssue(
                    issue_type="datetime_candidate",
                    severity="low",
                    description=(
                        f"Column '{column}' appears to contain datetime values."
                    ),
                    column=str(column),
                    affected_count=int(parsed.notna().sum()),
                    metadata={
                        "parse_ratio": parse_ratio,
                    },
                )
            )

        return issues
