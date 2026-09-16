import pandas as pd

from app.detection.base_detector import BaseDetector
from app.schemas.issues import DataQualityIssue


class ConstantFeatureDetector(BaseDetector):
    """Detect columns containing only one unique value."""

    name = "constant_features"

    def detect(self, df: pd.DataFrame) -> list[DataQualityIssue]:
        """Detect constant columns."""
        issues = []

        for column in df.columns:
            unique_count = int(df[column].nunique(dropna=False))

            if unique_count != 1:
                continue

            value = df[column].iloc[0]

            issues.append(
                DataQualityIssue(
                    issue_type="constant_feature",
                    severity="medium",
                    description=(
                        f"Column '{column}' contains only one unique value "
                        "and provides no variation."
                    ),
                    column=str(column),
                    affected_count=len(df),
                    metadata={
                        "unique_count": unique_count,
                        "constant_value": str(value),
                    },
                )
            )

        return issues
