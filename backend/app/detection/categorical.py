from collections import defaultdict

import pandas as pd

from app.detection.base_detector import BaseDetector
from app.schemas.issues import DataQualityIssue


class CategoricalConsistencyDetector(BaseDetector):
    """Detect inconsistent representations in categorical columns."""

    name = "categorical_consistency"

    def detect(self, df: pd.DataFrame) -> list[DataQualityIssue]:
        """Detect categorical values that differ only by casing or whitespace."""
        issues = []

        categorical_columns = df.select_dtypes(include=["object", "category"]).columns

        for column in categorical_columns:
            series = df[column].dropna()

            if series.empty:
                continue

            normalized_groups = defaultdict(list)

            for value in series:
                normalized_value = str(value).strip().lower()
                normalized_groups[normalized_value].append(str(value))

            inconsistent_groups = {
                normalized: sorted(set(values))
                for normalized, values in normalized_groups.items()
                if len(set(values)) > 1
            }

            if not inconsistent_groups:
                continue

            affected_count = sum(len(values) for values in inconsistent_groups.values())

            issues.append(
                DataQualityIssue(
                    issue_type="categorical_inconsistency",
                    severity="low",
                    description=(
                        f"Column '{column}' contains categorical values "
                        "with inconsistent casing or whitespace."
                    ),
                    column=str(column),
                    affected_count=affected_count,
                    metadata={
                        "inconsistent_groups": inconsistent_groups,
                    },
                )
            )

        return issues
