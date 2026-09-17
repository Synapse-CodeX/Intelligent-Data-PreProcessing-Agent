from typing import Any

import pandas as pd


class LeakageValidator:
    """Validate datasets for obvious target leakage."""

    def validate(
        self,
        df: pd.DataFrame,
        target_column: str | None = None,
    ) -> dict[str, Any]:
        """Check for features identical to the target."""
        if target_column is None:
            return {
                "validator": "leakage",
                "passed": True,
                "checks": {
                    "target_column_provided": False,
                },
                "issues": [],
            }

        if target_column not in df.columns:
            return {
                "validator": "leakage",
                "passed": False,
                "checks": {
                    "target_column_exists": False,
                },
                "issues": [f"Target column '{target_column}' does not exist."],
            }

        target = df[target_column]
        issues = []

        for column in df.columns:
            if column == target_column:
                continue

            if df[column].equals(target):
                issues.append(
                    {
                        "column": column,
                        "reason": "feature_identical_to_target",
                    }
                )

        return {
            "validator": "leakage",
            "passed": len(issues) == 0,
            "checks": {
                "target_column_exists": True,
                "no_identical_target_features": len(issues) == 0,
            },
            "issues": issues,
        }
