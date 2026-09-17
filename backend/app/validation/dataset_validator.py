from typing import Any

import pandas as pd


class DatasetValidator:
    """Validate basic dataset properties."""

    def validate(
        self,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Validate a dataset and return validation results."""
        checks: dict[str, bool] = {}

        checks["dataset_exists"] = df is not None
        checks["dataset_not_empty"] = df is not None and not df.empty
        checks["has_columns"] = df is not None and len(df.columns) > 0
        checks["has_rows"] = df is not None and len(df) > 0
        checks["column_names_unique"] = (
            df is not None and not df.columns.duplicated().any()
        )

        passed = all(checks.values())

        return {
            "validator": "dataset",
            "passed": passed,
            "checks": checks,
        }
