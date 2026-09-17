from typing import Any

import pandas as pd


class PreprocessingValidator:
    """Validate the results of preprocessing."""

    def validate(
        self,
        original_df: pd.DataFrame,
        processed_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Validate basic preprocessing invariants."""
        checks: dict[str, bool] = {}

        checks["dataset_not_empty"] = not processed_df.empty

        checks["row_count_preserved"] = len(original_df) == len(processed_df)

        checks["no_missing_values"] = int(processed_df.isna().sum().sum()) == 0

        checks["column_names_unique"] = not processed_df.columns.duplicated().any()

        checks["no_infinite_values"] = not self._contains_infinite_values(processed_df)

        return {
            "validator": "preprocessing",
            "passed": all(checks.values()),
            "checks": checks,
        }

    @staticmethod
    def _contains_infinite_values(
        df: pd.DataFrame,
    ) -> bool:
        """Check whether numerical columns contain infinity."""
        numeric_df = df.select_dtypes(include="number")

        if numeric_df.empty:
            return False

        return bool(numeric_df.isin([float("inf"), float("-inf")]).any().any())
