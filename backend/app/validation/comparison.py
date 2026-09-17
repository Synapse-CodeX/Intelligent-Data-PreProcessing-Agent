from typing import Any

import pandas as pd


class DatasetComparison:
    """Compare datasets before and after preprocessing."""

    def compare(
        self,
        original_df: pd.DataFrame,
        processed_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Generate a before/after dataset comparison."""
        original_missing = int(original_df.isna().sum().sum())

        processed_missing = int(processed_df.isna().sum().sum())

        original_duplicates = int(original_df.duplicated().sum())

        processed_duplicates = int(processed_df.duplicated().sum())

        return {
            "original": {
                "rows": len(original_df),
                "columns": len(original_df.columns),
                "missing_values": original_missing,
                "duplicate_rows": original_duplicates,
            },
            "processed": {
                "rows": len(processed_df),
                "columns": len(processed_df.columns),
                "missing_values": processed_missing,
                "duplicate_rows": processed_duplicates,
            },
            "changes": {
                "rows": len(processed_df) - len(original_df),
                "columns": (len(processed_df.columns) - len(original_df.columns)),
                "missing_values": (processed_missing - original_missing),
                "duplicate_rows": (processed_duplicates - original_duplicates),
            },
        }
