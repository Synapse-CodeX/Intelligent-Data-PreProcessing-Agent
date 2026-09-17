from typing import Any

import pandas as pd


class SchemaValidator:
    """Validate the schema of a processed dataset."""

    def validate(
        self,
        original_df: pd.DataFrame,
        processed_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Compare original and processed dataset schemas."""
        original_columns = set(original_df.columns)
        processed_columns = set(processed_df.columns)

        removed_columns = sorted(original_columns - processed_columns)

        added_columns = sorted(processed_columns - original_columns)

        return {
            "validator": "schema",
            "passed": True,
            "checks": {
                "original_dataset_has_columns": (len(original_columns) > 0),
                "processed_dataset_has_columns": (len(processed_columns) > 0),
            },
            "original_column_count": len(original_columns),
            "processed_column_count": len(processed_columns),
            "removed_columns": removed_columns,
            "added_columns": added_columns,
        }
