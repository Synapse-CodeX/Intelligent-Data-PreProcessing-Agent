from typing import Any

import pandas as pd

from app.preprocessing.base_transformer import BaseTransformer


class NumericInteractionTransformer(BaseTransformer):
    """Create interaction features between numerical columns."""

    name = "numeric_interactions"

    def __init__(
        self,
        columns: list[str],
    ) -> None:
        self.columns = columns
        self.fitted_columns: list[str] = []

    def fit(self, df: pd.DataFrame) -> "NumericInteractionTransformer":
        """Validate numerical columns."""
        self.fitted_columns = [
            column
            for column in self.columns
            if column in df.columns and pd.api.types.is_numeric_dtype(df[column])
        ]

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create pairwise multiplication features."""
        transformed = df.copy()

        for index, first_column in enumerate(self.fitted_columns):
            for second_column in self.fitted_columns[index + 1 :]:
                feature_name = f"{first_column}_x_{second_column}"

                transformed[feature_name] = (
                    transformed[first_column] * transformed[second_column]
                )

        return transformed

    def get_config(self) -> dict[str, Any]:
        """Return interaction transformer configuration."""
        return {
            "name": self.name,
            "columns": self.columns,
        }
