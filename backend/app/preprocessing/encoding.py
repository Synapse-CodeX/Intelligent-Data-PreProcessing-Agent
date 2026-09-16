from typing import Any

import pandas as pd

from app.preprocessing.base_transformer import BaseTransformer


class CategoricalEncoder(BaseTransformer):
    """Encode categorical columns using one-hot encoding."""

    name = "categorical_encoding"

    def __init__(
        self,
        columns: list[str] | None = None,
        drop_first: bool = True,
    ) -> None:
        self.columns = columns
        self.drop_first = drop_first
        self.fitted_columns: list[str] = []
        self.output_columns: list[str] = []

    def fit(self, df: pd.DataFrame) -> "CategoricalEncoder":
        """Identify categorical columns for encoding."""
        if self.columns is None:
            columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            columns = [column for column in self.columns if column in df.columns]

        self.fitted_columns = columns

        transformed = pd.get_dummies(
            df,
            columns=self.fitted_columns,
            drop_first=self.drop_first,
            dtype=int,
        )

        self.output_columns = transformed.columns.tolist()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply one-hot encoding."""
        transformed = pd.get_dummies(
            df,
            columns=self.fitted_columns,
            drop_first=self.drop_first,
            dtype=int,
        )

        transformed = transformed.reindex(
            columns=self.output_columns,
            fill_value=0,
        )

        return transformed

    def get_config(self) -> dict[str, Any]:
        """Return encoding configuration."""
        return {
            "name": self.name,
            "columns": self.columns,
            "drop_first": self.drop_first,
        }
