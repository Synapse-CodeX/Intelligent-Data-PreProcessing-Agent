from typing import Any

import pandas as pd

from app.preprocessing.base_transformer import BaseTransformer


class ConstantFeatureRemover(BaseTransformer):
    """Remove features containing no variation."""

    name = "constant_feature_removal"

    def __init__(self) -> None:
        self.removed_columns: list[str] = []

    def fit(self, df: pd.DataFrame) -> "ConstantFeatureRemover":
        """Identify constant columns."""
        self.removed_columns = [
            str(column)
            for column in df.columns
            if df[column].nunique(dropna=False) <= 1
        ]

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove constant columns."""
        return df.drop(
            columns=self.removed_columns,
            errors="ignore",
        )

    def get_config(self) -> dict[str, Any]:
        """Return feature-selection configuration."""
        return {
            "name": self.name,
        }
