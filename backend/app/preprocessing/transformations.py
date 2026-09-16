from typing import Any

import numpy as np
import pandas as pd

from app.preprocessing.base_transformer import BaseTransformer


class LogTransformer(BaseTransformer):
    """Apply a log1p transformation to numerical columns."""

    name = "log_transformation"

    def __init__(
        self,
        columns: list[str] | None = None,
    ) -> None:
        self.columns = columns
        self.fitted_columns: list[str] = []

    def fit(self, df: pd.DataFrame) -> "LogTransformer":
        """Identify numerical columns suitable for transformation."""
        if self.columns is None:
            columns = df.select_dtypes(include="number").columns.tolist()
        else:
            columns = [column for column in self.columns if column in df.columns]

        self.fitted_columns = columns

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log1p transformation."""
        transformed = df.copy()

        for column in self.fitted_columns:
            if column not in transformed.columns:
                continue

            if (transformed[column] < 0).any():
                raise ValueError(
                    f"Column '{column}' contains negative values. "
                    "Log transformation cannot be applied safely."
                )

            transformed[column] = np.log1p(transformed[column])

        return transformed

    def get_config(self) -> dict[str, Any]:
        """Return transformation configuration."""
        return {
            "name": self.name,
            "columns": self.columns,
        }
