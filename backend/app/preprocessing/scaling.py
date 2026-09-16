from typing import Any

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from app.preprocessing.base_transformer import BaseTransformer


class NumericalScaler(BaseTransformer):
    """Scale numerical columns."""

    name = "numerical_scaling"

    def __init__(
        self,
        columns: list[str] | None = None,
        method: str = "standard",
    ) -> None:
        self.columns = columns
        self.method = method
        self.scaler: StandardScaler | MinMaxScaler | None = None
        self.fitted_columns: list[str] = []

    def fit(self, df: pd.DataFrame) -> "NumericalScaler":
        """Fit the selected scaler."""
        if self.columns is None:
            columns = df.select_dtypes(include="number").columns.tolist()
        else:
            columns = [column for column in self.columns if column in df.columns]

        if not columns:
            self.fitted_columns = []
            return self

        if self.method == "standard":
            scaler = StandardScaler()
        elif self.method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError("method must be 'standard' or 'minmax'.")

        scaler.fit(df[columns])

        self.scaler = scaler
        self.fitted_columns = columns

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted scaler."""
        transformed = df.copy()

        if not self.fitted_columns or self.scaler is None:
            return transformed

        transformed[self.fitted_columns] = self.scaler.transform(
            transformed[self.fitted_columns]
        )

        return transformed

    def get_config(self) -> dict[str, Any]:
        """Return scaling configuration."""
        return {
            "name": self.name,
            "columns": self.columns,
            "method": self.method,
        }
