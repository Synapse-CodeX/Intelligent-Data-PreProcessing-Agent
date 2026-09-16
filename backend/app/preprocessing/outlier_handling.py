from typing import Any

import pandas as pd

from app.preprocessing.base_transformer import BaseTransformer


class IQRWinsorizer(BaseTransformer):
    """Clip numerical outliers using IQR boundaries."""

    name = "iqr_winsorization"

    def __init__(
        self,
        columns: list[str] | None = None,
        multiplier: float = 1.5,
    ) -> None:
        self.columns = columns
        self.multiplier = multiplier
        self.bounds: dict[str, tuple[float, float]] = {}

    def fit(self, df: pd.DataFrame) -> "IQRWinsorizer":
        """Calculate IQR-based clipping boundaries."""
        if self.columns is None:
            columns = df.select_dtypes(include="number").columns.tolist()
        else:
            columns = [column for column in self.columns if column in df.columns]

        for column in columns:
            series = df[column].dropna()

            if series.empty:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower_bound = q1 - self.multiplier * iqr
            upper_bound = q3 + self.multiplier * iqr

            self.bounds[column] = (
                float(lower_bound),
                float(upper_bound),
            )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip values outside learned IQR boundaries."""
        transformed = df.copy()

        for column, (lower_bound, upper_bound) in self.bounds.items():
            if column not in transformed.columns:
                continue

            transformed[column] = transformed[column].clip(
                lower=lower_bound,
                upper=upper_bound,
            )

        return transformed

    def get_config(self) -> dict[str, Any]:
        """Return outlier-handling configuration."""
        return {
            "name": self.name,
            "columns": self.columns,
            "multiplier": self.multiplier,
        }
