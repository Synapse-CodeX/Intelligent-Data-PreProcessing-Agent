from typing import Any

import pandas as pd

from app.preprocessing.base_transformer import BaseTransformer


class MissingValueImputer(BaseTransformer):
    """Impute missing values in numerical and categorical columns."""

    name = "missing_value_imputation"

    def __init__(
        self,
        numerical_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
    ) -> None:
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy

        self.numerical_fill_values: dict[str, Any] = {}
        self.categorical_fill_values: dict[str, Any] = {}

    def fit(self, df: pd.DataFrame) -> "MissingValueImputer":
        """Learn replacement values from the dataset."""
        numeric_columns = df.select_dtypes(include="number").columns

        for column in numeric_columns:
            series = df[column].dropna()

            if series.empty:
                continue

            if self.numerical_strategy == "mean":
                self.numerical_fill_values[column] = series.mean()
            elif self.numerical_strategy == "median":
                self.numerical_fill_values[column] = series.median()
            else:
                raise ValueError("numerical_strategy must be 'mean' or 'median'.")

        categorical_columns = df.select_dtypes(include=["object", "category"]).columns

        for column in categorical_columns:
            series = df[column].dropna()

            if series.empty:
                continue

            if self.categorical_strategy == "most_frequent":
                self.categorical_fill_values[column] = series.mode().iloc[0]
            elif self.categorical_strategy == "constant":
                self.categorical_fill_values[column] = "Unknown"
            else:
                raise ValueError(
                    "categorical_strategy must be 'most_frequent' or 'constant'."
                )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned replacement values."""
        transformed = df.copy()

        for column, value in self.numerical_fill_values.items():
            if column in transformed.columns:
                transformed[column] = transformed[column].fillna(value)

        for column, value in self.categorical_fill_values.items():
            if column in transformed.columns:
                transformed[column] = transformed[column].fillna(value)

        return transformed

    def get_config(self) -> dict[str, Any]:
        """Return imputation configuration."""
        return {
            "name": self.name,
            "numerical_strategy": self.numerical_strategy,
            "categorical_strategy": self.categorical_strategy,
        }
