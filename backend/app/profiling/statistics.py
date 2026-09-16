from typing import Any

import pandas as pd


def calculate_statistics(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Calculate descriptive statistics for numerical columns."""
    statistics = {}

    numeric_columns = df.select_dtypes(include="number").columns

    for column in numeric_columns:
        series = df[column]

        statistics[column] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "q1": float(series.quantile(0.25)),
            "q3": float(series.quantile(0.75)),
        }

    return statistics
