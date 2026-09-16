from typing import Any

import pandas as pd

from app.profiling.data_types import infer_semantic_types


def calculate_statistics(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Calculate descriptive statistics for numerical columns."""
    statistics: dict[str, dict[str, Any]] = {}

    semantic_types = infer_semantic_types(df)

    for column in df.columns:
        if semantic_types[column] != "numeric":
            continue

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if series.empty:
            continue

        statistics[str(column)] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "q1": float(series.quantile(0.25)),
            "q3": float(series.quantile(0.75)),
        }

    return statistics
