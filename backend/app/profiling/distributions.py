from typing import Any

import pandas as pd

from app.profiling.data_types import infer_semantic_types


def analyze_distributions(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Analyze distribution characteristics of numerical columns."""
    distributions: dict[str, dict[str, Any]] = {}

    semantic_types = infer_semantic_types(df)

    for column in df.columns:
        if semantic_types[column] != "numeric":
            continue

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if len(series) < 3:
            continue

        distributions[str(column)] = {
            "skewness": float(series.skew()),
            "kurtosis": float(series.kurtosis()),
        }

    return distributions
