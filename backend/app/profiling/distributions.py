from typing import Any

import pandas as pd


def analyze_distributions(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Analyze distribution characteristics of numerical columns."""
    distributions = {}

    numeric_columns = df.select_dtypes(include="number").columns

    for column in numeric_columns:
        series = df[column].dropna()

        distributions[column] = {
            "skewness": float(series.skew()),
            "kurtosis": float(series.kurtosis()),
        }

    return distributions
