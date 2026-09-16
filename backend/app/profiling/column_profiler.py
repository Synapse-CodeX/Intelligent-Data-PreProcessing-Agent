from typing import Any

import pandas as pd


def profile_columns(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Generate basic profiling information for each dataset column."""
    profiles = {}

    for column in df.columns:
        series = df[column]

        profiles[column] = {
            "data_type": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_percentage": float(series.isna().mean() * 100),
            "unique_count": int(series.nunique(dropna=True)),
        }

    return profiles
