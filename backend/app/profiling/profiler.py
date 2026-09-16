from typing import Any

import pandas as pd

from app.profiling.column_profiler import profile_columns
from app.profiling.correlations import calculate_correlations
from app.profiling.data_types import (
    infer_data_types,
    infer_semantic_types,
)
from app.profiling.distributions import analyze_distributions
from app.profiling.statistics import calculate_statistics


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Generate a comprehensive profile of a dataset."""
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "data_types": infer_data_types(df),
        "semantic_types": infer_semantic_types(df),
        "column_profiles": profile_columns(df),
        "statistics": calculate_statistics(df),
        "distributions": analyze_distributions(df),
        "correlations": calculate_correlations(df),
    }
